"""
Trained per-move sequence detector (thesis arm A3g, seed 0): the app's main
suspicion score.

A small BiGRU reads 17 engine-free features per ply, built from the frozen
rating model's own per-move outputs (both sides' rating estimates and the
attention weights) plus cheap board and clock facts, and pools per-ply
logits over the suspect side's plies with gated attention (multiple-instance
learning). Its output is a game-level score in [0, 1] for one side.

Everything here is ported from the thesis HPC code that trained and evaluated
the weights, not re-derived:

* feature construction: ``analysis/scripts/extract_detector_features_v2.py``
  (``stage_rating_cheap``, ``_cheap_features``, ``_traj_features``) and
  ``analysis/scripts/extra_arm_a3.py`` (``build_ply_matrix``, ``standardize``);
* model and pooling: ``extra_arm_a3.py`` (``make_model``, ``Batcher.get``,
  ``predict``).

Dtypes follow the originals step by step (float64 intermediates stored as
float32) so scores match the thesis run to about 1e-6, not merely 1e-4. The
standardization constants come from the training split and are recorded in
``src/models/detector_a3g_seed0.json`` alongside the provenance.

One deliberate mapping: the corpus scored a single suspect side per game
against its Maia band. The app scores each side in turn as the suspect, with
that side's resolved baseline rating (request, PGN header, or self-prediction
fallback) standing in for the band, which is exactly what the synthetic PGN
headers record.

The detector's per-move logits are not used for critical moves: in thesis
evaluation they localized substituted moves at or below chance.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chess
import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "detector_a3g_seed0.pt"
DEFAULT_PROVENANCE_PATH = MODELS_DIR / "detector_a3g_seed0.json"

# Identifies this score in cutoffs files so detector cutoffs are never applied
# to the computed score (or the reverse).
SCORE_ID = "detector_a3g_seed0"

FEATURES = [
    "r_hat_suspect", "r_hat_other", "dev_signed", "d_t", "alpha_n", "run_mean", "run_std",
    "first_diff", "second_diff", "is_capture", "is_check", "material_balance",
    "clock_remaining", "clock_delta", "clock_delta_missing", "is_suspect_move", "ply_pos",
]
MAX_PLIES = 100

# chess.PAWN..chess.KING, as in extract_detector_features_v2.PIECE_VAL.
_PIECE_VAL = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}

WHITE = 0
BLACK = 1


class MILGRU(nn.Module):
    """BiGRU + per-ply instance logits + gated-attention MIL pooling.

    Attribute names match ``extra_arm_a3.make_model`` so the thesis state dict
    loads with ``strict=True``.
    """

    def __init__(self, nin: int, hidden: int, dropout: float):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nin, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.inst = nn.Linear(2 * hidden, 1)
        self.att_v = nn.Linear(2 * hidden, hidden)
        self.att_u = nn.Linear(2 * hidden, hidden)
        self.att_w = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, pool_mask: torch.Tensor):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        h = self.inp(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        out = self.drop(out)
        s = self.inst(out).squeeze(-1)
        e = self.att_w(torch.tanh(self.att_v(out)) * torch.sigmoid(self.att_u(out))).squeeze(-1)
        e = e.masked_fill(~pool_mask, -1e4)
        a = torch.softmax(e, dim=1)
        return (a * s).sum(1) + self.bias, s, a


@dataclass
class Detector:
    model: MILGRU
    feat_mu: np.ndarray  # (17,) float32
    feat_sd: np.ndarray  # (17,) float32
    provenance: dict


def load_detector(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    provenance_path: Path = DEFAULT_PROVENANCE_PATH,
    device: torch.device | None = None,
) -> Detector:
    with Path(provenance_path).open(encoding="utf-8") as f:
        provenance = json.load(f)
    if provenance["features"] != FEATURES:
        raise ValueError("Detector provenance feature list does not match detector.FEATURES")
    cfg = provenance["config"]
    model = MILGRU(len(FEATURES), cfg["hidden"], cfg["dropout"])
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device or torch.device("cpu"))
    model.eval()
    std = provenance["standardization"]
    return Detector(
        model=model,
        feat_mu=np.asarray(std["mean"], dtype=np.float32),
        feat_sd=np.asarray(std["std"], dtype=np.float32),
        provenance=provenance,
    )


def _cheap_features(moves_uci: Sequence[str], suspect_idx: int):
    """is_capture, is_check, material_balance (suspect POV, before the move).

    Port of ``extract_detector_features_v2._cheap_features``. An illegal move
    stops the replay and leaves the remaining plies at zero, as there.
    """
    board = chess.Board()
    n = len(moves_uci)
    is_capture = np.zeros(n, dtype=np.float32)
    is_check = np.zeros(n, dtype=np.float32)
    material = np.zeros(n, dtype=np.float32)
    for t, uci in enumerate(moves_uci):
        bal = 0.0
        for pt, val in _PIECE_VAL.items():
            bal += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        material[t] = bal if suspect_idx == WHITE else -bal
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            mv = None
        if mv is None or mv not in board.legal_moves:
            break
        is_capture[t] = 1.0 if board.is_capture(mv) else 0.0
        is_check[t] = 1.0 if board.gives_check(mv) else 0.0
        board.push(mv)
    return is_capture, is_check, material


def _traj_features(series: np.ndarray):
    """Port of ``extract_detector_features_v2._traj_features``."""
    n = len(series)
    csum = np.cumsum(series)
    idx = np.arange(1, n + 1, dtype=np.float64)
    run_mean = csum / idx
    csum2 = np.cumsum(series ** 2)
    run_var = np.maximum(csum2 / idx - run_mean ** 2, 0.0)
    run_std = np.sqrt(run_var)
    first_diff = np.zeros(n, dtype=np.float64)
    first_diff[1:] = np.diff(series)
    second_diff = np.zeros(n, dtype=np.float64)
    second_diff[1:] = np.diff(first_diff)
    return run_mean, run_std, first_diff, second_diff


def build_features(
    per_move_preds: np.ndarray,
    attention: np.ndarray | None,
    moves_uci: Sequence[str],
    clock_seconds: Sequence[float],
    suspect_idx: int,
    baseline: float,
) -> np.ndarray:
    """Raw (unstandardized) (n_plies, 17) float32 feature matrix for one side.

    ``per_move_preds`` is the rating model's (n, 2) output on the Elo scale
    (float32, as the model produces it), ``attention`` its (n,) attention
    weights, ``clock_seconds`` the remaining clock per ply. Only the first
    ``MAX_PLIES`` plies are used, as in training.
    """
    preds = np.asarray(per_move_preds, dtype=np.float32)[:MAX_PLIES]
    n = preds.shape[0]
    if attention is None:
        attn = np.full(n, 1.0 / max(n, 1))
    else:
        attn = np.asarray(attention, dtype=np.float64)[:n]
    moves = list(moves_uci)[:n]

    rhat_s = preds[:n, suspect_idx].astype(np.float64)
    rhat_o = preds[:n, 1 - suspect_idx].astype(np.float64)
    d_t = np.abs(rhat_s - float(baseline))
    rmean, rstd, d1, d2 = _traj_features(rhat_s)
    cap, chk, mat = _cheap_features(moves, suspect_idx)
    is_sus = np.array([(t % 2) == suspect_idx for t in range(n)], dtype=np.float32)

    secs = np.asarray(list(clock_seconds)[:n], dtype=np.float64)
    clkrem = np.zeros(n, dtype=np.float64)
    clkrem[: secs.shape[0]] = secs
    clkdelta = np.full(n, np.nan, dtype=np.float64)
    for t in range(2, n):
        if clkrem[t - 2] > 0 and clkrem[t] > 0:
            clkdelta[t] = clkrem[t - 2] - clkrem[t]

    # Stored as float32 per-ply arrays first, exactly as rating_cheap.npz holds them.
    pp = {
        "r_hat_suspect": rhat_s.astype(np.float32),
        "r_hat_other": rhat_o.astype(np.float32),
        "alpha": attn.astype(np.float32),
        "d_t": d_t.astype(np.float32),
        "run_mean": rmean.astype(np.float32),
        "run_std": rstd.astype(np.float32),
        "first_diff": d1.astype(np.float32),
        "second_diff": d2.astype(np.float32),
        "is_capture": cap,
        "is_check": chk,
        "material_balance": mat,
        "clock_remaining": clkrem.astype(np.float32),
        "clock_delta": clkdelta.astype(np.float32),
        "is_suspect_move": is_sus,
        "ply_idx": np.arange(n, dtype=np.int32),
    }
    band = np.float32(baseline)
    nply = np.float32(n)
    cd = pp["clock_delta"].astype(np.float32)
    cols = {
        "r_hat_suspect": pp["r_hat_suspect"],
        "r_hat_other": pp["r_hat_other"],
        "dev_signed": pp["r_hat_suspect"] - band,
        "d_t": pp["d_t"],
        "alpha_n": pp["alpha"] * nply,
        "run_mean": pp["run_mean"],
        "run_std": pp["run_std"],
        "first_diff": pp["first_diff"],
        "second_diff": pp["second_diff"],
        "is_capture": pp["is_capture"],
        "is_check": pp["is_check"],
        "material_balance": pp["material_balance"],
        "clock_remaining": pp["clock_remaining"],
        "clock_delta": cd,
        "clock_delta_missing": (~np.isfinite(cd)).astype(np.float32),
        "is_suspect_move": pp["is_suspect_move"],
        "ply_pos": pp["ply_idx"] / 100.0,
    }
    return np.stack([np.asarray(cols[f], np.float32) for f in FEATURES], axis=1)


def standardize(raw: np.ndarray, feat_mu: np.ndarray, feat_sd: np.ndarray) -> np.ndarray:
    """Apply the training-split standardization; non-finite values become 0."""
    m = (raw - feat_mu) / feat_sd
    m[~np.isfinite(m)] = 0.0
    return m


def score_side(detector: Detector, raw_features: np.ndarray) -> float:
    """Game-level detector score in [0, 1] for the side whose plies are marked
    ``is_suspect_move`` in ``raw_features``."""
    x = standardize(raw_features, detector.feat_mu, detector.feat_sd)
    n = x.shape[0]
    if n == 0:
        raise ValueError("Cannot score a game with no plies")
    device = next(detector.model.parameters()).device
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    pool = torch.from_numpy(raw_features[:, FEATURES.index("is_suspect_move")] > 0.5).unsqueeze(0).to(device)
    if not bool(pool.any()):
        pool = torch.ones_like(pool)  # as Batcher.get: no suspect ply, pool all
    lengths = torch.tensor([n], dtype=torch.int64)
    with torch.no_grad():
        logit, _, _ = detector.model(xt, lengths, pool)
    score = float(torch.sigmoid(logit).item())
    if not math.isfinite(score):
        raise ValueError("Detector produced a non-finite score")
    return score


def score_game(
    detector: Detector,
    per_move_preds: np.ndarray,
    attention: np.ndarray | None,
    moves_uci: Sequence[str],
    clock_seconds: Sequence[float],
    white_baseline: float,
    black_baseline: float,
) -> tuple[float, float]:
    """(white_score, black_score): each side scored in turn as the suspect."""
    scores = []
    for side, baseline in ((WHITE, white_baseline), (BLACK, black_baseline)):
        raw = build_features(per_move_preds, attention, moves_uci, clock_seconds, side, baseline)
        scores.append(score_side(detector, raw))
    return scores[0], scores[1]

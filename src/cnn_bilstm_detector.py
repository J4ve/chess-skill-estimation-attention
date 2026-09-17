"""
Full CNN-BiLSTM detector (thesis arm A4): one of the selectable suspicion scores.

Unlike the other trained detectors, this one reads the board positions
themselves rather than the rating model's outputs: the frozen rating model's
CNN trunk embeds each position, then a fine-tuned copy of its BiLSTM and
causal Bahdanau attention reads the embeddings plus the clock, and a
gated-attention MIL head pools per-ply logits over the suspect side's plies
into a game-level score in [0, 1].

Ported from the thesis HPC code, not re-derived:
``analysis/scripts/extra_arm_a4.py`` (``cnn_trunk``, ``stage_cache``,
``make_model``, ``Batcher.get``, ``predict``). Two details follow it on
purpose: the CNN embeddings go through float16 and back, as the training cache
stored them, and every game is padded to 100 plies before the BiLSTM, as the
batcher does.

The CNN trunk weights are not shipped here: A4 kept them frozen, so they are
the served rating checkpoint's own conv layers. The per-ply logits are not
used for critical moves (they localized substituted moves at or below chance).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attention import BahdanauAttention

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_WEIGHTS_PATH = MODELS_DIR / "detector_cnn_bilstm_a4.pt"
DEFAULT_PROVENANCE_PATH = MODELS_DIR / "detector_cnn_bilstm_a4.json"

SCORE_ID = "cnn_bilstm_a4"

EMB_DIM = 256
MAXLEN = 100
WHITE = 0
BLACK = 1


class A4Model(nn.Module):
    """BiLSTM + causal Bahdanau attention + gated-attention MIL head.

    Attribute names match ``extra_arm_a4.make_model`` so the thesis state dict
    loads with ``strict=True``.
    """

    def __init__(self, lstm_h: int = 64, lstm_layers: int = 3, attention_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(EMB_DIM + 1, lstm_h, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.attention = BahdanauAttention(2 * lstm_h, attention_dim=attention_dim)
        self.drop = nn.Dropout(0.5)
        hidden = 2 * lstm_h
        self.inst = nn.Linear(hidden, 1)
        self.att_v = nn.Linear(hidden, 64)
        self.att_u = nn.Linear(hidden, 64)
        self.att_w = nn.Linear(64, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, emb, clk, lengths, pool_mask):
        x = torch.cat([self.drop(emb), clk.unsqueeze(2)], dim=2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=emb.shape[1])
        ctx, _ = self.attention.forward_causal(out, lengths)
        h = out + ctx
        s = self.inst(h).squeeze(-1)
        e = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        e = e.masked_fill(~pool_mask, -1e4)
        a = torch.softmax(e, dim=1)
        return (a * s).sum(1) + self.bias, s, a


@dataclass
class CnnBilstmDetector:
    model: A4Model
    provenance: dict


def load_cnn_bilstm_detector(
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
    provenance_path: Path = DEFAULT_PROVENANCE_PATH,
    device: torch.device | None = None,
) -> CnnBilstmDetector:
    with Path(provenance_path).open(encoding="utf-8") as f:
        provenance = json.load(f)
    model = A4Model()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device or torch.device("cpu"))
    model.eval()
    return CnnBilstmDetector(model=model, provenance=provenance)


def cnn_trunk(rating_model: nn.Module, boards: torch.Tensor) -> torch.Tensor:
    """Port of ``extra_arm_a4.cnn_trunk``: (n, 12, 8, 8) -> (n, 256), eval mode."""
    x = F.leaky_relu(rating_model.bn1(rating_model.conv1(boards)))
    x = rating_model.pool(x)
    x = F.leaky_relu(rating_model.bn2(rating_model.conv2(x)))
    x = rating_model.pool(x)
    x = F.leaky_relu(rating_model.bn3(rating_model.conv3(x)))
    x = rating_model.pool(x)
    x = F.leaky_relu(rating_model.bn4(rating_model.conv4(x)))
    return x.reshape(x.shape[0], -1)


def embed_positions(rating_model: nn.Module, positions: torch.Tensor) -> torch.Tensor:
    """(n, 256) float32 CNN embeddings of the first 100 positions, after the
    float16 round trip the training cache applied (``extra_arm_a4.stage_cache``).

    ``positions`` is the app's (seq, 12, 8, 8) or (1, seq, 12, 8, 8) board tensor.
    """
    boards = positions.reshape(-1, 12, 8, 8)[:MAXLEN].float()
    with torch.no_grad():
        return cnn_trunk(rating_model, boards).half().float()


def score_embeddings(detector: CnnBilstmDetector, emb: torch.Tensor, clocks_z: torch.Tensor) -> tuple[float, float]:
    """(white_score, black_score) from (n, 256) embeddings and the matching
    standardized clocks, each side scored in turn as the suspect.

    Mirrors ``extra_arm_a4.Batcher.get``: padded to 100 plies, pooling over the
    suspect side's plies (all valid plies if it has none).
    """
    device = next(detector.model.parameters()).device
    emb = emb[:MAXLEN].to(device).float()
    clk = clocks_z.reshape(-1)[:MAXLEN].to(device).float()
    n = emb.shape[0]
    if n == 0:
        raise ValueError("Cannot score a game with no plies")
    emb_pad = torch.zeros((2, MAXLEN, EMB_DIM), dtype=torch.float32, device=device)
    clk_pad = torch.zeros((2, MAXLEN), dtype=torch.float32, device=device)
    emb_pad[:, :n] = emb
    clk_pad[:, :n] = clk[:n]
    lengths = torch.tensor([n, n], dtype=torch.int64, device=device)
    ply = torch.arange(MAXLEN, device=device)
    valid = ply < n
    pool = torch.stack([valid & (ply % 2 == WHITE), valid & (ply % 2 == BLACK)])
    pool = pool | (~pool.any(dim=1, keepdim=True) & valid)
    with torch.no_grad():
        logit, _, _ = detector.model(emb_pad, clk_pad, lengths, pool)
    scores = torch.sigmoid(logit).cpu().tolist()
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("CNN-BiLSTM detector produced a non-finite score")
    return scores[0], scores[1]


def score_game(
    detector: CnnBilstmDetector,
    rating_model: nn.Module,
    positions: torch.Tensor,
    clocks_z: torch.Tensor,
) -> tuple[float, float]:
    """(white_score, black_score) for one game, from the app's board and
    standardized clock tensors exactly as fed to the rating model."""
    device = next(detector.model.parameters()).device
    emb = embed_positions(rating_model, positions.to(device))
    return score_embeddings(detector, emb, clocks_z)

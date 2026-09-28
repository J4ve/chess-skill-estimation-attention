"""
Parametrized RatingNet baseline with attention and anomaly extensions.

This file extends the original ``chess_rating_net.py`` from
``AstroBoy1/RatingNet`` (MIT) with:

* argparse + YAML configuration for all major training/inference knobs
* periodic checkpointing (every-epoch + latest/resume helper)
* optional Bahdanau/self-attention module wired into the model
* optional anomaly-detection branch

Training remains off by default so the frozen ``model_55.pth`` checkpoint can
be loaded for inference without GPU time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Any

# Allow running both as `python src/chess_rating_net.py` and via `from src.chess_rating_net import ...`.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from attention import BahdanauAttention, SelfAttention
from anomaly import AnomalyDetector
from format_data import board_to_array, categorize_time_control, rejected_games, time_to_seconds

logger = logging.getLogger(__name__)


class GameBlobStore:
    """Read-only random-access store of per-game pickles, keyed by .pkl basename.

    Backs training on the full corpus, where ~2.46M loose ``.pkl`` files (~500GB)
    do not fit the NVMe scratch disk but the same games compressed individually
    do (~11GB).  Built by ``build_corpus_store.py``; that module documents why
    this format was chosen over the alternatives.

    Safe under ``DataLoader(num_workers=N)``.  The SQLite connection is opened
    lazily and keyed by pid, so a connection is never shared across a ``fork``,
    and ``__getstate__`` drops it so the store also survives being pickled to
    ``spawn``-ed workers.  The database is opened ``mode=ro&immutable=1``, which
    tells SQLite the file cannot change and lets it skip locking entirely; that
    is correct here because training only ever reads.  Never point this at a
    store that is still being built.
    """

    FILENAME = "corpus.sqlite"

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._pid = None
        self._con = None

    @classmethod
    def open_if_present(cls, data_dir: str) -> GameBlobStore | None:
        """Return a store for ``data_dir`` if it holds one, else ``None``.

        Keeps ``--data_dir`` backwards compatible: a directory of loose ``.pkl``
        files still works exactly as before.
        """
        candidate = Path(data_dir) / cls.FILENAME
        return cls(candidate) if candidate.exists() else None

    def _uri(self) -> str:
        return f"file:{self.path}?mode=ro&immutable=1"

    def _connection(self) -> sqlite3.Connection:
        pid = os.getpid()
        if self._con is None or self._pid != pid:
            self._con = sqlite3.connect(self._uri(), uri=True)
            self._pid = pid
        return self._con

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_con"] = None
        state["_pid"] = None
        return state

    def names(self) -> list[str]:
        """Sorted ``.pkl`` basenames, the same list ``os.listdir`` used to give.

        Uses a short-lived connection so the parent process does not hold one
        open across the DataLoader's fork.
        """
        con = sqlite3.connect(self._uri(), uri=True)
        try:
            return [row[0] for row in con.execute("SELECT name FROM games ORDER BY name")]
        finally:
            con.close()

    def load(self, name: str) -> dict[str, Any]:
        row = self._connection().execute(
            "SELECT blob FROM games WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise KeyError(f"game {name!r} not found in corpus store {self.path}")
        return pickle.loads(zlib.decompress(row[0]))


class ChessGamesDataset(Dataset):
    """Dataset loader for preprocessed pickle games."""

    def __init__(
        self,
        filenames: list[str],
        max_moves: int = 100,
        ratings_mean: float = 1514,
        ratings_std: float = 366,
        clocks_mean: float = 273,
        clocks_std: float = 380,
        store: GameBlobStore | None = None,
    ):
        self.filenames = filenames
        self.store = store
        self.max_moves = max_moves
        self.ratings_mean = ratings_mean
        self.ratings_std = ratings_std
        self.clocks_mean = clocks_mean
        self.clocks_std = clocks_std

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name = self.filenames[idx]
        if self.store is None:
            with open(name, "rb") as f:
                game_info = pickle.load(f)
        else:
            game_info = self.store.load(name)
        clocks = [time_to_seconds(c) for c in game_info.get("Clocks", [])]
        clocks = [(c - self.clocks_mean) / self.clocks_std for c in clocks]
        clocks = torch.tensor(clocks, dtype=torch.float)[: self.max_moves]
        white = game_info.get("white", False)
        last_rating = game_info.get("rating_after_last_game")
        if last_rating is not None:
            last_rating = (last_rating - self.ratings_mean) / self.ratings_std
            last_rating = torch.tensor(last_rating, dtype=torch.float)
        positions = torch.stack(game_info["Positions"])[: self.max_moves]
        white_elo, black_elo = float(game_info["WhiteElo"]), float(game_info["BlackElo"])
        targets = torch.tensor([white_elo, black_elo], dtype=torch.float)
        targets = (targets - self.ratings_mean) / self.ratings_std

        length = len(positions)
        initial_time, increment = map(int, game_info["Time"].split("+"))
        estimated_duration = initial_time + 40 * increment
        time_control = categorize_time_control(estimated_duration)
        result = game_info.get("Result")

        return {
            "positions": positions,
            "clocks": clocks,
            "targets": targets,
            "length": length,
            "time_control": time_control,
            "white": white,
            "last_rating": last_rating,
            "result": result,
            "game_id": Path(name).stem,
            "white_elo": white_elo,
            "black_elo": black_elo,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sequences for batching."""
    positions = pad_sequence([item["positions"] for item in batch], batch_first=True)
    clocks = pad_sequence([item["clocks"] for item in batch], batch_first=True)
    targets = torch.stack([item["targets"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.int)
    time_controls = [item["time_control"] for item in batch]
    white = torch.tensor([item["white"] for item in batch])
    game_ids = [item["game_id"] for item in batch]
    white_elos = torch.tensor([item["white_elo"] for item in batch], dtype=torch.float)
    black_elos = torch.tensor([item["black_elo"] for item in batch], dtype=torch.float)
    last_rating = None
    if batch[0]["last_rating"] is not None:
        last_rating = torch.stack([item["last_rating"] for item in batch])
    if batch[0]["result"] is not None:
        results = [item["result"] for item in batch]
        return {
            "positions": positions,
            "clocks": clocks,
            "targets": targets,
            "lengths": lengths,
            "time_controls": time_controls,
            "white": white,
            "last_rating": last_rating,
            "results": results,
            "game_ids": game_ids,
            "white_elos": white_elos,
            "black_elos": black_elos,
        }
    return {
        "positions": positions,
        "clocks": clocks,
        "targets": targets,
        "lengths": lengths,
        "time_controls": time_controls,
        "white": white,
        "last_rating": last_rating,
        "game_ids": game_ids,
        "white_elos": white_elos,
        "black_elos": black_elos,
    }


class ChessEloPredictor(nn.Module):
    """CNN-BiLSTM rating estimator with optional attention and anomaly branch.

    The base architecture matches the released RatingNet checkpoint. When
    ``use_attention`` is False the forward pass is identical to the baseline,
    so the frozen ``model_55.pth`` weights load cleanly. When True, an
    attention module is attached; its weights are *not* present in the frozen
    checkpoint and must be trained on HPC (outside this prototype scope).

    Args:
        conv_filters: Number of filters in the first conv layer.
        lstm_layers: Number of LSTM layers.
        dropout_rate: Dropout probability.
        lstm_h: LSTM hidden size.
        fc1_h: First fully-connected layer size.
        bidirectional: Whether the LSTM is bidirectional.
        use_attention: If True, attach an attention module.
        attention_type: ``bahdanau`` or ``self``.
        attention_dim: Projection size for Bahdanau attention.
        use_anomaly: If True, expose an anomaly detector branch.
        separate_heads: If True, give White and Black their own rating head
            (each its own ``fc1``+``fc2``) instead of the shared ``fc1`` feeding
            one ``Linear(fc1_h, 2)``. Off by default, and when off the module
            list, the parameter names and the forward pass are unchanged.
    """

    def __init__(
        self,
        conv_filters: int = 32,
        lstm_layers: int = 3,
        dropout_rate: float = 0.5,
        lstm_h: int = 64,
        fc1_h: int = 32,
        bidirectional: bool = True,
        use_attention: bool = False,
        attention_type: str = "bahdanau",
        attention_dim: int = 64,
        use_anomaly: bool = False,
        deeper_cnn: bool = False,
        separate_heads: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_anomaly = use_anomaly
        self.deeper_cnn = deeper_cnn
        self.separate_heads = separate_heads
        self.bidirectional = bidirectional
        self.lstm_h = lstm_h

        # CNN trunk (identical to baseline)
        self.conv1 = nn.Conv2d(12, conv_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters)
        self.conv2 = nn.Conv2d(conv_filters, conv_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters * 2)
        self.conv3 = nn.Conv2d(conv_filters * 2, conv_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters * 4)
        self.conv4 = nn.Conv2d(conv_filters * 4, conv_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters * 8)
        # Optional extra conv at the 2x2 stage (Omori 2026-08-29: "add another
        # layer and train on the gpu").  Same channel count as conv3, so the
        # trunk output width and everything downstream (LSTM input, attention,
        # rating head) are byte-identical to the baseline.  Placed here, before
        # the third pool, because the feature map still has 2x2 spatial extent;
        # conv4 already runs on a 1x1 map so stacking depth there would be
        # degenerate.  Off by default -> forward pass unchanged.
        self.conv3b: nn.Module | None = None
        self.bn3b: nn.Module | None = None
        if deeper_cnn:
            self.conv3b = nn.Conv2d(conv_filters * 4, conv_filters * 4, kernel_size=3, padding=1)
            self.bn3b = nn.BatchNorm2d(conv_filters * 4)
        self.pool = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # BiLSTM (identical to baseline)
        lstm_input_size = conv_filters * 8 + 1
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_h,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_h * 2 if bidirectional else lstm_h

        # Optional attention module (does not change baseline head dims)
        self.attention: nn.Module | None = None
        if use_attention:
            if attention_type == "bahdanau":
                self.attention = BahdanauAttention(lstm_output_dim, attention_dim=attention_dim)
            elif attention_type == "self":
                self.attention = SelfAttention(lstm_output_dim)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")

        # Rating head.  Default: identical to baseline, one shared fc1 feeding a
        # single Linear(fc1_h, 2) whose two rows are the White and Black outputs.
        #
        # separate_heads splits the head into two independent per-side stacks.
        # Note that splitting only the final Linear(fc1_h, 2) into two
        # Linear(fc1_h, 1) layers would be an exact reparameterization: the two
        # rows of fc2 are already independent parameters, so that change alone
        # cannot make the sides differ.  The branch therefore starts one layer
        # earlier, at fc1, so each side owns the hidden projection that feeds it.
        if separate_heads:
            self.fc1_white = nn.Linear(lstm_output_dim, fc1_h)
            self.fc2_white = nn.Linear(fc1_h, 1)
            self.fc1_black = nn.Linear(lstm_output_dim, fc1_h)
            self.fc2_black = nn.Linear(fc1_h, 1)
        else:
            self.fc1 = nn.Linear(lstm_output_dim, fc1_h)
            self.fc2 = nn.Linear(fc1_h, 2)

        # Optional anomaly branch
        self.anomaly_detector: nn.Module | None = None
        if use_anomaly:
            self.anomaly_detector = AnomalyDetector()

    def forward(
        self,
        positions: torch.Tensor,
        clocks: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
        baseline: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        Returns:
            By default (return_attention=False): ``(per_move_preds, last_step_preds)``
                identical to the baseline when ``use_attention=False``.
            If return_attention=True: a dictionary with predictions, attention
                weights, anomaly scores, and intermediate values.
        """
        batch_size = positions.size(0)
        sequence_length = positions.size(1)
        positions = positions.view(-1, 12, 8, 8)

        x = F.leaky_relu(self.bn1(self.conv1(positions)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        if self.conv3b is not None:
            x = F.leaky_relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = self.dropout1(x)
        x = x.view(batch_size, sequence_length, -1)

        clocks = clocks.unsqueeze(2)
        lstm_input = torch.cat((x, clocks), dim=2)
        packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # --- Causal-cumulative attention (Fix 1) ---
        # When use_attention=True, compute per-ply attended context and add it
        # to the BiLSTM output before the rating head.  This makes attention
        # part of the gradient path so its parameters are actually trained.
        # When use_attention=False, this block is skipped and the forward pass
        # is identical to the baseline (frozen checkpoint loads cleanly).
        attention_weights = None
        rating_input = lstm_output  # default: raw BiLSTM output
        if self.attention is not None:
            attn_context, attn_weights_raw = self.attention.forward_causal(
                lstm_output, lengths
            )
            # Residual connection: preserves baseline signal while adding
            # the attention-modulated context.
            rating_input = lstm_output + attn_context
            # Store per-ply weights for diagnostics / anomaly.  For Bahdanau
            # the raw weights are (batch, seq, seq); collapse to (batch, seq)
            # by taking the diagonal (each ply's self-weight is its importance)
            # or, more usefully, the mean weight received across query positions.
            if attn_weights_raw.dim() == 3:
                # Bahdanau: (batch, seq_q, seq_k) → per-key importance
                # Use mean over query dim (how much each key is attended to).
                attention_weights = attn_weights_raw.mean(dim=1)
            else:
                # SelfAttention already returns (batch, seq) step_importance
                attention_weights = attn_weights_raw

        if self.separate_heads:
            y_w = self.dropout1(F.leaky_relu(self.fc1_white(rating_input)))
            y_b = self.dropout1(F.leaky_relu(self.fc1_black(rating_input)))
            # (batch, seq, 1) each -> (batch, seq, 2), column 0 White, 1 Black,
            # so every downstream consumer sees the baseline output shape.
            per_move_preds = torch.cat((self.fc2_white(y_w), self.fc2_black(y_b)), dim=-1)
        else:
            y = F.leaky_relu(self.fc1(rating_input))
            y = self.dropout1(y)
            per_move_preds = self.fc2(y)

        idx = torch.arange(batch_size, device=positions.device)
        last_time_step_output = per_move_preds[idx, lengths - 1, :]

        if not return_attention:
            return per_move_preds, last_time_step_output

        anomaly_out = None
        if self.anomaly_detector is not None and baseline is not None:
            anomaly_out = self.anomaly_detector(per_move_preds, baseline, attention_weights)

        return {
            "per_move_preds": per_move_preds,
            "last_step_preds": last_time_step_output,
            "attention_weights": attention_weights,
            "anomaly": anomaly_out,
        }

    def load_base_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = False) -> None:
        """Load a baseline checkpoint, ignoring keys that belong to attention/anomaly.

        Warns when the checkpoint lacks attention/anomaly parameters that this
        model actually uses, since those modules would stay randomly
        initialized and silently corrupt inference.
        """
        result = self.load_state_dict(state_dict, strict=strict)
        if not strict and result.missing_keys:
            served_missing = [
                k
                for k in result.missing_keys
                if (self.use_attention and k.startswith("attention."))
                or (self.use_anomaly and k.startswith("anomaly_detector."))
            ]
            if served_missing:
                logger.warning(
                    "Checkpoint is missing %d attention/anomaly parameter(s) the served model "
                    "uses; they remain randomly initialized: %s",
                    len(served_missing),
                    served_missing if len(served_missing) <= 8 else served_missing[:8] + ["..."],
                )


# --- White/Black separation experiment (both halves off by default) ---------
#
# Motivation: the released architecture's two rating outputs come out nearly
# identical within a game (median per-game spread 0.71 rating points across the
# 255,000 held-out test games, and the two rows of ``fc2`` have cosine
# similarity 0.9999).  Nothing in the objective rewards telling the sides apart,
# and Lichess pairs close opponents, so one shared estimate is almost free.
#
# GAP_WEIGHT_SCALE/GAP_WEIGHT_CAP define the Arm A weighting.  The scale is set
# so that the 300-point gap used as the wide-gap threshold in the mirroring
# analysis receives weight 4, i.e. four times an evenly matched game.  The cap
# bounds the rare extremes: the widest real gap in the test partition is 1,837
# points, which would otherwise carry over 19 times the weight of an even game
# and let a few hundred games dominate the gradient.
GAP_WEIGHT_SCALE = 100.0
GAP_WEIGHT_CAP = 10.0


def gap_weights(
    white_elos: torch.Tensor,
    black_elos: torch.Tensor,
    scale: float = GAP_WEIGHT_SCALE,
    cap: float = GAP_WEIGHT_CAP,
) -> torch.Tensor:
    """Per-game loss weight that grows with the two players' rating gap.

    ``w = min(1 + |white_elo - black_elo| / scale, cap)``, so an evenly matched
    game keeps weight 1 and a 300-point mismatch counts four times as much.
    Elos are raw rating points, not standardized.
    """
    gap = torch.abs(white_elos - black_elos)
    return torch.clamp(1.0 + gap / scale, max=cap)


def side_abs_error(preds_elo: torch.Tensor, targets_elo: torch.Tensor) -> torch.Tensor:
    """Mean absolute per-side error of each game, in rating points. (B,2) -> (B,)."""
    return torch.abs(preds_elo - targets_elo).mean(dim=1)


def rating_gap_abs_error(preds_elo: torch.Tensor, targets_elo: torch.Tensor) -> torch.Tensor:
    """Absolute error of the predicted White-minus-Black difference. (B,2) -> (B,).

    This is the term that penalizes the mirrored solution directly: when the two
    outputs are equal the predicted difference is zero, so this equals the true
    rating gap no matter how good the per-side estimates are.
    """
    pred_diff = preds_elo[:, 0] - preds_elo[:, 1]
    true_diff = targets_elo[:, 0] - targets_elo[:, 1]
    return torch.abs(pred_diff - true_diff)


def separation_training_loss(
    preds_elo: torch.Tensor,
    targets_elo: torch.Tensor,
    white_elos: torch.Tensor | None = None,
    black_elos: torch.Tensor | None = None,
    gap_weighting: bool = False,
    diff_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Training loss for the White/Black separation arms, in rating points.

    Returns ``(total, side_term, diff_term)``; ``diff_term`` is None when the
    difference loss is off.  With both options off the side term is the plain
    mean absolute error over every per-side output, the same quantity
    ``nn.L1Loss`` produces, but the trainer keeps using ``nn.L1Loss`` on that
    path so the unflagged run is bit-for-bit what it always was.

    ``gap_weighting`` turns the side term into a weighted mean with weights from
    ``gap_weights``.  A weighted *mean* (dividing by the summed weight rather
    than the count) keeps the loss in rating points and on the same scale as the
    unweighted arm, so the learning rate carries over unchanged.
    """
    per_game = side_abs_error(preds_elo, targets_elo)
    weights = None
    if gap_weighting:
        if white_elos is None or black_elos is None:
            raise ValueError("gap_weighting needs white_elos and black_elos")
        weights = gap_weights(white_elos, black_elos)
        side = (weights * per_game).sum() / weights.sum()
    else:
        side = per_game.mean()

    if diff_loss_weight <= 0.0:
        return side, side, None

    per_game_diff = rating_gap_abs_error(preds_elo, targets_elo)
    if weights is not None:
        diff = (weights * per_game_diff).sum() / weights.sum()
    else:
        diff = per_game_diff.mean()
    return side + diff_loss_weight * diff, side, diff


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
    dense_supervision: bool = False,
    gap_weighting: bool = False,
    diff_loss_weight: float = 0.0,
    loss_components: dict[str, float] | None = None,
) -> float:
    """Train for one epoch.

    When ``dense_supervision`` is True (Fix 4), the loss is computed on
    **every valid ply** (padding excluded) rather than only the last ply.
    The rating label is constant across the game, so every ply is a valid
    training target.  Evaluation remains last-ply-only.

    ``gap_weighting`` and ``diff_loss_weight`` are the two White/Black
    separation arms; see ``separation_training_loss``.  Both are off by
    default, and when both are off this function takes exactly the code path it
    always did.  ``loss_components``, when given, is filled with the epoch-mean
    side and difference terms so a run's log shows them moving separately.
    """
    use_separation = gap_weighting or diff_loss_weight > 0.0
    model.train()
    total_train_loss = 0.0
    total_side = 0.0
    total_diff = 0.0
    for batch in train_loader:
        positions = batch["positions"].to(device)
        clocks = batch["clocks"].to(device)
        targets = batch["targets"].to(device)  # (batch, 2)
        lengths = batch["lengths"]
        optimizer.zero_grad()
        per_move_preds, last_step_preds = model(positions, clocks, lengths)

        if use_separation:
            loss, side_term, diff_term = separation_training_loss(
                last_step_preds * ratings_std + ratings_mean,
                targets * ratings_std + ratings_mean,
                white_elos=batch["white_elos"].to(device),
                black_elos=batch["black_elos"].to(device),
                gap_weighting=gap_weighting,
                diff_loss_weight=diff_loss_weight,
            )
            total_side += side_term.item()
            if diff_term is not None:
                total_diff += diff_term.item()
        elif dense_supervision:
            # Expand targets to (batch, seq, 2) — same label for every ply.
            batch_size, seq_len, _ = per_move_preds.shape
            targets_expanded = targets.unsqueeze(1).expand(batch_size, seq_len, 2)
            # Build a per-ply validity mask (batch, seq, 1) to exclude padding.
            mask = (
                torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1).to(device)
            ).unsqueeze(-1).float()  # (batch, seq, 1)
            # De-standardize before L1 so the loss is in Elo points.
            preds_elo = per_move_preds * ratings_std + ratings_mean
            targets_elo = targets_expanded * ratings_std + ratings_mean
            # Masked mean: sum of valid-ply losses / number of valid plies.
            elementwise_loss = torch.abs(preds_elo - targets_elo) * mask  # (batch, seq, 2)
            loss = elementwise_loss.sum() / (mask.sum() * 2)  # mean over plies and sides
        else:
            loss = criterion(
                last_step_preds * ratings_std + ratings_mean,
                targets * ratings_std + ratings_mean,
            )

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    n_batches = len(train_loader)
    if loss_components is not None and use_separation:
        loss_components["side"] = total_side / n_batches
        loss_components["diff"] = total_diff / n_batches
    return total_train_loss / n_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
) -> float:
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            positions = batch["positions"].to(device)
            clocks = batch["clocks"].to(device)
            targets = batch["targets"].to(device)
            lengths = batch["lengths"]
            _, outputs = model(positions, clocks, lengths)
            loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)


def mae_per_item(outputs: torch.Tensor, targets: torch.Tensor, ratings_mean: float, ratings_std: float) -> torch.Tensor:
    outputs_rescaled = outputs * ratings_std + ratings_mean
    targets_rescaled = targets * ratings_std + ratings_mean
    return torch.abs(outputs_rescaled - targets_rescaled).mean(dim=1)


def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    ratings_mean: float = 1514,
    ratings_std: float = 366,
    per_game_csv: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate on the test set.

    When ``per_game_csv`` is given, also dumps one row per game
    (``game_id, white_err, black_err, white_signed_err, black_signed_err,
    time_control, white_elo, black_elo``) with errors in Elo points.
    ``white_err``/``black_err`` are absolute errors; the ``*_signed_err``
    columns carry the signed per-side error (predicted minus target) so the
    paired-bootstrap difference and calibration/shrinkage analyses can be
    computed downstream.
    """
    model.eval()
    total_test_loss = 0.0
    loss_by_time_control = {tc: 0.0 for tc in ["ultrabullet", "bullet", "blitz", "rapid", "classical"]}
    count_by_time_control = {tc: 0 for tc in ["ultrabullet", "bullet", "blitz", "rapid", "classical"]}

    csv_file = None
    csv_writer = None
    if per_game_csv is not None:
        csv_file = open(per_game_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["game_id", "white_err", "black_err", "white_signed_err", "black_signed_err", "time_control", "white_elo", "black_elo"])

    try:
        with torch.no_grad():
            for batch in test_loader:
                positions = batch["positions"].to(device)
                clocks = batch["clocks"].to(device)
                targets = batch["targets"].to(device)
                lengths = batch["lengths"]
                time_controls = batch["time_controls"]

                _, outputs = model(positions, clocks, lengths)
                loss = criterion(outputs * ratings_std + ratings_mean, targets * ratings_std + ratings_mean)
                total_test_loss += loss.item()

                if isinstance(criterion, nn.L1Loss):
                    mae = mae_per_item(outputs, targets, ratings_mean, ratings_std)
                    for idx, time_control in enumerate(time_controls):
                        loss_by_time_control[time_control] += mae[idx].item()
                        count_by_time_control[time_control] += 1

                if csv_writer is not None:
                    signed_errors = (
                        outputs * ratings_std + ratings_mean - targets * ratings_std - ratings_mean
                    )  # (batch, 2) signed Elo points
                    abs_errors = torch.abs(signed_errors)
                    for i, time_control in enumerate(time_controls):
                        csv_writer.writerow(
                            [
                                batch["game_ids"][i],
                                round(abs_errors[i, 0].item(), 4),
                                round(abs_errors[i, 1].item(), 4),
                                round(signed_errors[i, 0].item(), 4),
                                round(signed_errors[i, 1].item(), 4),
                                time_control,
                                batch["white_elos"][i].item(),
                                batch["black_elos"][i].item(),
                            ]
                        )
    finally:
        if csv_file is not None:
            csv_file.close()

    for key in loss_by_time_control:
        if count_by_time_control[key] > 0:
            loss_by_time_control[key] /= count_by_time_control[key]

    return total_test_loss / len(test_loader), loss_by_time_control


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    params: dict[str, Any],
    best_val_loss: float | None = None,
    scheduler: ReduceLROnPlateau | None = None,
    best_epoch: int | None = None,
) -> None:
    """Save optimizer/scheduler state, epoch counters, and hyperparameters alongside weights."""
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "params": params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
    scheduler: ReduceLROnPlateau | None = None,
) -> dict[str, Any]:
    """Resume from a checkpoint written by ``save_checkpoint``."""
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def load_or_create_split(
    data_dir: str,
    all_files: list[str],
    split_seed: int,
) -> tuple[list[str], list[str], list[str], str]:
    """Deterministic, auditable train/val/test split persisted to ``split_manifest.json``.

    On first run, splits the sorted basenames with ``split_seed`` and writes the
    manifest (three file lists + SHA-256 over the sorted basenames). On every
    subsequent run the manifest is loaded and verified instead of recomputed.

    Returns ``(train, val, test)`` basename lists and the manifest hash.
    """
    manifest_path = Path(data_dir) / "split_manifest.json"
    digest = hashlib.sha256("\n".join(all_files).encode("utf-8")).hexdigest()

    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("sha256") != digest:
            raise ValueError(
                f"split_manifest.json hash mismatch (manifest={manifest.get('sha256')}, "
                f"current={digest}): the data directory changed since the split was frozen. "
                f"Delete {manifest_path} to recompute the split."
            )
        train_names = manifest["train_files"]
        val_names = manifest["val_files"]
        test_names = manifest["test_files"]
        if sorted(train_names + val_names + test_names) != sorted(all_files):
            raise ValueError(
                f"split_manifest.json file lists do not match the contents of {data_dir}. "
                f"Delete {manifest_path} to recompute the split."
            )
        print(f"Loaded split manifest {manifest_path} (verified)")
    else:
        train_val_names, test_names = train_test_split(all_files, test_size=0.1, random_state=split_seed)
        train_names, val_names = train_test_split(train_val_names, test_size=0.2, random_state=split_seed)
        train_names, val_names, test_names = sorted(train_names), sorted(val_names), sorted(test_names)
        manifest = {
            "sha256": digest,
            "split_seed": split_seed,
            "train_files": train_names,
            "val_files": val_names,
            "test_files": test_names,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote split manifest {manifest_path}")

    return train_names, val_names, test_names, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RatingNet prototype trainer / inference runner")
    parser.add_argument("--data_dir", default="data/processed_games", help="Directory containing .pkl preprocessed games")
    parser.add_argument("--experiment", default="cnn_bilstm_clocks_all", help="Experiment name (used for model/log dirs)")
    parser.add_argument("--train", action="store_true", default=False, help="Run training (default: inference-only)")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--model_dir", default="models", help="Root directory for checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate in inference mode (default: discover model_55.pth)")
    parser.add_argument("--config", default=None, help="Optional YAML config file to override defaults")
    parser.add_argument("--use_attention", action="store_true", help="Attach attention module to the model")
    parser.add_argument("--use_anomaly", action="store_true", help="Attach anomaly-detection branch")
    parser.add_argument("--deeper_cnn", action="store_true", help="Insert conv3b (same-channel 3x3 conv + BN) at the 2x2 stage; trunk output and downstream dims unchanged")
    parser.add_argument("--val_batch_size", type=int, default=512, help="Validation/test batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for torch/numpy/random and training shuffle (vary across seed-ablation runs)")
    parser.add_argument("--split_seed", type=int, default=42, help="Train/val/test split seed; NEVER vary (preserves comparability with the paper's 182 MAE)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Adam weight decay")
    parser.add_argument("--patience", type=int, default=5, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="ReduceLROnPlateau factor")
    parser.add_argument("--conv_filters", type=int, default=32, help="CNN base filter count")
    parser.add_argument("--lstm_layers", type=int, default=3, help="Number of LSTM layers")
    parser.add_argument("--lstm_h", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--fc1_h", type=int, default=32, help="FC hidden size")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True, help="Use bidirectional LSTM")
    parser.add_argument("--attention_type", default="bahdanau", choices=["bahdanau", "self"], help="Attention variant")
    parser.add_argument("--attention_dim", type=int, default=64, help="Bahdanau attention projection size")
    parser.add_argument("--dense_supervision", action="store_true", default=False,
                        help="Supervise every valid ply (not just the last) during training")
    parser.add_argument("--gap_weighting", action="store_true", default=False,
                        help="Arm A of the White/Black separation experiment: weight each game's "
                             "loss by min(1 + |white_elo - black_elo| / 100, 10). Off by default.")
    parser.add_argument("--separate_heads", action="store_true", default=False,
                        help="Arm B architecture: give White and Black their own rating head "
                             "(own fc1 and fc2) instead of one shared Linear(fc1_h, 2). Off by default.")
    parser.add_argument("--diff_loss_weight", type=float, default=0.0,
                        help="Arm B objective: weight of the |predicted minus true White-Black rating "
                             "difference| term added to the per-side loss. 0 disables it (default).")
    parser.add_argument("--ratings_mean", type=float, default=1514,
                        help="Rating normalization mean. Default is the baseline paper's constant; "
                             "do not change for the primary/comparable runs (see AGENTS.md).")
    parser.add_argument("--ratings_std", type=float, default=366,
                        help="Rating normalization std. Default is the baseline paper's constant; "
                             "do not change for the primary/comparable runs (see AGENTS.md).")
    return parser


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    """Merge optional YAML config into argparse namespace."""
    if args.config is None:
        return args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> int:
    parser = build_parser()
    args = load_config(parser.parse_args())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Hyperparameters: CLI > YAML > checkpoint > defaults.
    params: dict[str, Any] = {
        "train_batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "optimizer": "Adam",
        "patience": args.patience,
        "lr_factor": args.lr_factor,
        "conv_filters": args.conv_filters,
        "lstm_layers": args.lstm_layers,
        "bidirectional": args.bidirectional,
        "dropout_rate": args.dropout_rate,
        "lstm_h": args.lstm_h,
        "fc1_h": args.fc1_h,
        "use_attention": args.use_attention,
        "attention_type": args.attention_type,
        "attention_dim": args.attention_dim,
        "use_anomaly": args.use_anomaly,
        "deeper_cnn": args.deeper_cnn,
        "dense_supervision": args.dense_supervision,
        "gap_weighting": args.gap_weighting,
        "separate_heads": args.separate_heads,
        "diff_loss_weight": args.diff_loss_weight,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "ratings_mean": args.ratings_mean,
        "ratings_std": args.ratings_std,
    }

    if args.dense_supervision and (args.gap_weighting or args.diff_loss_weight > 0.0):
        raise SystemExit(
            "--dense_supervision cannot be combined with --gap_weighting or "
            "--diff_loss_weight: the separation losses are defined on the last ply only."
        )
    if args.diff_loss_weight < 0.0:
        raise SystemExit("--diff_loss_weight must be zero or positive")

    data_dir = args.data_dir
    experiment_name = args.experiment
    model_dir = Path(args.model_dir) / experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("runs") / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    store = GameBlobStore.open_if_present(data_dir)
    if store is None:
        all_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
        if not all_files:
            raise FileNotFoundError(f"No .pkl files found in {data_dir}")
    else:
        all_files = store.names()
        if not all_files:
            raise FileNotFoundError(f"Corpus store {store.path} contains no games")
        print(f"Corpus store: {store.path} ({len(all_files)} games)")

    train_names, val_names, test_names, manifest_hash = load_or_create_split(data_dir, all_files, args.split_seed)
    if store is None:
        train_files = [os.path.join(data_dir, f) for f in train_names]
        val_files = [os.path.join(data_dir, f) for f in val_names]
        test_files = [os.path.join(data_dir, f) for f in test_names]
    else:
        # the store is keyed by basename, so the names are already the keys
        train_files, val_files, test_files = train_names, val_names, test_names

    print(f"Environment: data_dir={data_dir} files={len(all_files)} seed={args.seed} split_seed={args.split_seed}")
    print(f"Rejected {rejected_games} games for zero/partial clock annotation (of {len(all_files)} parsed)")
    print(f"Split: train={len(train_files)} val={len(val_files)} test={len(test_files)} manifest_sha256={manifest_hash}")
    print(f"Resolved hyperparameters: {params}")

    train_dataset = ChessGamesDataset(train_files, ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"], store=store)
    val_dataset = ChessGamesDataset(val_files, ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"], store=store)
    test_dataset = ChessGamesDataset(test_files, ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"], store=store)

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["val_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=params["val_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=params["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ChessEloPredictor(
        conv_filters=params["conv_filters"],
        lstm_layers=params["lstm_layers"],
        dropout_rate=params["dropout_rate"],
        lstm_h=params["lstm_h"],
        fc1_h=params["fc1_h"],
        bidirectional=params["bidirectional"],
        use_attention=params["use_attention"],
        attention_type=params["attention_type"],
        attention_dim=params["attention_dim"],
        use_anomaly=params["use_anomaly"],
        deeper_cnn=params.get("deeper_cnn", False),
        separate_heads=params.get("separate_heads", False),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=params["patience"], factor=params["lr_factor"])
    criterion = nn.L1Loss()

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    latest_path = model_dir / "latest.pth"

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = load_checkpoint(args.resume, model, optimizer, device, scheduler=scheduler)
        # A resume rebuilds the model and the objective from the CLI, not from the
        # checkpoint, so a launcher that forgets a flag would quietly carry on
        # training a different experiment under the same name. That has already
        # cost this project one full re-run (see hpc/README.md on launch_run.sh),
        # and the auto-resume watchdog makes it a real risk: refuse instead.
        saved = ckpt.get("params", {})
        drifted = [
            (key, saved.get(key, default), params[key])
            for key, default in (
                ("gap_weighting", False),
                ("separate_heads", False),
                ("diff_loss_weight", 0.0),
                ("use_attention", False),
                ("deeper_cnn", False),
                ("dense_supervision", False),
            )
            if saved.get(key, default) != params[key]
        ]
        if drifted:
            raise SystemExit(
                "refusing to resume: these settings differ between the checkpoint and this "
                "command line, so the resumed run would not be the run that wrote the "
                "checkpoint: "
                + "; ".join(f"{k}: checkpoint={c!r} command line={n!r}" for k, c, n in drifted)
            )
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_epoch = ckpt.get("best_epoch", 0)

    if not args.train:
        # Inference-only path: load a checkpoint for evaluation.
        # Prefer an explicit --checkpoint; otherwise discover the frozen baseline
        # (experiment-specific path, then the root models/ directory, e.g. models/model_55.pth).
        if args.checkpoint:
            best_path = Path(args.checkpoint)
            if not best_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {best_path}")
        else:
            candidate_paths = [model_dir / "model_55.pth", Path(args.model_dir) / "model_55.pth"]
            best_path = next((p for p in candidate_paths if p.exists()), None)
            if best_path is None:
                raise FileNotFoundError(
                    f"Frozen checkpoint not found. Place model_55.pth at {model_dir / 'model_55.pth'} "
                    f"or {Path(args.model_dir) / 'model_55.pth'}, or pass --checkpoint, then retry."
                )
        print(f"Loading checkpoint from {best_path}")
        saved_model = torch.load(best_path, map_location=device)
        model.load_base_state_dict(saved_model["model_state_dict"], strict=False)
        per_game_csv = model_dir / "per_game_errors.csv"
        test_loss, loss_by_tc = test(
            model, test_loader, device, criterion,
            ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"],
            per_game_csv=str(per_game_csv),
        )
        print("Test Loss:", test_loss)
        print("Loss by time control:", loss_by_tc)
        print(f"Wrote per-game errors to {per_game_csv}")
        return 0

    writer = SummaryWriter(log_dir=str(log_dir))
    print("Training model")
    start = time.time()

    for epoch in range(start_epoch, params["epochs"]):
        epoch_start = time.time()
        components: dict[str, float] = {}
        train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimizer,
            ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"],
            dense_supervision=params.get("dense_supervision", False),
            gap_weighting=params.get("gap_weighting", False),
            diff_loss_weight=params.get("diff_loss_weight", 0.0),
            loss_components=components,
        )
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        if components:
            print(
                f"Epoch {epoch + 1}, Train Side Loss: {components['side']:.4f}, "
                f"Train Diff Loss: {components['diff']:.4f}"
            )
        val_loss = validate(
            model, val_loader, device, criterion,
            ratings_mean=params["ratings_mean"], ratings_std=params["ratings_std"],
        )
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        epoch_duration = (time.time() - epoch_start) / 60
        writer.add_scalar("Timing/Epoch Duration", epoch_duration, epoch)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt = model_dir / f"best_model.pth"
            save_checkpoint(
                best_ckpt, model, optimizer, epoch + 1, params,
                best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
            )
            print("Saved new best model")

        # Periodic every-epoch checkpoint. Written after the best-loss update so
        # it carries the post-update best (resume never sees a stale value).
        epoch_ckpt = model_dir / f"model_{epoch + 1}.pth"
        save_checkpoint(
            epoch_ckpt, model, optimizer, epoch + 1, params,
            best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
        )
        save_checkpoint(
            latest_path, model, optimizer, epoch + 1, params,
            best_val_loss=best_val_loss, scheduler=scheduler, best_epoch=best_epoch,
        )
        print(f"Saved epoch checkpoint {epoch_ckpt}")

    end = time.time()
    print("Training duration (min):", (end - start) / 60)
    print("best val loss:", best_val_loss)
    print("best val epoch:", best_epoch)
    writer.close()

    # Unambiguous completion marker for an unattended run. The auto-resume
    # watchdog keys off the "Training duration (min):" line above; this file is
    # the same signal in a form a later reader can test with one `test -f`,
    # and it records which epoch to evaluate. Written to a temp file and
    # renamed so a reader never sees a half-written marker.
    marker = model_dir / "FINISHED"
    tmp_marker = model_dir / "FINISHED.tmp"
    with open(tmp_marker, "w") as f:
        json.dump(
            {
                "experiment": experiment_name,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epochs_completed": params["epochs"],
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "best_checkpoint": str(model_dir / "best_model.pth"),
                "training_minutes": (end - start) / 60,
                "params": params,
            },
            f,
            indent=2,
            default=str,
        )
    os.replace(tmp_marker, marker)
    print(f"Wrote completion marker {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

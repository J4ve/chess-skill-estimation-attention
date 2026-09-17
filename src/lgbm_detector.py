"""
LightGBM game-level detector (thesis arm A0g): one of the selectable suspicion
scores.

A gradient-boosted tree ensemble reads 97 per-game summary statistics pooled
from the same per-ply arrays the per-move detector uses (rating estimates,
attention, deviation from baseline, board and clock facts) and outputs a
game-level score in [0, 1] for one side.

Ported from the thesis HPC code, not re-derived:

* pooled features: ``analysis/scripts/train_anomaly_detector.py``
  (``pool_game``, ``_stats``), called with ``eng_pp=None`` as
  ``extra_arms_common.pooled_features`` does;
* model: ``analysis/scripts/extra_arm_a0.py`` (``fit_lgbm``) on the grouped
  split. The thesis run saved only its scores, so the booster in
  ``src/models/detector_lgbm_a0g.txt`` is a re-fit with the unchanged code,
  data and seed that reproduces every stored A0g score exactly (see
  ``src/models/detector_lgbm_a0g.json``).

Inference walks the saved LightGBM text model directly (numerical splits
only, LightGBM's own missing-value rules), so the app needs no ``lightgbm``
package. The per-ply arrays come from ``detector.per_ply_arrays``, so a side
is scored against the same baseline as the per-move detector.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import detector as detector_module

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "detector_lgbm_a0g.txt"
DEFAULT_PROVENANCE_PATH = MODELS_DIR / "detector_lgbm_a0g.json"

SCORE_ID = "lgbm_a0g"

PCTS = (90, 99)

# LightGBM decision_type bits (include/LightGBM/tree.h).
_CATEGORICAL_MASK = 1
_DEFAULT_LEFT_MASK = 2
_MISSING_NONE, _MISSING_ZERO, _MISSING_NAN = 0, 1, 2
_ZERO_THRESHOLD = 1e-35


@dataclass
class _Tree:
    split_feature: list[int]
    threshold: list[float]
    decision_type: list[int]
    left_child: list[int]
    right_child: list[int]
    leaf_value: list[float]

    def raw(self, x: Sequence[float]) -> float:
        if not self.split_feature:  # single-leaf tree
            return self.leaf_value[0]
        node = 0
        while node >= 0:
            fval = x[self.split_feature[node]]
            dt = self.decision_type[node]
            missing = (dt >> 2) & 3
            if math.isnan(fval) and missing != _MISSING_NAN:
                fval = 0.0
            if (missing == _MISSING_ZERO and abs(fval) <= _ZERO_THRESHOLD) or (
                missing == _MISSING_NAN and math.isnan(fval)
            ):
                node = self.left_child[node] if dt & _DEFAULT_LEFT_MASK else self.right_child[node]
            elif fval <= self.threshold[node]:
                node = self.left_child[node]
            else:
                node = self.right_child[node]
        return self.leaf_value[~node]


@dataclass
class LgbmDetector:
    trees: list[_Tree]
    feature_names: list[str]
    sigmoid: float
    provenance: dict

    def predict(self, x: Sequence[float]) -> float:
        raw = 0.0
        for tree in self.trees:
            raw += tree.raw(x)
        return 1.0 / (1.0 + math.exp(-self.sigmoid * raw))


def _parse_list(value: str, cast) -> list:
    return [cast(v) for v in value.split()] if value else []


def load_model_text(text: str) -> tuple[list[_Tree], list[str], float]:
    """Parse a LightGBM text model (``Booster.save_model``) with numerical splits."""
    header: dict[str, str] = {}
    trees: list[_Tree] = []
    block: dict[str, str] | None = None

    def flush():
        if block is None:
            return
        if int(block.get("num_cat", "0")) != 0:
            raise ValueError("Categorical splits are not supported")
        trees.append(
            _Tree(
                split_feature=_parse_list(block.get("split_feature", ""), int),
                threshold=_parse_list(block.get("threshold", ""), float),
                decision_type=_parse_list(block.get("decision_type", ""), int),
                left_child=_parse_list(block.get("left_child", ""), int),
                right_child=_parse_list(block.get("right_child", ""), int),
                leaf_value=_parse_list(block["leaf_value"], float),
            )
        )

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Tree="):
            flush()
            block = {}
            continue
        if line == "end of trees":
            flush()
            block = None
            break
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        (block if block is not None else header)[key] = value

    if any(dt & _CATEGORICAL_MASK for t in trees for dt in t.decision_type):
        raise ValueError("Categorical splits are not supported")
    objective = header.get("objective", "")
    if not objective.startswith("binary"):
        raise ValueError(f"Expected a binary objective, got {objective!r}")
    sigmoid = 1.0
    for part in objective.split()[1:]:
        if part.startswith("sigmoid:"):
            sigmoid = float(part.split(":", 1)[1])
    return trees, header["feature_names"].split(), sigmoid


def load_lgbm_detector(
    model_path: Path = DEFAULT_MODEL_PATH, provenance_path: Path = DEFAULT_PROVENANCE_PATH
) -> LgbmDetector:
    with Path(provenance_path).open(encoding="utf-8") as f:
        provenance = json.load(f)
    trees, names, sigmoid = load_model_text(Path(model_path).read_text(encoding="utf-8"))
    if len(names) != len(provenance["features"]):
        raise ValueError("LightGBM model and provenance disagree on the feature count")
    return LgbmDetector(trees=trees, feature_names=list(provenance["features"]), sigmoid=sigmoid, provenance=provenance)


def _stats(x, prefix, out, pcts=PCTS, with_std=True):
    """Port of ``train_anomaly_detector._stats``."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_max"] = np.nan
        for p in pcts:
            out[f"{prefix}_p{p}"] = np.nan
        if with_std:
            out[f"{prefix}_std"] = np.nan
        return
    out[f"{prefix}_mean"] = float(x.mean())
    out[f"{prefix}_max"] = float(x.max())
    for p in pcts:
        out[f"{prefix}_p{p}"] = float(np.percentile(x, p))
    if with_std:
        out[f"{prefix}_std"] = float(x.std())


def pool_game(pp: dict[str, np.ndarray], n_real: int) -> dict[str, float]:
    """Port of ``train_anomaly_detector.pool_game`` with ``eng_pp=None``."""
    o: dict[str, float] = {}
    is_sus = pp["is_suspect_move"][:n_real].astype(bool)

    def both(name, arr):
        arr = np.asarray(arr, dtype=np.float64)[:n_real]
        _stats(arr, f"{name}_all", o)
        _stats(arr[is_sus], f"{name}_sus", o)

    both("d_t", pp["d_t"])
    ad = np.asarray(pp["alpha"][:n_real], np.float64) * np.asarray(pp["d_t"][:n_real], np.float64)
    both("alphad", ad)
    both("r_hat_suspect", pp["r_hat_suspect"])
    both("run_std", pp["run_std"])
    both("abs_first_diff", np.abs(pp["first_diff"][:n_real]))
    both("abs_second_diff", np.abs(pp["second_diff"][:n_real]))
    both("material_balance", pp["material_balance"])
    both("clock_remaining", pp["clock_remaining"])
    both("clock_delta", pp["clock_delta"])

    o["frac_capture"] = float(np.nanmean(pp["is_capture"][:n_real])) if n_real else np.nan
    o["frac_check"] = float(np.nanmean(pp["is_check"][:n_real])) if n_real else np.nan
    o["n_plies"] = float(n_real)
    o["n_suspect_moves"] = float(is_sus.sum())
    o["r_hat_final"] = float(pp["r_hat_suspect"][n_real - 1]) if n_real else np.nan
    o["run_std_final"] = float(pp["run_std"][n_real - 1]) if n_real else np.nan
    q = max(1, n_real // 4)
    o["d_t_lastq_mean"] = float(np.mean(pp["d_t"][n_real - q:n_real]))
    return o


def score_side(model: LgbmDetector, pp: dict[str, np.ndarray]) -> float:
    n = len(pp["ply_idx"])
    if n == 0:
        raise ValueError("Cannot score a game with no plies")
    pooled = pool_game(pp, n)
    x = [float(pooled.get(k, np.nan)) for k in model.feature_names]
    score = model.predict(x)
    if not math.isfinite(score):
        raise ValueError("LightGBM detector produced a non-finite score")
    return score


def score_game(
    model: LgbmDetector,
    per_move_preds: np.ndarray,
    attention: np.ndarray | None,
    moves_uci: Sequence[str],
    clock_seconds: Sequence[float],
    white_baseline: float,
    black_baseline: float,
) -> tuple[float, float]:
    """(white_score, black_score): each side scored in turn as the suspect."""
    scores = []
    for side, baseline in ((detector_module.WHITE, white_baseline), (detector_module.BLACK, black_baseline)):
        pp = detector_module.per_ply_arrays(per_move_preds, attention, moves_uci, clock_seconds, side, baseline)
        scores.append(score_side(model, pp))
    return scores[0], scores[1]

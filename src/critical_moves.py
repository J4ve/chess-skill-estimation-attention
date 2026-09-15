"""
Critical-move ranking for human fair-play review.

Ranks moves by how much attention the model paid to them combined with how
far they pulled the side's rating estimate from its baseline. This surfaces
the moves a reviewer should look at first, instead of making them read every
per-move row.
"""

from __future__ import annotations

from typing import Any

DEFAULT_TOP_K = 5
DEFAULT_MIN_PLY = 10

ANOMALY_UNAVAILABLE_WARNING = (
    "Critical moves are unavailable because the served checkpoint does not "
    "have an attention/anomaly branch."
)


def compute_critical_moves(
    per_move: list[dict[str, Any]],
    anomaly_available: bool,
    top_k: int = DEFAULT_TOP_K,
    min_ply: int = DEFAULT_MIN_PLY,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (critical_moves, warnings) for both sides.

    Each side is ranked independently by ``attention_weight * |deviation|``
    for that side, and the top ``top_k`` per side are returned. Plies before
    ``min_ply`` are excluded from ranking: the model has little context in
    the opening, so early plies otherwise dominate the list without being
    informative. When the anomaly branch is off, every deviation and
    attention weight is zero, so ranking would be meaningless; return an
    empty list with a warning instead.
    """
    if not anomaly_available:
        return [], [ANOMALY_UNAVAILABLE_WARNING]

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if min_ply < 0:
        raise ValueError("min_ply must be at least 0")

    candidates: list[dict[str, Any]] = []
    for record in per_move:
        if record["ply"] < min_ply:
            continue
        attention_weight = record.get("attention_weight") or 0.0
        for side, deviation_key in (("white", "white_deviation"), ("black", "black_deviation")):
            deviation = record.get(deviation_key) or 0.0
            weighted_score = attention_weight * abs(deviation)
            candidates.append(
                {
                    "ply": record["ply"],
                    "move": record.get("move"),
                    "side": side,
                    "attention_weight": round(attention_weight, 6),
                    "deviation": round(deviation, 4),
                    "weighted_score": round(weighted_score, 6),
                }
            )

    result: list[dict[str, Any]] = []
    for side in ("white", "black"):
        side_candidates = sorted(
            (c for c in candidates if c["side"] == side),
            key=lambda c: c["weighted_score"],
            reverse=True,
        )
        result.extend(side_candidates[:top_k])

    return result, []

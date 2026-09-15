"""
Baseline-rating resolution for suspicion scoring.

The anomaly module needs a pre-game baseline rating per side to measure
deviation against. Without one it silently falls back to the model's own
final-ply prediction, which makes the suspicion score compare the model
against itself and is close to meaningless. This module makes the
resolution order explicit and reports which source was used so the API
response (and the UI) can say so plainly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

BASELINE_SOURCE_REQUEST = "request"
BASELINE_SOURCE_PGN_HEADER = "pgn_header"
BASELINE_SOURCE_SELF_PREDICTION_FALLBACK = "self_prediction_fallback"


@dataclass
class BaselineResolution:
    value: float
    source: str
    warning: str | None


def _parse_header_rating(header_value: str | None) -> float | None:
    """Parse a PGN Elo header value, e.g. '1500'. Returns None if missing or non-numeric.

    Lichess exports use '?' for unrated/unknown ratings; that and any other
    non-numeric or non-positive value are treated as unusable.
    """
    if header_value is None:
        return None
    try:
        parsed = float(str(header_value).strip())
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def resolve_baseline(
    side_label: str,
    requested: float | None,
    header_value: str | None,
    self_prediction: Callable[[], float],
) -> BaselineResolution:
    """Resolve a side's baseline rating.

    Resolution order:
    1. ``requested`` (explicit value from the API caller), if given.
    2. The PGN header rating (e.g. WhiteElo/BlackElo), if present and numeric.
    3. The model's own final-ply prediction, with a warning that this makes
       the suspicion score for this side close to meaningless.
    """
    if requested is not None:
        return BaselineResolution(value=float(requested), source=BASELINE_SOURCE_REQUEST, warning=None)

    header_rating = _parse_header_rating(header_value)
    if header_rating is not None:
        return BaselineResolution(value=header_rating, source=BASELINE_SOURCE_PGN_HEADER, warning=None)

    value = float(self_prediction())
    warning = (
        f"{side_label} baseline was not supplied and the PGN header is missing or "
        f"non-numeric; falling back to the model's own final-ply prediction. The "
        f"{side_label.lower()} suspicion score compares the model against itself "
        f"and is close to meaningless until a real baseline is supplied."
    )
    return BaselineResolution(value=value, source=BASELINE_SOURCE_SELF_PREDICTION_FALLBACK, warning=warning)

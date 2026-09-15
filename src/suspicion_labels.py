"""
Plain-language labels for a suspicion score, resolved against a cutoffs file.

Deliberately generic: ``label_for_score`` and ``resolve_cutoffs`` take a score
and a cutoffs object shaped like ``src/static/suspicion_cutoffs.json``
(``{"overall": {"p75": ..., "p95": ...}, "by_time_control": {tc: {...}}}``), not
anything specific to S_att. A future trained detector can reuse this module by
shipping its own cutoffs file in the same shape.

The wording here intentionally never claims cheating or likelihood: it only
describes how a score compares with the ordinary human games the cutoffs were
computed from. See ``suspicion_cutoffs.json``'s own ``note`` field and the
README "Suspicion labels" section for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TYPICAL = "typical"
UNUSUAL = "unusual"
HIGHLY_UNUSUAL = "highly_unusual"

LABEL_TEXT = {
    TYPICAL: "Typical",
    UNUSUAL: "Unusual",
    HIGHLY_UNUSUAL: "Highly unusual: worth a human review",
}

CUTOFF_SOURCE_TIME_CONTROL = "time_control"
CUTOFF_SOURCE_OVERALL = "overall"


@dataclass
class ResolvedCutoffs:
    p75: float
    p95: float
    source: str  # "time_control" or "overall"


def resolve_cutoffs(cutoffs: dict[str, Any], time_control: str | None) -> ResolvedCutoffs:
    """Pick the p75/p95 pair to use for a game's time control.

    Uses the time-control-specific cutoffs when ``time_control`` has an entry
    that is not itself flagged ``use_overall`` (too few sides to be stable);
    falls back to the overall cutoffs otherwise, including when
    ``time_control`` is None or unrecognized.
    """
    by_tc = cutoffs.get("by_time_control") or {}
    entry = by_tc.get(time_control) if time_control else None
    if entry is not None and not entry.get("use_overall", False):
        return ResolvedCutoffs(p75=entry["p75"], p95=entry["p95"], source=CUTOFF_SOURCE_TIME_CONTROL)

    overall = cutoffs["overall"]
    return ResolvedCutoffs(p75=overall["p75"], p95=overall["p95"], source=CUTOFF_SOURCE_OVERALL)


def label_for_score(score: float, resolved: ResolvedCutoffs) -> str:
    """Classify a score as typical, unusual, or highly unusual.

    Boundaries: below p75 is typical; p75 up to and including p95 is unusual;
    above p95 is highly unusual.
    """
    if score > resolved.p95:
        return HIGHLY_UNUSUAL
    if score >= resolved.p75:
        return UNUSUAL
    return TYPICAL

from suspicion_labels import (
    CUTOFF_SOURCE_OVERALL,
    CUTOFF_SOURCE_TIME_CONTROL,
    HIGHLY_UNUSUAL,
    TYPICAL,
    UNUSUAL,
    label_for_score,
    resolve_cutoffs,
)

CUTOFFS = {
    "overall": {"p75": 200.0, "p95": 400.0},
    "by_time_control": {
        "bullet": {"p75": 220.0, "p95": 450.0, "use_overall": False},
        "ultrabullet": {"use_overall": True, "reason": "fewer than 300 sides"},
    },
}


def test_resolve_cutoffs_uses_time_control_entry():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert resolved.p75 == 220.0
    assert resolved.p95 == 450.0
    assert resolved.source == CUTOFF_SOURCE_TIME_CONTROL


def test_resolve_cutoffs_falls_back_when_marked_use_overall():
    resolved = resolve_cutoffs(CUTOFFS, "ultrabullet")
    assert resolved.p75 == 200.0
    assert resolved.p95 == 400.0
    assert resolved.source == CUTOFF_SOURCE_OVERALL


def test_resolve_cutoffs_falls_back_when_time_control_unknown():
    resolved = resolve_cutoffs(CUTOFFS, "chess960")
    assert resolved.source == CUTOFF_SOURCE_OVERALL


def test_resolve_cutoffs_falls_back_when_time_control_is_none():
    resolved = resolve_cutoffs(CUTOFFS, None)
    assert resolved.source == CUTOFF_SOURCE_OVERALL


def test_label_below_p75_is_typical():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert label_for_score(0.0, resolved) == TYPICAL
    assert label_for_score(219.99, resolved) == TYPICAL


def test_label_at_p75_boundary_is_unusual():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert label_for_score(220.0, resolved) == UNUSUAL


def test_label_between_p75_and_p95_is_unusual():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert label_for_score(300.0, resolved) == UNUSUAL


def test_label_at_p95_boundary_is_unusual_not_highly_unusual():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert label_for_score(450.0, resolved) == UNUSUAL


def test_label_above_p95_is_highly_unusual():
    resolved = resolve_cutoffs(CUTOFFS, "bullet")
    assert label_for_score(450.01, resolved) == HIGHLY_UNUSUAL
    assert label_for_score(1097.0, resolved) == HIGHLY_UNUSUAL


def test_label_uses_overall_cutoffs_for_unrecognized_time_control():
    resolved = resolve_cutoffs(CUTOFFS, "chess960")
    assert label_for_score(199.0, resolved) == TYPICAL
    assert label_for_score(200.0, resolved) == UNUSUAL
    assert label_for_score(400.01, resolved) == HIGHLY_UNUSUAL

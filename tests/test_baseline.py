from baseline import (
    BASELINE_SOURCE_PGN_HEADER,
    BASELINE_SOURCE_REQUEST,
    BASELINE_SOURCE_SELF_PREDICTION_FALLBACK,
    resolve_baseline,
)


def _fallback():
    return 1499.0


def test_request_value_wins_over_everything():
    resolution = resolve_baseline("White", 1800.0, "1500", _fallback)
    assert resolution.value == 1800.0
    assert resolution.source == BASELINE_SOURCE_REQUEST
    assert resolution.warning is None


def test_pgn_header_used_when_no_request():
    resolution = resolve_baseline("White", None, "1650", _fallback)
    assert resolution.value == 1650.0
    assert resolution.source == BASELINE_SOURCE_PGN_HEADER
    assert resolution.warning is None


def test_missing_header_falls_back_with_warning():
    resolution = resolve_baseline("Black", None, None, _fallback)
    assert resolution.value == 1499.0
    assert resolution.source == BASELINE_SOURCE_SELF_PREDICTION_FALLBACK
    assert resolution.warning is not None
    assert "Black" in resolution.warning


def test_non_numeric_header_falls_back():
    resolution = resolve_baseline("White", None, "?", _fallback)
    assert resolution.source == BASELINE_SOURCE_SELF_PREDICTION_FALLBACK
    assert resolution.warning is not None


def test_zero_or_negative_header_treated_as_missing():
    resolution = resolve_baseline("White", None, "0", _fallback)
    assert resolution.source == BASELINE_SOURCE_SELF_PREDICTION_FALLBACK

    resolution_negative = resolve_baseline("White", None, "-5", _fallback)
    assert resolution_negative.source == BASELINE_SOURCE_SELF_PREDICTION_FALLBACK


def test_header_with_whitespace_is_parsed():
    resolution = resolve_baseline("White", None, "  1700  ", _fallback)
    assert resolution.value == 1700.0
    assert resolution.source == BASELINE_SOURCE_PGN_HEADER

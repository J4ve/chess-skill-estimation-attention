from format_data import compute_time_spent, parse_time_control, time_control_bucket


def test_bullet_time_control():
    assert time_control_bucket("60+0") == "bullet"


def test_blitz_time_control():
    assert time_control_bucket("180+0") == "blitz"


def test_rapid_time_control():
    assert time_control_bucket("600+0") == "rapid"


def test_classical_time_control():
    assert time_control_bucket("1800+0") == "classical"


def test_ultrabullet_time_control():
    assert time_control_bucket("15+0") == "ultrabullet"


def test_increment_counted_toward_estimated_duration():
    # 60 base + 40 * 5s increment = 260s, which is blitz (179-478), not bullet.
    assert time_control_bucket("60+5") == "blitz"


def test_missing_header_is_none():
    assert time_control_bucket(None) is None
    assert time_control_bucket("") is None


def test_correspondence_dash_is_none():
    assert time_control_bucket("-") is None


def test_malformed_header_is_none():
    assert time_control_bucket("not-a-time-control") is None
    assert time_control_bucket("60") is None


def test_parse_time_control_normal():
    assert parse_time_control("180+0") == (180, 0)
    assert parse_time_control("60+2") == (60, 2)


def test_parse_time_control_missing_or_malformed():
    assert parse_time_control(None) == (None, None)
    assert parse_time_control("") == (None, None)
    assert parse_time_control("-") == (None, None)
    assert parse_time_control("60") == (None, None)


def test_compute_time_spent_normal_no_increment():
    # TimeControl "180+0": base 180, no increment.
    clocks = [175, 176, 168, 170]
    assert compute_time_spent(clocks, 180, 0) == [5, 4, 7, 6]


def test_compute_time_spent_with_increment():
    # TimeControl "60+2": each move's clock loss is offset by +2s increment.
    clocks = [59, 58, 60, 59]
    assert compute_time_spent(clocks, 60, 2) == [3, 4, 1, 1]


def test_compute_time_spent_missing_clocks_is_empty():
    assert compute_time_spent([], 180, 0) == []


def test_compute_time_spent_missing_time_control_nulls_first_moves_only():
    # No TimeControl header (base=None): each side's first move has no earlier
    # own clock to compare against, so it's null; later plies still compute
    # from the directly observed previous clock two plies back.
    clocks = [100, 95, 90]
    assert compute_time_spent(clocks, None, None) == [None, None, 10]

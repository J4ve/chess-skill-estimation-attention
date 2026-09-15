from format_data import time_control_bucket


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

import pytest

from critical_moves import ANOMALY_UNAVAILABLE_WARNING, compute_critical_moves


def _record(ply, move, attention_weight, white_deviation, black_deviation):
    return {
        "ply": ply,
        "move": move,
        "attention_weight": attention_weight,
        "white_deviation": white_deviation,
        "black_deviation": black_deviation,
    }


def test_anomaly_unavailable_returns_empty_with_warning():
    per_move = [_record(1, "e4", 0.5, 100.0, 50.0)]
    moves, warnings = compute_critical_moves(per_move, anomaly_available=False)
    assert moves == []
    assert warnings == [ANOMALY_UNAVAILABLE_WARNING]


def test_ranks_by_attention_times_deviation_per_side():
    per_move = [
        _record(1, "e4", 0.1, 10.0, 5.0),
        _record(2, "e5", 0.9, 20.0, 200.0),
        _record(3, "Nf3", 0.5, 500.0, 1.0),
    ]
    moves, warnings = compute_critical_moves(per_move, anomaly_available=True, top_k=2, min_ply=0)
    assert warnings == []

    white_moves = [m for m in moves if m["side"] == "white"]
    black_moves = [m for m in moves if m["side"] == "black"]
    assert len(white_moves) == 2
    assert len(black_moves) == 2

    # ply 3: 0.5*500=250 > ply 1: 0.1*10=1.0 > ply 2: 0.9*20=18 -> top2 white = [3, 2]
    assert [m["ply"] for m in white_moves] == [3, 2]
    # ply 2: 0.9*200=180 is the largest black weighted score
    assert black_moves[0]["ply"] == 2


def test_top_k_limits_results_per_side():
    per_move = [_record(i, f"m{i}", 1.0, float(i), float(i)) for i in range(1, 11)]
    moves, _ = compute_critical_moves(per_move, anomaly_available=True, top_k=3, min_ply=0)
    assert sum(1 for m in moves if m["side"] == "white") == 3
    assert sum(1 for m in moves if m["side"] == "black") == 3


def test_negative_deviation_uses_absolute_value():
    per_move = [
        _record(1, "e4", 1.0, -300.0, 0.0),
        _record(2, "e5", 1.0, 10.0, 0.0),
    ]
    moves, _ = compute_critical_moves(per_move, anomaly_available=True, top_k=1, min_ply=0)
    white_top = next(m for m in moves if m["side"] == "white")
    assert white_top["ply"] == 1
    assert white_top["weighted_score"] == pytest.approx(300.0)


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        compute_critical_moves([_record(1, "e4", 1.0, 1.0, 1.0)], anomaly_available=True, top_k=0)


def test_min_ply_excludes_opening_moves_by_default():
    per_move = [
        _record(1, "e4", 1.0, 500.0, 500.0),
        _record(9, "Nf3", 1.0, 400.0, 400.0),
        _record(10, "Bb5", 1.0, 10.0, 10.0),
        _record(15, "O-O", 1.0, 20.0, 20.0),
    ]
    moves, warnings = compute_critical_moves(per_move, anomaly_available=True, top_k=5)
    assert warnings == []
    plies = {m["ply"] for m in moves}
    assert plies == {10, 15}


def test_min_ply_is_configurable():
    per_move = [
        _record(1, "e4", 1.0, 500.0, 500.0),
        _record(5, "Nf3", 1.0, 400.0, 400.0),
    ]
    moves, _ = compute_critical_moves(per_move, anomaly_available=True, top_k=5, min_ply=0)
    plies = {m["ply"] for m in moves}
    assert plies == {1, 5}

    moves, _ = compute_critical_moves(per_move, anomaly_available=True, top_k=5, min_ply=5)
    plies = {m["ply"] for m in moves}
    assert plies == {5}


def test_invalid_min_ply_raises():
    with pytest.raises(ValueError):
        compute_critical_moves([_record(1, "e4", 1.0, 1.0, 1.0)], anomaly_available=True, min_ply=-1)


def test_min_ply_excluding_everything_returns_empty_no_error():
    per_move = [_record(1, "e4", 1.0, 500.0, 500.0)]
    moves, warnings = compute_critical_moves(per_move, anomaly_available=True, min_ply=10)
    assert moves == []
    assert warnings == []

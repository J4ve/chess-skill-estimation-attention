"""
Error-path tests for PGN input handling that do not require the model
checkpoint: these exercise the parsing step of api._pgn_to_tensor_inputs,
which raises before any tensor/model code runs.
"""

import pytest

from api import _pgn_to_tensor_inputs

NO_CLOCK_PGN = (
    '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1500"]\n\n1. e4 e5 2. Nf3 Nc6 *'
)

WITH_CLOCK_PGN = (
    '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1500"]\n[TimeControl "300+0"]\n\n'
    "1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} "
    "2. Nf3 {[%clk 0:04:58]} Nc6 {[%clk 0:04:58]} *"
)

UNPARSEABLE_TEXT = "this is not a pgn file at all, just prose."

FROM_POSITION_PGN = (
    '[Event "Rated Rapid Arena"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1500"]\n[Variant "From Position"]\n[SetUp "1"]\n'
    '[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"]\n\n'
    "1. Nf3 *"
)

CHESS960_PGN = (
    '[Event "Rated Chess960"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1500"]\n[Variant "Chess960"]\n[SetUp "1"]\n'
    '[FEN "nbbrknrq/pppppppp/8/8/8/8/PPPPPPPP/NBBRKNRQ w KQkq - 0 1"]\n\n'
    "1. Nf3 *"
)


def test_pgn_without_clocks_raises_value_error():
    with pytest.raises(ValueError, match="clock"):
        _pgn_to_tensor_inputs(NO_CLOCK_PGN)


def test_from_position_pgn_raises_custom_position_error_not_clock_error():
    # A "From Position" thematic arena also lacks clocks on its first move, but
    # the custom-position message must win: it is the more useful, specific
    # reason, and it must be checked before the clock check (see api.py's
    # _pgn_to_tensor_inputs and live.py's LiveGamePrefix.seed_from_pgn).
    with pytest.raises(ValueError, match="custom position or variant"):
        _pgn_to_tensor_inputs(FROM_POSITION_PGN)


def test_chess960_pgn_raises_custom_position_error():
    with pytest.raises(ValueError, match="custom position or variant"):
        _pgn_to_tensor_inputs(CHESS960_PGN)


def test_empty_pgn_raises_value_error():
    with pytest.raises(ValueError):
        _pgn_to_tensor_inputs("")


def test_unparseable_text_raises_value_error():
    # python-chess is lenient with garbage input (it will not find any moves
    # or clocks in prose), so this still surfaces as a clean ValueError.
    with pytest.raises(ValueError):
        _pgn_to_tensor_inputs(UNPARSEABLE_TEXT)


def test_pgn_with_clocks_parses_successfully():
    positions, clocks, lengths, moves, san_moves, headers, raw_clock_seconds = _pgn_to_tensor_inputs(WITH_CLOCK_PGN)
    assert positions.shape[1] == 4  # 4 plies
    assert clocks.shape[1] == 4
    assert lengths.tolist() == [4]
    assert moves == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert san_moves == ["e4", "e5", "Nf3", "Nc6"]
    assert headers.get("WhiteElo") == "1500"
    assert raw_clock_seconds == [300, 300, 298, 298]

"""
Tests for src/live.py: NDJSON parsing, prefix-based PGN bookkeeping, and the
stream_game/stream_tv orchestration, all without any real network access
(httpx.MockTransport stands in for Lichess).
"""

import asyncio
import json

import chess
import httpx
import pytest

from live import (
    LiveGamePrefix,
    LiveStreamError,
    build_update_event,
    classify_game_stream_message,
    clock_seconds_for_ply,
    headers_from_tv_players,
    iter_ndjson_lines,
    parse_ndjson_line,
    seconds_to_clock_str,
    stream_game,
    stream_tv,
)

FIXTURE_PGN = (
    '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1600"]\n[Result "*"]\n[TimeControl "300+0"]\n\n'
    "1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} "
    "2. Nf3 {[%clk 0:04:58]} Nc6 {[%clk 0:04:58]} *"
)

FINISHED_PGN = FIXTURE_PGN.replace('[Result "*"]', '[Result "1-0"]').replace(" *", " 1-0")
FIXTURE_UCIS = ("e2e4", "e7e5", "g1f3", "b8c6")


def _fen_after(*ucis):
    """A real Lichess-style TV feed FEN for the position after these moves."""
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board.fen()


def run(coro):
    return asyncio.run(coro)


# --- pure parsing/formatting helpers -----------------------------------------


def test_parse_ndjson_line_skips_blank():
    assert parse_ndjson_line("   \n") is None
    assert parse_ndjson_line("") is None


def test_parse_ndjson_line_parses_json():
    assert parse_ndjson_line('{"a": 1}') == {"a": 1}


def test_classify_meta_position_final():
    assert classify_game_stream_message({"id": "x", "players": {}}) == "meta"
    assert classify_game_stream_message({"fen": "...", "wc": 5, "bc": 5}) == "position"
    assert classify_game_stream_message({"fen": "...", "status": {"name": "mate"}}) == "final"


def test_seconds_to_clock_str_formats_hms():
    assert seconds_to_clock_str(0) == "0:00:00"
    assert seconds_to_clock_str(65) == "0:01:05"
    assert seconds_to_clock_str(3661) == "1:01:01"


def test_seconds_to_clock_str_clamps_negative():
    assert seconds_to_clock_str(-5) == "0:00:00"


def test_clock_seconds_for_ply_odd_is_white():
    assert clock_seconds_for_ply(1, wc=59, bc=60) == 59
    assert clock_seconds_for_ply(3, wc=55, bc=58) == 55


def test_clock_seconds_for_ply_even_is_black():
    assert clock_seconds_for_ply(2, wc=59, bc=58) == 58
    assert clock_seconds_for_ply(4, wc=55, bc=57) == 57


def test_headers_from_tv_players_maps_colors():
    players = [
        {"color": "white", "user": {"name": "alice"}, "rating": 2100},
        {"color": "black", "user": {"name": "bob"}, "rating": 1980},
    ]
    headers = headers_from_tv_players(players)
    assert headers["White"] == "alice"
    assert headers["WhiteElo"] == "2100"
    assert headers["Black"] == "bob"
    assert headers["BlackElo"] == "1980"
    assert headers["Result"] == "*"


# --- LiveGamePrefix -----------------------------------------------------------


def test_prefix_add_move_builds_valid_pgn():
    prefix = LiveGamePrefix({"White": "A", "Black": "B", "Result": "*"}, max_plies=100)
    prefix.add_move("e2e4", 300)
    prefix.add_move("e7e5", 299)
    assert prefix.ply_count == 2
    pgn_text = prefix.to_pgn_text()
    assert "1. e4 {[%clk 0:05:00]}" in pgn_text
    assert "e5 {[%clk 0:04:59]}" in pgn_text


def test_prefix_seed_from_pgn_reads_existing_moves():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)
    assert prefix.ply_count == 4
    assert prefix.headers["WhiteElo"] == "1500"
    # A move made after seeding continues from the seeded position correctly.
    prefix.add_move("f1c4", 295)
    assert prefix.ply_count == 5
    assert "5. Bc4" not in prefix.to_pgn_text()  # it's move 3 (Bc4 is White's 3rd move)
    assert "3. Bc4" in prefix.to_pgn_text()


def test_prefix_seed_from_pgn_with_zero_moves_seeds_headers_only():
    zero_move_pgn = '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n*'
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(zero_move_pgn)
    assert prefix.ply_count == 0
    assert prefix.headers["White"] == "A"


def test_prefix_seed_from_pgn_without_clocks_raises():
    no_clock_pgn = '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. e4 e5 *'
    prefix = LiveGamePrefix({}, max_plies=100)
    with pytest.raises(ValueError, match="clock"):
        prefix.seed_from_pgn(no_clock_pgn)


def test_prefix_seed_from_pgn_from_position_raises_custom_position_message():
    from_position_pgn = (
        '[Event "Rated Rapid Arena"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n'
        '[Variant "From Position"]\n[SetUp "1"]\n'
        '[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"]\n\n'
        "1. Nf3 *"
    )
    prefix = LiveGamePrefix({}, max_plies=100)
    with pytest.raises(ValueError, match="custom position or variant"):
        prefix.seed_from_pgn(from_position_pgn)


def test_prefix_seed_from_pgn_chess960_raises_custom_position_message():
    chess960_pgn = (
        '[Event "Rated Chess960"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n'
        '[Variant "Chess960"]\n[SetUp "1"]\n'
        '[FEN "nbbrknrq/pppppppp/8/8/8/8/PPPPPPPP/NBBRKNRQ w KQkq - 0 1"]\n\n'
        "1. Nf3 *"
    )
    prefix = LiveGamePrefix({}, max_plies=100)
    with pytest.raises(ValueError, match="custom position or variant"):
        prefix.seed_from_pgn(chess960_pgn)


def test_prefix_add_move_caps_at_max_plies():
    prefix = LiveGamePrefix({"Result": "*"}, max_plies=2)
    assert prefix.add_move("e2e4", 300) is True
    assert prefix.add_move("e7e5", 300) is True
    assert prefix.truncated is False
    assert prefix.add_move("g1f3", 298) is False
    assert prefix.truncated is True
    assert prefix.ply_count == 2


# --- build_update_event --------------------------------------------------------


def _fake_run_inference(pgn_text, white_baseline, black_baseline, top_k, min_ply):
    return {"status": "ok", "per_move": [{"ply": 1}], "warnings": []}


def test_build_update_event_returns_none_for_empty_prefix():
    prefix = LiveGamePrefix({"Result": "*"}, max_plies=100)
    event = build_update_event(prefix, _fake_run_inference, 5, 10, None, None, "abcd1234")
    assert event is None


def test_build_update_event_wraps_result_with_metadata():
    prefix = LiveGamePrefix({"Result": "*"}, max_plies=100)
    prefix.add_move("e2e4", 300)
    event = build_update_event(prefix, _fake_run_inference, 5, 10, 1500.0, 1600.0, "abcd1234")
    assert event.startswith("data: ")
    payload = json.loads(event[len("data: ") :].strip())
    assert payload["type"] == "update"
    assert payload["result"]["lichess_game_id"] == "abcd1234"
    assert payload["result"]["live"] is True
    assert "inference_ms" in payload["result"]


def test_build_update_event_adds_truncation_warning():
    prefix = LiveGamePrefix({"Result": "*"}, max_plies=1)
    prefix.add_move("e2e4", 300)
    prefix.add_move("e7e5", 300)  # no-op beyond the cap, sets truncated
    event = build_update_event(prefix, _fake_run_inference, 5, 10, None, None, "abcd1234")
    payload = json.loads(event[len("data: ") :].strip())
    assert any("capped" in w for w in payload["result"]["warnings"])


# --- iter_ndjson_lines / mocked transport --------------------------------------


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_iter_ndjson_lines_yields_parsed_objects():
    body = '{"a": 1}\n\n{"a": 2}\n'

    def handler(request):
        return httpx.Response(200, text=body)

    async def body_coro():
        async with _mock_client(handler) as client:
            return [msg async for msg in iter_ndjson_lines(client, "https://example.test/stream")]

    assert run(body_coro()) == [{"a": 1}, {"a": 2}]


def test_iter_ndjson_lines_maps_404():
    def handler(request):
        return httpx.Response(404)

    async def body_coro():
        async with _mock_client(handler) as client:
            async for _ in iter_ndjson_lines(client, "https://example.test/stream"):
                pass

    with pytest.raises(LiveStreamError) as excinfo:
        run(body_coro())
    assert excinfo.value.status_code == 404


# --- stream_game / stream_tv orchestration -------------------------------------


def _ndjson(*objs) -> str:
    return "".join(json.dumps(o) + "\n" for o in objs)


def test_stream_game_already_finished_runs_once_no_stream():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FINISHED_PGN)

    async def body_coro():
        events = []
        async for chunk in stream_game("abcd1234", prefix, _fake_run_inference, 5, 10, None, None):
            events.append(chunk)
        return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    assert payloads[0]["type"] == "status"
    assert payloads[0]["state"] == "finished"
    assert payloads[1]["type"] == "update"


def test_stream_game_processes_new_moves_and_detects_finish():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)  # already has e4 e5 Nf3 Nc6 (4 plies)

    backlog = [
        {"id": "abcd1234", "players": {}},
        {"fen": "startpos", "wc": 300, "bc": 300},
        {"fen": "f1", "lm": "e2e4", "wc": 300, "bc": 300},
        {"fen": "f2", "lm": "e7e5", "wc": 300, "bc": 300},
        {"fen": "f3", "lm": "g1f3", "wc": 298, "bc": 300},
        {"fen": "f4", "lm": "b8c6", "wc": 298, "bc": 297},
        # one new live move beyond the seeded prefix:
        {"fen": "f5", "lm": "f1b5", "wc": 296, "bc": 297},
        {"fen": "f6", "status": {"id": 1, "name": "mate"}, "winner": "white"},
    ]

    def handler(request):
        return httpx.Response(200, text=_ndjson(*backlog))

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_game(
                "abcd1234", prefix, _fake_run_inference, 5, 10, None, None, client=client
            ):
                events.append(chunk)
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    kinds = [(p["type"], p.get("state")) for p in payloads]

    assert ("status", "connected") in kinds
    update_events = [p for p in payloads if p["type"] == "update"]
    # one from the initial seeded-prefix emit, one for the single new live move
    assert len(update_events) == 2
    assert ("status", "finished") in kinds
    assert prefix.ply_count == 5  # 4 seeded + 1 new live move


def test_stream_game_reconnects_once_then_succeeds():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)  # 4 plies already known

    dropped_stream = [
        {"id": "abcd1234", "players": {}},
        {"fen": "startpos", "wc": 300, "bc": 300},
        {"fen": "f1", "lm": "e2e4", "wc": 300, "bc": 300},
        # stream ends here with no "final" message: simulates a drop
    ]
    full_stream = [
        {"id": "abcd1234", "players": {}},
        {"fen": "startpos", "wc": 300, "bc": 300},
        {"fen": "f1", "lm": "e2e4", "wc": 300, "bc": 300},
        {"fen": "f2", "lm": "e7e5", "wc": 300, "bc": 300},
        {"fen": "f3", "lm": "g1f3", "wc": 298, "bc": 300},
        {"fen": "f4", "lm": "b8c6", "wc": 298, "bc": 297},
        {"fen": "f6", "status": {"id": 1, "name": "resign"}, "winner": "black"},
    ]
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        body = dropped_stream if call_count["n"] == 1 else full_stream
        return httpx.Response(200, text=_ndjson(*body))

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_game(
                "abcd1234", prefix, _fake_run_inference, 5, 10, None, None, client=client
            ):
                events.append(chunk)
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    states = [p.get("state") for p in payloads if p["type"] == "status"]

    assert "reconnecting" in states
    assert "finished" in states
    assert call_count["n"] == 2
    assert prefix.ply_count == 4  # no new moves beyond the already-seeded ones this time


def test_stream_game_gives_up_after_reconnect_fails_again():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)

    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_game(
                "abcd1234", prefix, _fake_run_inference, 5, 10, None, None, client=client
            ):
                events.append(chunk)
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    assert payloads[-1]["type"] == "error"


def test_stream_tv_skips_unsupported_game_and_waits_for_next_featured():
    # Regression test: an illegal/unsupported move (e.g. a variant game slipping
    # into a "standard" TV channel) must not crash the generator. It should be
    # skipped, and a later "featured" switch to a supported game should still work.
    tv_feed = [
        {"t": "featured", "d": {"id": "bad00001", "players": [], "fen": "x"}},
        {"t": "fen", "d": {"fen": "y", "lm": "e4d5", "wc": 55, "bc": 57}},  # illegal: e4 is empty at game start
        {"t": "featured", "d": {"id": "wxyz9876", "players": [{"color": "white"}, {"color": "black"}], "fen": "x"}},
        {"t": "fen", "d": {"fen": _fen_after("e2e4"), "lm": "e2e4", "wc": 55, "bc": 57}},
    ]

    def handler(request):
        return httpx.Response(200, text=_ndjson(*tv_feed))

    async def fake_fetch_export(game_id):
        raise LiveStreamError(404, "not found yet")  # force seed_from_pgn's except-branch

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_tv(fake_fetch_export, _fake_run_inference, 5, 10, 100, client=client):
                events.append(chunk)
                if len(events) >= 6:  # avoid the reconnect tail; we only care about the recovery
                    break
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    update_events = [p for p in payloads if p["type"] == "update"]
    assert all(p["result"]["lichess_game_id"] != "bad00001" for p in update_events)
    assert any(p["result"]["lichess_game_id"] == "wxyz9876" for p in update_events)


def test_stream_game_yields_error_on_unsupported_move():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)

    backlog = [
        {"id": "abcd1234", "players": {}},
        {"fen": "startpos", "wc": 300, "bc": 300},
        {"fen": "f1", "lm": "e2e4", "wc": 300, "bc": 300},
        {"fen": "f2", "lm": "e7e5", "wc": 300, "bc": 300},
        {"fen": "f3", "lm": "g1f3", "wc": 298, "bc": 300},
        {"fen": "f4", "lm": "b8c6", "wc": 298, "bc": 297},
        {"fen": "f5", "lm": "f4g5", "wc": 296, "bc": 297},  # illegal: f4 is empty in this position
    ]

    def handler(request):
        return httpx.Response(200, text=_ndjson(*backlog))

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_game(
                "abcd1234", prefix, _fake_run_inference, 5, 10, None, None, client=client
            ):
                events.append(chunk)
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    assert payloads[-1]["type"] == "error"


def test_stream_tv_seeds_from_export_and_follows_moves():
    # A MockTransport response body is inherently finite, so the real TV feed
    # (which never sends a "final" message and would otherwise keep going
    # forever) looks to our code like it "ended" once this canned body is
    # exhausted -> it correctly tries to reconnect once. Make the second
    # connection attempt fail outright so this test only has to check the
    # first pass's events, not the reconnect-and-replay-the-backlog path
    # (that path is covered by the stream_game reconnect test instead).
    tv_feed = [
        {"t": "featured", "d": {"id": "wxyz9876", "players": [{"color": "white"}, {"color": "black"}], "fen": "x"}},
        {"t": "fen", "d": {"fen": _fen_after(*FIXTURE_UCIS, "f1c4"), "lm": "f1c4", "wc": 55, "bc": 57}},
    ]
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, text=_ndjson(*tv_feed))
        return httpx.Response(404)

    async def fake_fetch_export(game_id):
        return FIXTURE_PGN  # 4 plies already played when we "joined"

    async def body_coro():
        async with _mock_client(handler) as client:
            events = []
            async for chunk in stream_tv(fake_fetch_export, _fake_run_inference, 5, 10, 100, client=client):
                events.append(chunk)
            return events

    events = run(body_coro())
    payloads = [json.loads(e[len("data: ") :].strip()) for e in events]
    update_events = [p for p in payloads if p["type"] == "update"]
    # one for the seeded prefix on "featured", one for the new live move
    assert len(update_events) == 2
    assert all(p["result"]["lichess_game_id"] == "wxyz9876" for p in update_events)
    assert payloads[-1]["type"] == "error"


def test_stream_tv_reports_custom_position_game_and_waits_for_next_featured():
    from_position_pgn = (
        '[Event "Rated Rapid Arena"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n'
        '[Variant "From Position"]\n[SetUp "1"]\n'
        '[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"]\n\n'
        "1. Nf3 {[%clk 0:05:00]} *"
    )
    tv_feed = [
        {"t": "featured", "d": {"id": "thema001", "players": [], "fen": "x"}},
        {"t": "fen", "d": {"fen": "y", "lm": "b8c6", "wc": 55, "bc": 57}},
        {"t": "featured", "d": {"id": "wxyz9876", "players": [{"color": "white"}, {"color": "black"}], "fen": "x"}},
    ]
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, text=_ndjson(*tv_feed))
        return httpx.Response(404)

    async def fake_fetch_export(game_id):
        return from_position_pgn if game_id == "thema001" else FIXTURE_PGN

    async def body_coro():
        async with _mock_client(handler) as client:
            return [chunk async for chunk in stream_tv(fake_fetch_export, _fake_run_inference, 5, 10, 100, client=client)]

    payloads = [json.loads(e[len("data: ") :].strip()) for e in run(body_coro())]
    unsupported = [p for p in payloads if p.get("state") == "unsupported"]
    assert len(unsupported) == 1
    assert "custom position or variant" in unsupported[0]["message"]
    update_ids = [p["result"]["lichess_game_id"] for p in payloads if p["type"] == "update"]
    assert update_ids == ["wxyz9876"]


@pytest.mark.parametrize(
    "variant_headers",
    [
        '[Variant "From Position"]\n[SetUp "1"]\n[FEN "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"]\n',
        '[Variant "Chess960"]\n[SetUp "1"]\n[FEN "nbbrknrq/pppppppp/8/8/8/8/PPPPPPPP/NBBRKNRQ w KQkq - 0 1"]\n',
    ],
)
def test_live_stream_endpoint_sends_custom_position_error_as_sse_event(monkeypatch, variant_headers):
    import api

    pgn = f'[Event "Arena"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n{variant_headers}\n1. Nf3 {{[%clk 0:05:00]}} *'

    async def fake_fetch(game_id):
        return pgn

    monkeypatch.setattr(api, "fetch_game_pgn", fake_fetch)

    async def body_coro():
        return [chunk async for chunk in api._live_stream_game_events("bhjiWekv", 5, 10, None, None)]

    payloads = [json.loads(e[len("data: ") :].strip()) for e in run(body_coro())]
    assert len(payloads) == 1
    assert payloads[0]["type"] == "error"
    assert "custom position or variant" in payloads[0]["detail"]


def test_stream_tv_ignores_repeated_featured_event_for_same_game():
    tv_feed = [
        {"t": "featured", "d": {"id": "wxyz9876", "players": [{"color": "white"}, {"color": "black"}], "fen": "x"}},
        {"t": "featured", "d": {"id": "wxyz9876", "players": [{"color": "white"}, {"color": "black"}], "fen": "x"}},
        {"t": "fen", "d": {"fen": _fen_after(*FIXTURE_UCIS, "f1c4"), "lm": "f1c4", "wc": 55, "bc": 57}},
    ]
    call_count = {"n": 0}

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, text=_ndjson(*tv_feed))
        return httpx.Response(404)

    async def fake_fetch_export(game_id):
        return FIXTURE_PGN

    async def body_coro():
        async with _mock_client(handler) as client:
            return [chunk async for chunk in stream_tv(fake_fetch_export, _fake_run_inference, 5, 10, 100, client=client)]

    payloads = [json.loads(e[len("data: ") :].strip()) for e in run(body_coro())]
    connected = [p for p in payloads if p.get("state") == "connected"]
    assert len(connected) == 1
    # one update for the seeded prefix, one for the live move; none for the repeat
    assert len([p for p in payloads if p["type"] == "update"]) == 2


def test_prefix_add_move_rejects_illegal_move():
    prefix = LiveGamePrefix({"Result": "*"}, max_plies=100)
    with pytest.raises(ValueError, match="not legal"):
        prefix.add_move("e2e5", 300)
    assert prefix.ply_count == 0


def test_prefix_find_bridge_finds_missing_plies():
    prefix = LiveGamePrefix({}, max_plies=100)
    prefix.seed_from_pgn(FIXTURE_PGN)
    target = chess.Board(_fen_after(*FIXTURE_UCIS, "f1c4", "g8f6")).board_fen()
    assert prefix.find_bridge(target) == ["f1c4", "g8f6"]
    assert prefix.ply_count == 4  # searching does not change the prefix
    assert prefix.find_bridge(chess.Board(_fen_after("d2d4")).board_fen()) is None
    assert prefix.find_bridge("not-a-fen") is None


def test_stream_tv_bridges_plies_missing_from_a_lagging_export():
    # Lichess's export lags the TV feed: when TV features the game, the export
    # has 4 plies but the game is at ply 5, and the next feed move is ply 6.
    tv_feed = [
        {
            "t": "featured",
            "d": {
                "id": "wxyz9876",
                "players": [{"color": "white", "seconds": 290}, {"color": "black", "seconds": 295}],
                "fen": _fen_after(*FIXTURE_UCIS, "f1c4"),
            },
        },
        {"t": "fen", "d": {"fen": _fen_after(*FIXTURE_UCIS, "f1c4", "g8f6"), "lm": "g8f6", "wc": 290, "bc": 292}},
    ]
    call_count = {"n": 0}
    scored_pgns = []

    def handler(request):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, text=_ndjson(*tv_feed))
        return httpx.Response(404)

    async def fake_fetch_export(game_id):
        return FIXTURE_PGN

    def recording_run_inference(pgn_text, *args):
        scored_pgns.append(pgn_text)
        return _fake_run_inference(pgn_text, *args)

    async def body_coro():
        async with _mock_client(handler) as client:
            return [
                chunk
                async for chunk in stream_tv(fake_fetch_export, recording_run_inference, 5, 10, 100, client=client)
            ]

    payloads = [json.loads(e[len("data: ") :].strip()) for e in run(body_coro())]
    assert not any("Lost sync" in (p.get("message") or "") for p in payloads)
    assert len([p for p in payloads if p["type"] == "update"]) == 2
    assert "3. Bc4 {[%clk 0:04:50]}" in scored_pgns[0]
    assert "Nf6 {[%clk 0:04:52]}" in scored_pgns[1]

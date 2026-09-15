"""
Live game streaming and prefix-based inference bookkeeping.

The model's BiLSTM is not incremental (see README "Limitations"): scoring
ply *t* means re-running the whole model over plies 1..*t*. Live mode makes
this workable by re-running the same inference pipeline the batch endpoints
use (injected here as ``run_inference_fn``, normally ``api._run_inference``)
on the growing PGN prefix each time a new move arrives, taking only the
estimate at the final ply of that run. Earlier chart points are never
revised once plotted; see ``api.py`` and the README "Live mode" section.

Two Lichess NDJSON sources feed this, each verified against the live API on
2026-09-15 (see AGENTS.md for the verified message shapes):

* ``GET /api/stream/game/{id}``: one metadata line with no "fen" key, then
  one position line per ply *from the start of the game* (replayed as
  backlog even if the game is already in progress) with an optional "lm"
  (last-move UCI, absent only on the initial reset line) and "wc"/"bc"
  clocks in integer seconds, then a final metadata line carrying "status"
  and "winner" when the game ends.
* ``GET /api/tv/feed``: only the *current* snapshot on connect (a
  "featured" event with id/players/fen, no backlog), then live "{t:fen}"
  updates, and a fresh "featured" event whenever TV switches games.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import chess
import chess.pgn
import httpx

from format_data import parse_game

USER_AGENT = "RatingNet-Prototype/0.1 (+https://github.com/J4ve/RatingNet)"
STREAM_GAME_URL_TEMPLATE = "https://lichess.org/api/stream/game/{game_id}"
TV_FEED_URL = "https://lichess.org/api/tv/feed"
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)

# Initial connection plus one reconnect attempt, per the live-mode brief.
MAX_STREAM_ATTEMPTS = 2

RunInferenceFn = Callable[[str, float | None, float | None, int, int], dict[str, Any]]
FetchExportPgnFn = Callable[[str], Awaitable[str]]


class LiveStreamError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def parse_ndjson_line(line: str) -> dict[str, Any] | None:
    """Parse one NDJSON line, returning None for a blank keep-alive line."""
    stripped = line.strip()
    if not stripped:
        return None
    return json.loads(stripped)


def classify_game_stream_message(msg: dict[str, Any]) -> str:
    """Classify a message from GET /api/stream/game/{id}.

    Returns "final" (game-over metadata, carries "status"/"winner"),
    "position" (a fen/wc/bc update, with "lm" for every ply but the initial
    reset), or "meta" (the connection-opening game-info line).
    """
    if "status" in msg:
        return "final"
    if "fen" in msg:
        return "position"
    return "meta"


def seconds_to_clock_str(seconds: int) -> str:
    """Format integer seconds as Lichess's 'H:MM:SS' PGN clock-comment string."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def clock_seconds_for_ply(ply_number: int, wc: int, bc: int) -> int:
    """Pick the mover's own remaining-time field for the ply that was just made.

    Odd plies are White's moves (so White's own clock, "wc", reflects the
    time White had left right after making that move); even plies are
    Black's, using "bc" the same way. This matches how Lichess attaches a
    %clk comment to each half-move in its PGN export.
    """
    return wc if ply_number % 2 == 1 else bc


class LiveGamePrefix:
    """Tracks a growing move/clock prefix for one live game as PGN text.

    Rendering the prefix back out as PGN text (with %clk comments) lets the
    existing batch inference pipeline run over it unchanged, instead of
    duplicating tensor-building logic here.
    """

    def __init__(self, headers: dict[str, str], max_plies: int):
        self.headers = dict(headers)
        self.max_plies = max_plies
        self._board = chess.Board()
        self._sans: list[str] = []
        self._clocks: list[str] = []
        self.truncated = False

    @property
    def ply_count(self) -> int:
        return len(self._sans)

    def seed_from_pgn(self, pgn_text: str) -> None:
        """Seed the prefix from the already-played portion of the game (export PGN)."""
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("Could not parse PGN text")
        self.headers = dict(game.headers)
        total_moves = sum(1 for _ in game.mainline_moves())
        if total_moves == 0:
            return
        game_info = parse_game(game, max_plies=self.max_plies)
        if game_info is None:
            raise ValueError(
                "This game has no per-move [%clk ...] clock annotations; live "
                "analysis requires clocks, same as batch analysis."
            )
        for uci, san, clock in zip(game_info["Moves"], game_info["SAN"], game_info["Clocks"]):
            self._board.push(chess.Move.from_uci(uci))
            self._sans.append(san)
            self._clocks.append(clock)
        if total_moves > self.max_plies:
            self.truncated = True

    def add_move(self, uci: str, clock_seconds: int) -> bool:
        """Append one live move. Returns False (no-op) once the ply cap is hit."""
        if self.ply_count >= self.max_plies:
            self.truncated = True
            return False
        move = chess.Move.from_uci(uci)
        san = self._board.san(move)
        self._board.push(move)
        self._sans.append(san)
        self._clocks.append(seconds_to_clock_str(clock_seconds))
        return True

    def to_pgn_text(self) -> str:
        header_lines = "".join(f'[{k} "{v}"]\n' for k, v in self.headers.items())
        move_parts = []
        for i, (san, clock) in enumerate(zip(self._sans, self._clocks)):
            if i % 2 == 0:
                move_parts.append(f"{i // 2 + 1}. {san} {{[%clk {clock}]}}")
            else:
                move_parts.append(f"{san} {{[%clk {clock}]}}")
        result = self.headers.get("Result") or "*"
        move_text = " ".join([*move_parts, result]) if move_parts else result
        return f"{header_lines}\n{move_text}\n"


async def iter_ndjson_lines(
    client: httpx.AsyncClient, url: str, params: dict[str, str] | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Stream an NDJSON endpoint, yielding one parsed object per non-blank line."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/x-ndjson"}
    async with client.stream("GET", url, params=params, headers=headers, timeout=STREAM_TIMEOUT) as response:
        if response.status_code == 404:
            raise LiveStreamError(404, f"Lichess stream '{url}' was not found")
        if response.status_code == 429:
            raise LiveStreamError(429, "Lichess API rate limit exceeded; wait a moment and try again")
        if response.status_code != 200:
            raise LiveStreamError(502, f"Lichess stream returned an unexpected status ({response.status_code})")
        async for line in response.aiter_lines():
            msg = parse_ndjson_line(line)
            if msg is not None:
                yield msg


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


def build_update_event(
    prefix: LiveGamePrefix,
    run_inference_fn: RunInferenceFn,
    top_k: int,
    min_ply: int,
    white_baseline: float | None,
    black_baseline: float | None,
    game_id: str,
) -> str | None:
    """Run inference over the current prefix and format it as an "update" SSE event."""
    if prefix.ply_count == 0:
        return None
    start = time.monotonic()
    result = run_inference_fn(prefix.to_pgn_text(), white_baseline, black_baseline, top_k, min_ply)
    result["inference_ms"] = round((time.monotonic() - start) * 1000.0, 1)
    result["lichess_game_id"] = game_id
    result["live"] = True
    if prefix.truncated:
        result.setdefault("warnings", []).append(
            f"Live analysis is capped at {prefix.max_plies} plies to match model training; "
            "further moves are shown on the board but not scored."
        )
    return sse_event({"type": "update", "result": result})


def headers_from_tv_players(players: list[dict[str, Any]]) -> dict[str, str]:
    """Build minimal PGN-style headers from a TV "featured" event's player list."""
    headers = {"White": "White", "Black": "Black", "Result": "*"}
    for player in players:
        color = player.get("color")
        name = (player.get("user") or {}).get("name", "?")
        rating = player.get("rating")
        if color == "white":
            headers["White"] = name
            if rating is not None:
                headers["WhiteElo"] = str(rating)
        elif color == "black":
            headers["Black"] = name
            if rating is not None:
                headers["BlackElo"] = str(rating)
    return headers


async def stream_game(
    game_id: str,
    prefix: LiveGamePrefix,
    run_inference_fn: RunInferenceFn,
    top_k: int,
    min_ply: int,
    white_baseline: float | None,
    black_baseline: float | None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Follow one Lichess game by ID, yielding SSE-formatted text chunks.

    ``prefix`` should already be seeded (via ``LiveGamePrefix.seed_from_pgn``)
    with any moves played before this connection was opened.
    """
    if (prefix.headers.get("Result") or "*").strip() != "*":
        yield sse_event(
            {
                "type": "status",
                "state": "finished",
                "message": "This game has already finished; showing its full-game analysis.",
            }
        )
        event = build_update_event(prefix, run_inference_fn, top_k, min_ply, white_baseline, black_baseline, game_id)
        if event:
            yield event
        return

    yield sse_event(
        {
            "type": "status",
            "state": "connected",
            "message": "Connected. Spectator view may lag a few moves behind live play (Lichess broadcast delay).",
        }
    )
    event = build_update_event(prefix, run_inference_fn, top_k, min_ply, white_baseline, black_baseline, game_id)
    if event:
        yield event

    attempts = 0
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        while attempts < MAX_STREAM_ATTEMPTS:
            attempts += 1
            seen_ply_events = 0  # /api/stream/game/{id} always replays from ply 0 on each new connection
            try:
                async for msg in iter_ndjson_lines(client, STREAM_GAME_URL_TEMPLATE.format(game_id=game_id)):
                    kind = classify_game_stream_message(msg)
                    if kind == "position":
                        if "lm" not in msg:
                            continue
                        seen_ply_events += 1
                        if seen_ply_events <= prefix.ply_count:
                            continue  # backlog already covered by the seed or an earlier attempt
                        clock = clock_seconds_for_ply(seen_ply_events, msg.get("wc", 0), msg.get("bc", 0))
                        try:
                            added = prefix.add_move(msg["lm"], clock)
                        except Exception as exc:
                            yield sse_event(
                                {
                                    "type": "error",
                                    "detail": (
                                        f"This game could not be scored (unsupported move or variant): {exc}"
                                    ),
                                }
                            )
                            return
                        if added:
                            event = build_update_event(
                                prefix, run_inference_fn, top_k, min_ply, white_baseline, black_baseline, game_id
                            )
                            if event:
                                yield event
                        else:
                            yield sse_event(
                                {
                                    "type": "status",
                                    "state": "capped",
                                    "message": (
                                        f"Live analysis is capped at {prefix.max_plies} plies; further "
                                        "moves are not scored."
                                    ),
                                }
                            )
                    elif kind == "final":
                        status_name = (msg.get("status") or {}).get("name", "finished")
                        yield sse_event(
                            {"type": "status", "state": "finished", "message": f"Game over ({status_name})."}
                        )
                        return
                    # "meta" lines only repeat game info already covered by the export seed; ignore.
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": "Stream ended unexpectedly; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": "Lost connection to Lichess and could not reconnect."})
                return
            except LiveStreamError as exc:
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": f"{exc.detail}; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": exc.detail})
                return
            except httpx.HTTPError as exc:
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": "Connection error; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": f"Could not reach Lichess: {exc}"})
                return
    finally:
        if owns_client:
            await client.aclose()


async def stream_tv(
    fetch_export_pgn_fn: FetchExportPgnFn,
    run_inference_fn: RunInferenceFn,
    top_k: int,
    min_ply: int,
    max_plies: int,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Follow Lichess TV's featured game, switching prefix state whenever TV switches games."""
    yield sse_event({"type": "status", "state": "connecting", "message": "Connecting to Lichess TV..."})

    prefix: LiveGamePrefix | None = None
    current_game_id: str | None = None
    attempts = 0
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()
    try:
        while attempts < MAX_STREAM_ATTEMPTS:
            attempts += 1
            seen_ply_events = 0
            try:
                async for msg in iter_ndjson_lines(client, TV_FEED_URL):
                    msg_type = msg.get("t")
                    data = msg.get("d") or {}
                    if msg_type == "featured":
                        new_game_id = data.get("id")
                        if not new_game_id:
                            continue
                        current_game_id = new_game_id
                        prefix = LiveGamePrefix(headers_from_tv_players(data.get("players", [])), max_plies=max_plies)
                        try:
                            export_pgn = await fetch_export_pgn_fn(current_game_id)
                            prefix.seed_from_pgn(export_pgn)
                        except Exception:
                            # Either the export just isn't available yet (fine: we'll catch
                            # up move by move from the live feed) or this is a non-standard
                            # game (e.g. a variant slipping into a "standard" TV channel);
                            # the add_move guard below will catch the latter case too.
                            pass
                        seen_ply_events = prefix.ply_count
                        yield sse_event(
                            {
                                "type": "status",
                                "state": "connected",
                                "message": (
                                    f"Watching Lichess TV: game {current_game_id}. Spectator view may lag a "
                                    "few moves behind live play (Lichess broadcast delay)."
                                ),
                            }
                        )
                        event = build_update_event(prefix, run_inference_fn, top_k, min_ply, None, None, current_game_id)
                        if event:
                            yield event
                    elif msg_type == "fen":
                        if prefix is None or "lm" not in data:
                            continue
                        seen_ply_events += 1
                        if seen_ply_events <= prefix.ply_count:
                            continue
                        clock = clock_seconds_for_ply(seen_ply_events, data.get("wc", 0), data.get("bc", 0))
                        try:
                            added = prefix.add_move(data["lm"], clock)
                        except Exception:
                            # The live move didn't match our tracked board (an unsupported
                            # variant, or the feed skipped ahead); drop this game and wait
                            # for the next TV switch rather than crashing the stream.
                            prefix = None
                            yield sse_event(
                                {
                                    "type": "status",
                                    "state": "connecting",
                                    "message": f"Lost sync with game {current_game_id}; waiting for TV to switch...",
                                }
                            )
                            continue
                        if added:
                            event = build_update_event(
                                prefix, run_inference_fn, top_k, min_ply, None, None, current_game_id
                            )
                            if event:
                                yield event
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": "TV stream ended unexpectedly; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": "Lost connection to Lichess TV and could not reconnect."})
                return
            except LiveStreamError as exc:
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": f"{exc.detail}; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": exc.detail})
                return
            except httpx.HTTPError as exc:
                if attempts < MAX_STREAM_ATTEMPTS:
                    yield sse_event(
                        {"type": "status", "state": "reconnecting", "message": "Connection error; reconnecting..."}
                    )
                    continue
                yield sse_event({"type": "error", "detail": f"Could not reach Lichess: {exc}"})
                return
    finally:
        if owns_client:
            await client.aclose()

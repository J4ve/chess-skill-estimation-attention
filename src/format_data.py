"""
Minimal data-formatting utilities adapted from RatingNet.

Only the pieces needed for inference are included: board-to-tensor encoding,
PGN parsing helpers, and clock normalization. Full preprocessing (bulk PGN to
pickle) lives in the upstream fork at J4ve/RatingNet.
"""

import re
import chess
import chess.pgn
import torch

rejected_games = 0


def board_to_array(board: chess.Board) -> torch.Tensor:
    """Convert a chess board into a 12-layer 8x8 float tensor.

    Planes 0-5 hold white pieces (pawn, knight, bishop, rook, queen, king);
    planes 6-11 hold the same for black. This matches the baseline RatingNet
    input encoding exactly.
    """
    board_array = torch.zeros((12, 8, 8), dtype=torch.float32)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        index = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        row, col = divmod(square, 8)
        board_array[index, 7 - row, col] = 1
    return board_array


def time_to_seconds(time_str: str) -> int:
    """Convert a 'HH:MM:SS' clock string to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def parse_game(
    game: chess.pgn.Game,
    max_plies: int = 100,
) -> dict | None:
    """Parse a single PGN game into positions, clocks, and headers.

    Mirrors the upstream ``format_data.parse_game`` logic but keeps only the
    fields required for inference. Games without clock annotations are skipped
    because the model was trained with clock features. Games with only partial
    clock annotation are also rejected so clocks never silently misalign
    against positions.
    """
    global rejected_games
    time_control = game.headers.get("TimeControl", "")
    board = game.board()
    moves = []
    san_moves = []
    clocks = []
    positions = []
    node = game
    ply_count = 0
    while node.variations and ply_count < max_plies:
        next_node = node.variation(0)
        move = next_node.move
        san_moves.append(board.san(move))
        board.push(move)
        positions.append(board_to_array(board))
        moves.append(move.uci())

        comment = next_node.comment
        clock_match = re.search(r"\[%clk\s+([^\]]+)\]", comment)
        if clock_match:
            clocks.append(clock_match.group(1))

        node = next_node
        ply_count += 1

    if not clocks:
        rejected_games += 1
        return None

    if len(clocks) != len(positions):
        rejected_games += 1
        return None

    white_elo = game.headers.get("WhiteElo")
    black_elo = game.headers.get("BlackElo")
    result = game.headers.get("Result")
    return {
        "WhiteElo": white_elo,
        "BlackElo": black_elo,
        "White": game.headers.get("White"),
        "Black": game.headers.get("Black"),
        "Result": result,
        "Clocks": clocks,
        "Positions": positions,
        "Moves": moves,
        "SAN": san_moves,
        "Time": time_control,
    }


def categorize_time_control(estimated_duration: int) -> str:
    """Map an estimated game duration in seconds to a Lichess time-control bucket."""
    if estimated_duration < 29:
        return "ultrabullet"
    elif estimated_duration < 179:
        return "bullet"
    elif estimated_duration < 479:
        return "blitz"
    elif estimated_duration < 1499:
        return "rapid"
    else:
        return "classical"


NON_STANDARD_GAME_MESSAGE = (
    "This game starts from a custom position or variant (e.g. a thematic arena). "
    "The model was trained only on standard games from the normal starting "
    "position, so it cannot analyze it."
)


def game_setup_error(headers) -> str | None:
    """Return why this PGN can't be analyzed due to its setup, or None if it's a
    standard game from the normal starting position.

    Catches a non-``Standard`` ``Variant`` header (Chess960, Crazyhouse, ...) and
    an explicit start position (``SetUp "1"`` with a ``FEN`` header, as Lichess
    thematic "From Position" arenas use), both up front and before any clock
    check, since the model was trained only on standard games from the normal
    starting position.
    """
    variant = (headers.get("Variant") or "Standard").strip()
    setup = (headers.get("SetUp") or "").strip()
    fen = (headers.get("FEN") or "").strip()
    if variant.lower() != "standard" or setup == "1" or fen:
        return NON_STANDARD_GAME_MESSAGE
    return None


def parse_time_control(time_control_header: str | None) -> tuple[int | None, int | None]:
    """Parse a PGN ``TimeControl`` header (``"{base_seconds}+{increment_seconds}"``,
    e.g. "180+0") into ``(base, increment)``. Returns ``(None, None)`` for a
    missing header, correspondence ("-"), or any value that isn't exactly two
    digit parts.
    """
    if not time_control_header:
        return None, None
    parts = time_control_header.split("+")
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None, None
    return int(parts[0]), int(parts[1])


def time_control_bucket(time_control_header: str | None) -> str | None:
    """Derive a Lichess time-control bucket from a PGN ``TimeControl`` header.

    Returns None when ``parse_time_control`` can't parse the header, so callers
    can fall back to an overall/ungrouped cutoff instead of guessing.
    """
    base, inc = parse_time_control(time_control_header)
    if base is None:
        return None
    return categorize_time_control(base + 40 * inc)


def compute_time_spent(
    clock_seconds: list[int], base_seconds: int | None, increment_seconds: int | None
) -> list[int | None]:
    """For each ply's remaining clock (sides alternate: ply 0 is White's first
    move, ply 1 is Black's, ply 2 is White's second, ...), compute the time
    spent on that move: that side's previous remaining clock minus this ply's
    clock, plus the increment.

    A side's first move (ply 0 or 1) has no earlier in-game clock to compare
    against, so it falls back to the ``TimeControl`` base allotment; when that
    header couldn't be parsed (``base_seconds`` is None), the result for that
    ply is None rather than a guess.
    """
    inc = increment_seconds or 0
    spent: list[int | None] = []
    for i, clock in enumerate(clock_seconds):
        previous = clock_seconds[i - 2] if i >= 2 else base_seconds
        spent.append(None if previous is None else previous - clock + inc)
    return spent

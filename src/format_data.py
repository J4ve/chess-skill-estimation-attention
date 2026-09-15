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

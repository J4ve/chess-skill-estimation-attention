"""
Stream-preprocess a Lichess lichess_db_standard_rated_YYYY-MM.pgn.zst dump into
per-game pickle files consumable by chess_rating_net.ChessGamesDataset.

Never materializes the full decompressed PGN on disk: reads the .zst archive
through a streaming zstandard decompressor directly into python-chess's PGN
parser, and writes one small .pkl per surviving game.

Sampling: the FULL archive is streamed and surviving games are kept via
reservoir sampling (Algorithm R, reservoir size --max-games), so every
eligible game in the month has an equal probability of inclusion. This
replaces an older version that stopped at the first --max-games surviving
games, which — because Lichess monthly dumps are ordered chronologically —
silently kept only the first thin time slice of each month and starved the
slow time-control buckets. Per-time-control counts of the realized sample are
printed at the end and written to <output-dir>/.sampling_summary.json.

Usage:
    python preprocess_lichess.py \
        --input /path/to/lichess_db_standard_rated_2026-07.pgn.zst \
        --output-dir /path/to/processed_games \
        --max-games 30000

Mirrors format_data.parse_game (12-plane board tensor encoding, 100-ply cap,
clock-annotation requirement) from prototype/src/format_data.py so pickles are
schema-compatible with chess_rating_net.py's ChessGamesDataset.
"""

import argparse
import io
import json
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

import chess.pgn
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_data import categorize_time_control, parse_game  # noqa: E402

TC_BUCKETS = ["ultrabullet", "bullet", "blitz", "rapid", "classical"]

# Must stay byte-identical to the [%clk ...] pattern inside format_data.parse_game:
# the scan-phase fast path below is only valid while both use the same match.
CLOCK_COMMENT_RE = re.compile(r"\[%clk\s+([^\]]+)\]")


def stream_games(fh):
    """Yield chess.pgn.Game objects streamed from an open binary file handle
    (local .zst file or a curl subprocess's stdout pipe) without ever writing
    the decompressed PGN to disk."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with dctx.stream_reader(fh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                return
            yield game


def is_eligible(game: chess.pgn.Game) -> bool:
    headers = game.headers
    if headers.get("Variant", "Standard") != "Standard":
        return False
    if "Rated" not in headers.get("Event", ""):
        return False
    if not headers.get("WhiteElo", "").isdigit():
        return False
    if not headers.get("BlackElo", "").isdigit():
        return False
    tc = headers.get("TimeControl", "")
    parts = tc.split("+")
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return False  # skip correspondence ("-") and malformed controls
    return True


def has_full_clocks(game: chess.pgn.Game, max_plies: int) -> bool:
    """Scan-phase eligibility fast path.

    Contract: returns True exactly when parse_game(game, max_plies) would return
    non-None, i.e. the game has at least one mainline ply and every mainline ply
    up to max_plies carries a [%clk] comment. Walks the same mainline as
    parse_game but skips board.push() and the 12-plane tensor build, which the
    eligibility decision never reads.
    """
    node = game
    ply_count = 0
    while node.variations and ply_count < max_plies:
        node = node.variation(0)
        if not CLOCK_COMMENT_RE.search(node.comment):
            return False  # missing or partial clock annotation
        ply_count += 1
    return ply_count > 0


def month_label(source: str) -> str:
    """Extract YYYY-MM from a Lichess dump filename, else 'unknown'."""
    m = re.search(r"(\d{4}-\d{2})", source)
    return m.group(1) if m else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Path to a local .pgn.zst archive")
    src.add_argument("--url", type=str, help="URL to a .pgn.zst archive; streamed via curl, "
                                              "never saved to disk (the full archive is streamed)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output .pkl files")
    parser.add_argument("--max-games", type=int, default=30000,
                        help="Reservoir size: number of surviving games kept from the full archive")
    parser.add_argument("--max-plies", type=int, default=100, help="Ply cap per game (matches baseline)")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reservoir sampling")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curl_proc = None
    if args.url:
        import subprocess

        curl_proc = subprocess.Popen(["curl", "-s", args.url], stdout=subprocess.PIPE)
        source_fh = curl_proc.stdout
    else:
        source_fh = open(args.input, "rb")

    rng = random.Random(args.seed)
    reservoir = []  # chess.pgn.Game objects that passed the full eligibility filter
    seen = 0
    eligible = 0
    try:
        for game in stream_games(source_fh):
            seen += 1
            if seen % args.log_every == 0:
                print(f"scanned={seen} eligible={eligible} reservoir={len(reservoir)}", flush=True)
            if not is_eligible(game):
                continue
            if not has_full_clocks(game, max_plies=args.max_plies):
                continue  # missing or partial clock annotations
            eligible += 1
            # Algorithm R reservoir sampling: uniform over all eligible games
            # once the full archive has been streamed.
            if len(reservoir) < args.max_games:
                reservoir.append(game)
            else:
                j = rng.randint(0, eligible - 1)
                if j < args.max_games:
                    reservoir[j] = game
    finally:
        if curl_proc is not None:
            curl_proc.stdout.close()
            curl_proc.terminate()
            curl_proc.wait()
        else:
            source_fh.close()

    tc_counts = Counter()
    kept = 0
    for game in reservoir:
        parsed = parse_game(game, max_plies=args.max_plies)
        if parsed is None:
            # Unreachable while has_full_clocks matches parse_game exactly. A
            # nonzero count here means the fast path diverged: the month's kept
            # count will fall short and corpus_stream will retry/FATAL, so make
            # the divergence loud in the log instead of silent.
            print("WARNING: reservoir game dropped at write time "
                  "(fast-path divergence?)", flush=True)
            continue
        base, inc = (int(x) for x in game.headers["TimeControl"].split("+"))
        tc_counts[categorize_time_control(base + 40 * inc)] += 1
        out_path = args.output_dir / f"game_{kept:07d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(parsed, f)
        kept += 1

    month = month_label(str(args.input) if args.input else args.url)
    summary = {
        "month": month,
        "source": str(args.input) if args.input else args.url,
        "sampling": "reservoir (Algorithm R) over all eligible games in the full archive",
        "seed": args.seed,
        "reservoir_size": args.max_games,
        "scanned": seen,
        "eligible": eligible,
        "kept": kept,
        "time_control_counts": {tc: tc_counts.get(tc, 0) for tc in TC_BUCKETS},
    }
    summary_path = args.output_dir / ".sampling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    tc_str = " ".join(f"{tc}={tc_counts.get(tc, 0)}" for tc in TC_BUCKETS)
    print(f"SUMMARY month={month} scanned={seen} eligible={eligible} kept={kept} {tc_str}", flush=True)
    print(f"DONE scanned={seen} kept={kept} output_dir={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

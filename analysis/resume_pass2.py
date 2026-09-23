#!/usr/bin/env python3
"""Resume-only pass 2 for preprocess_onepass.py, when pass 1's clkscan run
already completed and its .selected.pgn.tmp survived a crash in pass 2 (the
pickle-writing step) - lets a crash there be recovered without re-running the
whole-month scan (the expensive part).

Usage:
    python resume_pass2.py --pgn <output_dir>/.selected.pgn.tmp \\
        --output-dir <output_dir> --month YYYY-MM --input <original .zst path> \\
        --max-games 30000 --seed 42
"""

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype" / "src"))
from format_data import categorize_time_control, parse_game  # noqa: E402
from preprocess_lichess import TC_BUCKETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", required=True, type=Path, help="surviving .selected.pgn.tmp")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--month", required=True, type=str)
    ap.add_argument("--input", required=True, type=str, help="original .zst path, for the summary record only")
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t1 = time.perf_counter()
    tc_counts: Counter = Counter()
    kept = 0
    parsed_count = 0
    with open(args.pgn, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            parsed_count += 1
            parsed = parse_game(game, max_plies=args.max_plies)
            if parsed is None:
                print(f"WARNING: extracted game failed parse_game (slot {parsed_count - 1})", flush=True)
                continue
            base, inc = (int(x) for x in parsed["Time"].split("+"))
            tc_counts[categorize_time_control(base + 40 * inc)] += 1
            with open(args.output_dir / f"game_{kept:07d}.pkl", "wb") as f:
                pickle.dump(parsed, f)
            kept += 1
    t_parse = time.perf_counter() - t1

    if parsed_count != args.max_games:
        sys.exit(f"extracted PGN held {parsed_count} games, expected {args.max_games} - "
                 "the surviving PGN looks incomplete or truncated, refusing to write a partial sample")

    summary = {
        "month": args.month,
        "source": args.input,
        "sampling": "reservoir (Algorithm R) over all eligible games in the full archive",
        "seed": args.seed,
        "reservoir_size": args.max_games,
        "kept": kept,
        "time_control_counts": {tc: tc_counts.get(tc, 0) for tc in TC_BUCKETS},
        "note": "resumed pass 2 after a mid-write I/O crash; pass 1 (scan/select) was not re-run",
    }
    with open(args.output_dir / ".sampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    tc_str = " ".join(f"{tc}={tc_counts.get(tc, 0)}" for tc in TC_BUCKETS)
    print(f"pass2 elapsed_s={t_parse:.1f}", flush=True)
    print(f"SUMMARY month={args.month} kept={kept} {tc_str}", flush=True)
    print(f"DONE kept={kept} output_dir={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

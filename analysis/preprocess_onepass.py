#!/usr/bin/env python3
"""One-pass drop-in replacement for preprocess_lichess.py's scan+sample phase.

Successor to `preprocess_fast.py`. Both produce the same output; the difference
is how much of the month python-chess has to touch.

    preprocess_lichess.py  python-chess fully parses all ~92M games      ~76 min+
    preprocess_fast.py     clkscan pass 1, then python-chess skip_game
                           across all ~92M games to reach the kept 30k   ~76 min
    preprocess_onepass.py  clkscan scans AND samples AND extracts in one
                           pass; python-chess only ever sees the 30k     ~21 min

The saving is the entire pass-2 `skip_game` walk. `preprocess_fast.py` still had
to drag python-chess's parser across every rejected game just to advance the
stream to the next one; here `clkscan` already holds the selected games' text by
the time the scan finishes, so nothing re-reads the archive.

What this script is responsible for is deliberately small: run the binary, then
run the *unmodified production* `parse_game` over the ~30,000 games it emitted
and pickle the results. Sampling is not reimplemented here (clkscan does it,
using a byte-exact port of CPython's Mersenne Twister), and parsing is not
reimplemented here (format_data.parse_game does it, unchanged). See
`analysis/rust-pass2-preprocessor.md` for the equivalence evidence.

Usage:
    python preprocess_onepass.py --input month.pgn.zst --output-dir out/ \\
        --max-games 30000 --clkscan ./bin/clkscan
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype" / "src"))
from format_data import categorize_time_control, parse_game  # noqa: E402
from preprocess_lichess import TC_BUCKETS, month_label  # noqa: E402


def run_clkscan(
    clkscan: Path,
    src: Path,
    out_pgn: Path,
    stats_path: Path,
    max_plies: int,
    max_games: int,
    seed: int,
    verdicts: Path | None,
    log_every: int,
) -> dict:
    """Pass 1: scan, sample inline, emit the selected games' PGN.

    stdout/stderr are inherited rather than captured so the progress lines reach
    the pipeline's per-month log live. A 21-minute run that prints nothing until
    it finishes is indistinguishable from a hung one, which is exactly the
    complaint this rewrite was asked to fix.
    """
    cmd = [
        str(clkscan),
        "--input", str(src),
        "--emit-pgn", str(out_pgn),
        "--stats-json", str(stats_path),
        "--max-plies", str(max_plies),
        "--max-games", str(max_games),
        "--seed", str(seed),
        "--log-every", str(log_every),
    ]
    if verdicts is not None:
        cmd += ["--verdicts", str(verdicts)]

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        # clkscan already printed a specific diagnostic (bad path, truncated
        # archive, ...). Do not paper over it with a generic message.
        sys.exit(f"clkscan failed (rc={proc.returncode}); see the message above")

    stats = json.loads(stats_path.read_text())
    if stats.get("truncated"):
        # Belt and braces: clkscan refuses this itself in select mode. If that
        # check is ever relaxed, this one still stops a partial month from being
        # written out as if it were whole.
        sys.exit(
            "clkscan reported a truncated archive. Refusing to proceed: a partial "
            "month would silently produce a partial, non-uniform sample."
        )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--clkscan", required=True, type=Path, help="path to the clkscan binary")
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=200000)
    ap.add_argument("--keep-verdicts", type=Path, help="also write pass 1's bitstream here")
    ap.add_argument("--keep-pgn", type=Path,
                    help="keep the extracted PGN here instead of in a temp file")
    args = ap.parse_args()

    if not args.clkscan.exists():
        sys.exit(f"clkscan binary not found at {args.clkscan}; build it with "
                 "`cargo build --release` in analysis/scanner-rs/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_pgn = args.keep_pgn or (args.output_dir / ".selected.pgn.tmp")
    stats_path = args.output_dir / ".clkscan_stats.tmp"

    t0 = time.perf_counter()
    stats = run_clkscan(
        args.clkscan, args.input, out_pgn, stats_path,
        args.max_plies, args.max_games, args.seed, args.keep_verdicts, args.log_every,
    )
    t_scan = time.perf_counter() - t0
    scanned = stats["scanned"]
    eligible_count = stats["eligible"]
    selected = stats["selected"]
    print(f"pass1 scanned={scanned} eligible={eligible_count} selected={selected} "
          f"elapsed_s={t_scan:.1f} games_per_sec={scanned / max(t_scan, 1e-9):,.0f}", flush=True)

    # Pass 2: the only python-chess work left is the ~30,000 games we kept.
    # They arrive in reservoir-slot order, which is the order
    # preprocess_lichess.py writes them in, so the filenames line up.
    t1 = time.perf_counter()
    tc_counts: Counter = Counter()
    kept = 0
    parsed_count = 0
    with open(out_pgn, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            parsed_count += 1
            parsed = parse_game(game, max_plies=args.max_plies)
            if parsed is None:
                # Unreachable while clkscan's predicates match parse_game. Loud,
                # not silent: this means the two have diverged.
                print("WARNING: extracted game failed parse_game "
                      f"(slot {parsed_count - 1}); scanner divergence", flush=True)
                continue
            base, inc = (int(x) for x in parsed["Time"].split("+"))
            tc_counts[categorize_time_control(base + 40 * inc)] += 1
            with open(args.output_dir / f"game_{kept:07d}.pkl", "wb") as f:
                pickle.dump(parsed, f)
            kept += 1
    t_parse = time.perf_counter() - t1

    if parsed_count != selected:
        # The emitted PGN must contain exactly the games clkscan says it
        # selected. A mismatch means the extraction is dropping or duplicating
        # games, which would silently change the sample.
        sys.exit(f"extracted PGN held {parsed_count} games but clkscan selected "
                 f"{selected}; refusing to write a sample that does not match pass 1")

    summary = {
        "month": month_label(str(args.input)),
        "source": str(args.input),
        "sampling": "reservoir (Algorithm R) over all eligible games in the full archive",
        "seed": args.seed,
        "reservoir_size": args.max_games,
        "scanned": scanned,
        "eligible": eligible_count,
        "kept": kept,
        "time_control_counts": {tc: tc_counts.get(tc, 0) for tc in TC_BUCKETS},
    }
    with open(args.output_dir / ".sampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    stats_path.unlink(missing_ok=True)
    if not args.keep_pgn:
        out_pgn.unlink(missing_ok=True)

    tc_str = " ".join(f"{tc}={tc_counts.get(tc, 0)}" for tc in TC_BUCKETS)
    print(f"pass2 elapsed_s={t_parse:.1f}", flush=True)
    print(f"SUMMARY month={summary['month']} scanned={scanned} eligible={eligible_count} "
          f"kept={kept} {tc_str}", flush=True)
    print(f"DONE scanned={scanned} kept={kept} output_dir={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Two-pass drop-in replacement for preprocess_lichess.py's scan+sample phase.

GOAL: select exactly the same games as `preprocess_lichess.py`, write them under
the same filenames in the same order, with the same .sampling_summary.json, in a
fraction of the time. Not "approximately the same sample". The same one.
Anything less is not worth the risk on a corpus whose raw archives are deleted
right after processing.

A note on what "the same" can mean here: the .pkl files are NOT byte-comparable,
and that is a property of the existing pipeline, not of this script. torch
serializes each tensor's storage under a key derived from its memory address, so
`preprocess_lichess.py` does not reproduce its own bytes across two runs on
identical input either (verified: 300/300 files differ byte-wise, 0/300 differ in
content). The meaningful equivalence criterion is therefore content: same games,
same order, same Moves/Clocks/headers, and tensors equal under torch.equal. That
is what was checked, and it held for 2000/2000 games.

Why this is much faster
-----------------------
The existing scanner spends nearly all of its time in one place: fully parsing
every one of a month's ~92M games with python-chess just to decide whether each
one has complete clock annotations. Only 30,000 of them are ever kept.

    pass 1   clkscan (Rust)  decides eligibility for every game   ~150,000 g/s
    pass 2   python-chess    FULLY parses only the ~30,000 kept       ~979 g/s
             python-chess    skips everything else via skip_game   ~28,000 g/s

Measured on this project's own hardware: see analysis/faster-scan-research.md.

Why the sample is provably the same
-----------------------------------
The reservoir sampler in preprocess_lichess.py draws from `rng` ONLY for an
eligible game once the reservoir is already full, and the draw is
`rng.randint(0, eligible - 1)` where `eligible` is the running count. So the
entire random sequence is a function of the ELIGIBILITY BITSTREAM alone, not of
any game's contents. Pass 1 produces exactly that bitstream, so pass 2 can
replay Algorithm R step for step and land on the same reservoir, in the same
slot order. That is what `replay_reservoir` below does, deliberately mirroring
preprocess_lichess.py:147-154 line for line.

This buys speed WITHOUT reimplementing sampling or parsing: the kept games are
still parsed by the same `parse_game`, and the games are still selected by the
same `random.Random` call sequence.

STATUS: verified byte-identical against preprocess_lichess.py on a real Lichess
sample (see analysis/faster-scan-research.md, "End-to-end equivalence"). It has
NOT yet been run on a full 30GB month, and it has not been run on the HPC.
Treat it as validated-in-principle, not yet production-blessed.

Usage:
    python preprocess_fast.py --input month.pgn.zst --output-dir out/ \\
        --max-games 30000 --clkscan ./bin/clkscan
"""

import argparse
import io
import json
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import chess.pgn
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype" / "src"))
from format_data import categorize_time_control, parse_game  # noqa: E402
from preprocess_lichess import TC_BUCKETS, month_label  # noqa: E402


def run_clkscan(clkscan: Path, src: Path, max_plies: int, verdict_path: Path) -> tuple[int, int]:
    """Pass 1. Returns (scanned, eligible)."""
    proc = subprocess.run(
        [str(clkscan), "--input", str(src), "--max-plies", str(max_plies),
         "--verdicts", str(verdict_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"clkscan failed (rc={proc.returncode}): {proc.stderr.strip()}")
    stats = dict(
        kv.split("=", 1) for kv in proc.stdout.split() if "=" in kv
    )
    if stats.get("truncated") == "true":
        sys.exit(
            "clkscan reported a truncated archive. Refusing to proceed: a partial "
            "month would silently produce a partial, non-uniform sample."
        )
    return int(stats["scanned"]), int(stats["eligible"])


def replay_reservoir(verdicts: bytes, max_games: int, seed: int) -> list[int]:
    """Replay Algorithm R over the eligibility bitstream.

    Mirrors preprocess_lichess.py:147-154 exactly, storing each game's ordinal
    position in the file instead of its parsed Game object. The returned list is
    in reservoir-slot order, which is the order the originals are written out in,
    so filenames line up too.
    """
    rng = random.Random(seed)
    reservoir: list[int] = []
    eligible = 0
    for ordinal, byte in enumerate(verdicts):
        if byte != 0x31:  # b'1'
            continue
        eligible += 1
        if len(reservoir) < max_games:
            reservoir.append(ordinal)
        else:
            j = rng.randint(0, eligible - 1)
            if j < max_games:
                reservoir[j] = ordinal
    return reservoir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--clkscan", required=True, type=Path, help="path to the clkscan binary")
    ap.add_argument("--max-games", type=int, default=30000)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-verdicts", type=Path, help="keep pass 1's bitstream here")
    args = ap.parse_args()

    if not args.clkscan.exists():
        sys.exit(f"clkscan binary not found at {args.clkscan}; build it with "
                 "`cargo build --release` in analysis/scanner-rs/")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    verdict_path = args.keep_verdicts or (args.output_dir / ".verdicts.tmp")

    t0 = time.perf_counter()
    scanned, eligible_count = run_clkscan(args.clkscan, args.input, args.max_plies, verdict_path)
    t_scan = time.perf_counter() - t0
    print(f"pass1 scanned={scanned} eligible={eligible_count} "
          f"elapsed_s={t_scan:.1f} games_per_sec={scanned / max(t_scan, 1e-9):,.0f}", flush=True)

    verdicts = verdict_path.read_bytes().replace(b"\n", b"")
    if len(verdicts) != scanned:
        sys.exit(f"verdict stream length {len(verdicts)} != scanned {scanned}")

    reservoir = replay_reservoir(verdicts, args.max_games, args.seed)
    wanted = {ordinal: slot for slot, ordinal in enumerate(reservoir)}
    print(f"reservoir selected {len(reservoir)} games from {eligible_count} eligible", flush=True)

    # Pass 2: fully parse only the selected games; skip the rest.
    t1 = time.perf_counter()
    parsed_by_slot: dict[int, dict] = {}
    with open(args.input, "rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            ordinal = 0
            remaining = len(wanted)
            while remaining > 0:
                slot = wanted.get(ordinal)
                if slot is None:
                    if chess.pgn.skip_game(text) is False:
                        break
                else:
                    game = chess.pgn.read_game(text)
                    if game is None:
                        break
                    parsed = parse_game(game, max_plies=args.max_plies)
                    if parsed is None:
                        # Unreachable while clkscan matches has_full_clocks. Loud,
                        # not silent: this means the two predicates have diverged.
                        print("WARNING: selected game failed parse_game "
                              f"(ordinal {ordinal}); scanner divergence", flush=True)
                    else:
                        parsed_by_slot[slot] = parsed
                    remaining -= 1
                ordinal += 1
    t_parse = time.perf_counter() - t1

    tc_counts: Counter = Counter()
    kept = 0
    for slot in range(len(reservoir)):
        parsed = parsed_by_slot.get(slot)
        if parsed is None:
            continue
        base, inc = (int(x) for x in parsed["Time"].split("+"))
        tc_counts[categorize_time_control(base + 40 * inc)] += 1
        with open(args.output_dir / f"game_{kept:07d}.pkl", "wb") as f:
            pickle.dump(parsed, f)
        kept += 1

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

    if not args.keep_verdicts:
        verdict_path.unlink(missing_ok=True)

    tc_str = " ".join(f"{tc}={tc_counts.get(tc, 0)}" for tc in TC_BUCKETS)
    print(f"pass2 elapsed_s={t_parse:.1f}", flush=True)
    print(f"SUMMARY month={summary['month']} scanned={scanned} eligible={eligible_count} "
          f"kept={kept} {tc_str}", flush=True)
    print(f"DONE scanned={scanned} kept={kept} output_dir={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

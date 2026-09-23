#!/usr/bin/env python3
"""Content-equivalence oracle for two processed-month directories.

This is the evidence the one-pass Rust preprocessor stands on. It answers one
question: did `preprocess_onepass.py` select and materialise *exactly* the games
`preprocess_lichess.py` would have, in exactly the same slot order, with exactly
the same content?

Why content and not bytes
-------------------------
The `.pkl` files are NOT byte-comparable, and that is a property of the existing
pipeline rather than of anything new here: torch serialises each tensor's storage
under a key derived from its memory address, so `preprocess_lichess.py` does not
reproduce its own bytes across two runs on identical input either. The meaningful
criterion is therefore content: same games, same order, same
Moves/Clocks/headers, and `torch.equal` on every one of the 12-plane position
tensors. Anything weaker (counts, or a sampled spot-check) would not distinguish
"the same 30,000 games" from "30,000 equally plausible games", which is precisely
the failure this check exists to rule out.

Exits nonzero on any disagreement, so it can gate CI or a pipeline run.

Usage:
    python compare_processed_dirs.py --a ref_out/ --b new_out/ [--max-report 20]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

SCALAR_FIELDS = ("WhiteElo", "BlackElo", "White", "Black", "Result", "Time")


def load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def game_files(d: Path) -> list[Path]:
    return sorted(d.glob("game_*.pkl"))


def compare_one(a: dict, b: dict) -> list[str]:
    """Return a list of human-readable differences, empty if identical."""
    diffs: list[str] = []

    if set(a.keys()) != set(b.keys()):
        diffs.append(f"key sets differ: {sorted(a.keys())} vs {sorted(b.keys())}")
        return diffs

    for field in SCALAR_FIELDS:
        if a.get(field) != b.get(field):
            diffs.append(f"{field}: {a.get(field)!r} vs {b.get(field)!r}")

    if a["Moves"] != b["Moves"]:
        n = min(len(a["Moves"]), len(b["Moves"]))
        first = next((i for i in range(n) if a["Moves"][i] != b["Moves"][i]), n)
        diffs.append(
            f"Moves differ (len {len(a['Moves'])} vs {len(b['Moves'])}), "
            f"first at ply {first}: {a['Moves'][first:first + 3]} vs "
            f"{b['Moves'][first:first + 3]}"
        )

    if a["Clocks"] != b["Clocks"]:
        n = min(len(a["Clocks"]), len(b["Clocks"]))
        first = next((i for i in range(n) if a["Clocks"][i] != b["Clocks"][i]), n)
        diffs.append(
            f"Clocks differ (len {len(a['Clocks'])} vs {len(b['Clocks'])}), "
            f"first at ply {first}: {a['Clocks'][first:first + 3]} vs "
            f"{b['Clocks'][first:first + 3]}"
        )

    pa, pb = a["Positions"], b["Positions"]
    if len(pa) != len(pb):
        diffs.append(f"Positions length differs: {len(pa)} vs {len(pb)}")
    else:
        for i, (ta, tb) in enumerate(zip(pa, pb)):
            if not torch.equal(ta, tb):
                diffs.append(f"Positions[{i}] tensors differ")
                break

    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path, help="reference output dir")
    ap.add_argument("--b", required=True, type=Path, help="candidate output dir")
    ap.add_argument("--max-report", type=int, default=20)
    args = ap.parse_args()

    fa, fb = game_files(args.a), game_files(args.b)
    print(f"A: {len(fa)} games in {args.a}")
    print(f"B: {len(fb)} games in {args.b}")

    problems = 0

    if len(fa) != len(fb):
        print(f"MISMATCH: game counts differ ({len(fa)} vs {len(fb)})")
        problems += 1

    names_a = [p.name for p in fa]
    names_b = [p.name for p in fb]
    if names_a != names_b:
        print("MISMATCH: filename sequences differ")
        problems += 1

    compared = 0
    differing = 0
    for pa, pb in zip(fa, fb):
        ga, gb = load(pa), load(pb)
        diffs = compare_one(ga, gb)
        compared += 1
        if diffs:
            differing += 1
            if differing <= args.max_report:
                print(f"MISMATCH in slot {pa.name}:")
                for d in diffs:
                    print(f"    {d}")

    # The sampling summaries must agree too: they carry the scanned/eligible
    # counts the pipeline's own sanity checks read.
    sa, sb = args.a / ".sampling_summary.json", args.b / ".sampling_summary.json"
    if sa.exists() and sb.exists():
        ja, jb = json.loads(sa.read_text()), json.loads(sb.read_text())
        for field in ("scanned", "eligible", "kept", "seed", "reservoir_size",
                      "time_control_counts"):
            if ja.get(field) != jb.get(field):
                print(f"MISMATCH: .sampling_summary.json {field}: "
                      f"{ja.get(field)!r} vs {jb.get(field)!r}")
                problems += 1
    else:
        print("note: one or both .sampling_summary.json files are absent; not compared")

    print()
    print(f"games compared:   {compared}")
    print(f"games differing:  {differing}")
    print(f"other problems:   {problems}")

    if differing == 0 and problems == 0 and compared > 0:
        print(f"RESULT: IDENTICAL - {compared}/{compared} games match on "
              "headers, Moves, Clocks and all position tensors")
        return 0
    print("RESULT: DIVERGENT")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

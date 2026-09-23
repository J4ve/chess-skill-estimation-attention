"""Deterministically sample N game ids from the frozen test partition.

Reads the frozen split manifest (``data/split_manifest_2p55M_seed42.json`` on
the HPC) and draws a fixed-seed sample of game basenames from ``test_files``
for the deployment-measurement scripts. Does not import or modify anything
under ``prototype/``.

Usage:
    python sample_test_games.py --manifest data/split_manifest_2p55M_seed42.json \
        --n 100 --seed 20260913 --out games_100.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)

    test_files = sorted(manifest["test_files"])
    rng = random.Random(args.seed)
    sample = rng.sample(test_files, args.n)
    sample.sort()

    out = {
        "manifest_path": str(manifest_path),
        "manifest_sha256_field": manifest["sha256"],
        "manifest_split_seed": manifest["split_seed"],
        "test_files_count": len(test_files),
        "sample_seed": args.seed,
        "sample_n": len(sample),
        "game_ids": sample,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(sample)} game ids to {args.out}")
    print(f"manifest sha256 field: {manifest['sha256'][:12]}...")


if __name__ == "__main__":
    main()

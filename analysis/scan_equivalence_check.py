#!/usr/bin/env python3
"""Baseline benchmark + equivalence oracle for the Rust scan-phase prototype.

Two jobs, one code path:

1. Measure the CURRENT Python scanner's throughput on a real ``.pgn.zst`` sample,
   on this machine, so the Rust prototype is compared against a number measured
   here rather than against a remembered HPC figure.

2. Emit a per-game eligibility verdict stream (one ``1``/``0`` per line, in file
   order) that must match ``clkscan --verdicts`` byte for byte. That diff is the
   equivalence evidence. Without it the Rust scanner is just a fast program that
   prints a plausible number.

The predicates are IMPORTED from ``prototype/src/preprocess_lichess.py``, never
re-implemented here, so this measures and validates against the real production
decision rather than a copy of it that could drift.

``format_data`` imports torch at module scope purely for the 12-plane tensor
build, which the scan phase never calls. Installing torch just to import two
regex predicates is not worth it, so a stub stands in. If the stub is ever
actually used, the scan phase has stopped being tensor-free and this script
should start failing loudly, which is why the stub raises rather than returning
a dummy value.

Usage:
    python scan_equivalence_check.py --input sample.pgn.zst \
        [--verdicts out.txt] [--limit N] [--max-plies 100]
"""

import argparse
import sys
import time
import types
from pathlib import Path


class _StubTorch(types.ModuleType):
    """Stands in for torch during import; every attribute access is a bug."""

    Tensor = object

    def __getattr__(self, name):
        raise RuntimeError(
            f"scan phase unexpectedly used torch.{name}; the eligibility check is "
            "supposed to be tensor-free, so this benchmark is no longer measuring "
            "what it claims to measure"
        )


def _install_predicates(repo_root: Path):
    """Import the real is_eligible / has_full_clocks / stream_games."""
    sys.modules.setdefault("torch", _StubTorch("torch"))
    src = repo_root / "prototype" / "src"
    if not (src / "preprocess_lichess.py").exists():
        sys.exit(f"cannot find preprocess_lichess.py under {src}")
    sys.path.insert(0, str(src))
    import preprocess_lichess as pl  # noqa: E402

    return pl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--verdicts", type=Path, help="write one 1/0 line per game")
    ap.add_argument("--limit", type=int, default=0, help="stop after N games (0 = all)")
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repo root containing prototype/src",
    )
    args = ap.parse_args()

    pl = _install_predicates(args.repo_root)

    out = open(args.verdicts, "w") if args.verdicts else None
    scanned = 0
    eligible = 0
    truncated = False

    fh = open(args.input, "rb")
    start = time.perf_counter()
    try:
        for game in pl.stream_games(fh):
            ok = pl.is_eligible(game) and pl.has_full_clocks(game, max_plies=args.max_plies)
            scanned += 1
            if ok:
                eligible += 1
            if out:
                out.write("1\n" if ok else "0\n")
            if args.limit and scanned >= args.limit:
                break
    except Exception as exc:
        # A deliberately truncated benchmark sample ends mid-frame. Expected.
        print(f"note: input ended early ({type(exc).__name__}: {exc})", file=sys.stderr)
        truncated = True
    finally:
        elapsed = time.perf_counter() - start
        fh.close()
        if out:
            out.close()

    rate = scanned / elapsed if elapsed > 0 else 0.0
    print(
        f"scanned={scanned} eligible={eligible} elapsed_s={elapsed:.3f} "
        f"games_per_sec={rate:.0f} truncated={truncated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

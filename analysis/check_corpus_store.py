#!/usr/bin/env python3
"""Independent verification of a corpus store built by ``build_corpus_store.py``.

Written to be run by someone other than its author, on the machine that will do
the training, without trusting anything in this repo.  It answers the four
questions that decide whether the store is safe to train on:

1. **Does it hold the same games?**  The sorted ``.pkl`` basename list must be
   identical to what ``os.listdir`` gives for the loose directory, because
   ``load_or_create_split`` hashes exactly that list.  If the hash moves, the
   frozen split moves with it and ``chess_rating_net.py`` aborts.
2. **Does it return the same bytes?**  Every game is unpickled from both sources
   and compared field by field, including tensor-by-tensor equality.
3. **Is it safe under DataLoader workers?**  Reads are driven from N forked
   processes after the parent has already opened a connection, which is the
   arrangement that corrupts naively-shared handles.  Also checks the store
   survives being pickled to a ``spawn``-ed worker.
4. **What does it actually cost per read?**  Times loose vs store on the same
   random access order, single-threaded, which is the comparison the ZIP
   measurement of 2026-08-24 used.

Usage::

    # full check against the loose 170k directory
    python analysis/scripts/check_corpus_store.py \
        --store /tmp/ratingnet_store --loose /tmp/ratingnet_data_flat

    # store-only checks (no loose copy to compare against)
    python analysis/scripts/check_corpus_store.py --store /tmp/ratingnet_store

Exit code is 0 only if every check that could be run passed.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import sqlite3
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototype" / "src"))


def _load_store_class():
    """Import GameBlobStore from the trainer without importing torch.

    The trainer pulls in torch at module scope, which is heavy and may not be
    present wherever this check runs, so the class is lifted out by AST.
    """
    import ast
    from typing import Any

    src = Path(__file__).resolve().parents[2] / "prototype" / "src" / "chess_rating_net.py"
    tree = ast.parse(src.read_text())
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GameBlobStore"
    )
    ns = {"os": os, "Path": Path, "sqlite3": sqlite3, "pickle": pickle,
          "zlib": zlib, "Any": Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), "exec"), ns)
    cls = ns["GameBlobStore"]
    cls.__module__ = "__main__"
    globals()["GameBlobStore"] = cls
    return cls


GameBlobStore = _load_store_class()

STORE = None       # set in main, so forked workers inherit it
FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def games_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if k == "Positions":
            if len(va) != len(vb):
                return False
            for pa, pb in zip(va, vb):
                # torch tensors need .equal(); anything else compares directly
                if hasattr(pa, "equal"):
                    if not pa.equal(pb):
                        return False
                elif pa != pb:
                    return False
        elif va != vb:
            return False
    return True


def _worker(args):
    """Structural check is not an identity check: the pickled games carry no
    self-identifying field (no ``game_id`` key - see EXPECTED_KEYS in
    build_corpus_store.py), so a well-formed-but-wrong game (served under the
    wrong name by a cursor/connection mixup between forked workers) would
    pass a structural-only check silently. When a loose reference directory
    is available, compare the loaded game against the true content for that
    exact name instead of just checking it looks like *a* valid game."""
    seed, names, n, loose_dir = args
    rnd = random.Random(seed)
    bad = 0
    for _ in range(n):
        name = rnd.choice(names)
        g = STORE.load(name)
        if loose_dir is not None:
            with open(os.path.join(loose_dir, name), "rb") as f:
                ref = pickle.load(f)
            if not games_equal(g, ref):
                bad += 1
        elif len(g["Positions"]) < 1 or len(g["Clocks"]) != len(g["Positions"]):
            bad += 1
    return os.getpid(), bad


def main() -> int:
    global STORE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="store directory (holds corpus.sqlite)")
    ap.add_argument("--loose", help="loose .pkl directory to compare against")
    ap.add_argument("--workers", type=int, default=8, help="forked readers (default 8)")
    ap.add_argument("--reads", type=int, default=500, help="reads per worker")
    ap.add_argument("--sample", type=int, default=0,
                    help="compare only N random games instead of all (0 = all)")
    ap.add_argument("--time-reads", type=int, default=2000,
                    help="random reads for the timing comparison")
    args = ap.parse_args()

    STORE = GameBlobStore.open_if_present(args.store)
    if STORE is None:
        print(f"no corpus.sqlite in {args.store}", file=sys.stderr)
        return 1
    print(f"store: {STORE.path}\n")

    names = STORE.names()
    check("store is non-empty", len(names) > 0, f"{len(names):,} games")
    check("names are sorted and unique",
          names == sorted(names) and len(names) == len(set(names)))
    check("every name ends in .pkl", all(n.endswith(".pkl") for n in names))

    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    print(f"       split manifest sha256 = {digest}")

    # ---- comparisons that need the loose directory -----------------------
    if args.loose:
        loose_names = sorted(f for f in os.listdir(args.loose) if f.endswith(".pkl"))
        same = names == loose_names
        check("name list identical to os.listdir(loose)", same,
              f"{len(names):,} vs {len(loose_names):,}")
        if not same:
            only_store = set(names) - set(loose_names)
            only_loose = set(loose_names) - set(names)
            print(f"       only in store: {len(only_store)}  only in loose: {len(only_loose)}")
            for n in list(only_store)[:3]:
                print(f"         store-only: {n}")
            for n in list(only_loose)[:3]:
                print(f"         loose-only: {n}")

        d_loose = hashlib.sha256("\n".join(loose_names).encode("utf-8")).hexdigest()
        check("split_manifest sha256 unchanged", d_loose == digest,
              "the frozen split stays valid" if d_loose == digest
              else f"loose={d_loose[:16]}... store={digest[:16]}...")

        subject = names if not args.sample else random.Random(0).sample(names, min(args.sample, len(names)))
        bad, checked = 0, 0
        t0 = time.time()
        for n in subject:
            with open(os.path.join(args.loose, n), "rb") as f:
                ref = pickle.load(f)
            if not games_equal(STORE.load(n), ref):
                bad += 1
                if bad <= 3:
                    print(f"       first mismatches: {n}")
            checked += 1
        check(f"round-trip equality over {checked:,} games", bad == 0,
              f"{bad} mismatches, {time.time()-t0:.0f}s")

    # ---- concurrency -----------------------------------------------------
    STORE.load(names[0])          # parent opens a connection first, on purpose
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        out = pool.map(_worker, [(i, names, args.reads, args.loose) for i in range(args.workers)])
    pids = {pid for pid, _ in out}
    corrupt = sum(b for _, b in out)
    check(f"{args.workers} forked workers x {args.reads} reads",
          corrupt == 0 and len(pids) == args.workers,
          f"{len(pids)} distinct pids, {corrupt} corrupt reads")

    revived = pickle.loads(pickle.dumps(STORE))
    check("store survives pickling to a spawned worker",
          games_equal(revived.load(names[0]), STORE.load(names[0])))

    # ---- timing ----------------------------------------------------------
    rnd = random.Random(4242)
    order = [rnd.choice(names) for _ in range(args.time_reads)]
    for n in order[:200]:
        STORE.load(n)
    t0 = time.perf_counter()
    for n in order:
        STORE.load(n)
    store_ms = (time.perf_counter() - t0) / len(order) * 1000
    print(f"\n       store : {store_ms:.3f} ms/game (random, single-threaded)")

    if args.loose:
        for n in order[:200]:
            with open(os.path.join(args.loose, n), "rb") as f:
                pickle.load(f)
        t0 = time.perf_counter()
        for n in order:
            with open(os.path.join(args.loose, n), "rb") as f:
                pickle.load(f)
        loose_ms = (time.perf_counter() - t0) / len(order) * 1000
        print(f"       loose : {loose_ms:.3f} ms/game")
        print(f"       delta : {store_ms - loose_ms:+.3f} ms/game "
              f"({store_ms/loose_ms:.2f}x)")
        print("       (compare against the 2026-08-24 ZIP measurement: loose 3.791, "
              "ZIP 5.396, delta +1.605)")
        print("       NOTE: at full-corpus scale the loose set is ~500GB and cannot sit in\n"
              "       page cache, while the store is ~11GB and can. A benchmark on a small\n"
              "       subset where BOTH fit in cache measures CPU only and understates the\n"
              "       store. Re-read the caveat in analysis/corpus-storage-solution.md.")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

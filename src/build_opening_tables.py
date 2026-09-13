#!/usr/bin/env python3
"""Build per-band real-opening frequency tables for the v2 synthetic anomaly
corpus (design: data/anomaly-corpus-v2/design.md, decision 1).

Every synthetic v2 game is seeded with a real opening line drawn from the same
Lichess corpus the synthetic set is meant to stand in for (2,550,000 games,
2019-01..2026-07). This script walks that corpus once, extracts the first
``--line-plies`` plies (UCI) of every game, buckets each game by the 200-point
rating band centred on the Maia band it matches, and writes one
frequency table per band.

Band assignment
---------------
A game's band key is the average of WhiteElo and BlackElo. It is added to band
B when ``B - 100 <= avg < B + 100`` (a 200-point window centred on B). The
windows of adjacent Maia bands overlap by 100 points, so a game near a band
edge contributes to both neighbouring tables; this is deliberate (it is a
frequency table used only for opening seeding, and the overlap smooths the
tail). The two edge bands are half-open: band 1100 takes everything with
avg < 1200, band 1900 everything with avg >= 1800, so every game lands in at
least one table.

Only games with at least ``--line-plies`` plies are kept (a full-length line is
needed; the generator replays a 6-10 ply prefix of it).

Input
-----
Either a random-access blob store built by ``build_corpus_store.py``
(``--store <dir with corpus.sqlite>`` or ``--store <path to .sqlite>``) or a
directory of ``*.tar.gz`` per-month archives (``--archives <dir>``). The store
path is the fast one on HPC (``data/corpus_store_backup.sqlite``).

Output
------
``<out>/openings_band{B}.json`` for B in 1100..1900 step 100, plus
``<out>/opening_tables_summary.json``. Each per-band file:

    {
      "band": 1500,
      "line_plies": 12,
      "n_games": 283145,           # games that landed in this band
      "n_unique_lines": 197034,
      "coverage": 1.0,             # kept count mass / total (1.0 = nothing dropped)
      "lines": [["e2e4 e7e5 ...", 4021], ...]   # UCI moves space-joined, count desc
    }

The generator samples a line with probability proportional to
``count ** opening_temperature`` (default 0.7 < 1, which flattens toward the
long tail so rare-but-real lines still appear); see the generator's
``--opening-temperature``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tarfile
import time
import zlib
from collections import Counter
from pathlib import Path

BANDS = list(range(1100, 2000, 100))


def band_keys(avg: float) -> list[int]:
    """Every Maia band whose 200-point window contains ``avg`` (edge bands
    half-open so nothing is dropped)."""
    keys = []
    for b in BANDS:
        lo = b - 100
        hi = b + 100
        if b == BANDS[0]:
            lo = float("-inf")
        if b == BANDS[-1]:
            hi = float("inf")
        if lo <= avg < hi:
            keys.append(b)
    return keys


def _to_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


class _NoTorchUnpickler(pickle.Unpickler):
    """Unpickle a corpus game without importing torch or rebuilding the
    position tensors. This script only needs Moves / WhiteElo / BlackElo, and
    torch tensor reconstruction is the bulk of the per-game cost. Any torch
    global is replaced with a stub that consumes its args and returns None, so
    ``Positions`` deserializes to a list of Nones.
    """

    def find_class(self, module: str, name: str):
        if module.split(".", 1)[0] == "torch":
            return lambda *a, **k: None
        return super().find_class(module, name)


def _loads_fast(raw: bytes) -> dict:
    import io as _io

    return _NoTorchUnpickler(_io.BytesIO(raw)).load()


def iter_store_games(sqlite_path: Path, nshards: int = 1, shard: int = 0):
    import sqlite3

    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        if nshards > 1:
            cur = con.execute(
                "SELECT blob FROM games WHERE (rowid % ?) = ?", (nshards, shard))
        else:
            cur = con.execute("SELECT blob FROM games")
        while True:
            rows = cur.fetchmany(2000)
            if not rows:
                break
            for (blob,) in rows:
                try:
                    data = zlib.decompress(blob)
                except zlib.error:
                    data = blob
                try:
                    yield _loads_fast(data)
                except Exception:
                    yield pickle.loads(data)
    finally:
        con.close()


def iter_archive_games(archives_dir: Path):
    for arch in sorted(archives_dir.glob("*.tar.gz")):
        with tarfile.open(arch, "r|gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".pkl"):
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                raw = fobj.read()
                try:
                    yield pickle.loads(raw)
                except pickle.UnpicklingError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--store", type=Path, help="corpus.sqlite (or its parent dir)")
    src.add_argument("--archives", type=Path, help="dir of *.tar.gz month archives")
    ap.add_argument("--out", type=Path, required=True, help="output dir for openings_band*.json")
    ap.add_argument("--line-plies", type=int, default=12,
                    help="plies stored per opening line (default 12)")
    ap.add_argument("--max-lines-per-band", type=int, default=0,
                    help="if >0, keep only the top-N lines by count per band "
                         "(0 = keep all; unique lines never exceed games in a band)")
    ap.add_argument("--log-every", type=int, default=200000)
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the store scan into N interleaved shards (rowid %% N); "
                         "run one process per shard, then --finalize")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index [0, nshards)")
    ap.add_argument("--finalize", action="store_true",
                    help="merge <out>/_parts/part_*.pkl into the final per-band JSON + summary")
    args = ap.parse_args()

    parts_dir = args.out / "_parts"

    if args.finalize:
        return _finalize(args, parts_dir)

    if args.store is not None:
        p = args.store
        if p.is_dir():
            p = p / "corpus.sqlite"
        if not p.exists():
            ap.error(f"store not found: {p}")
        game_iter = iter_store_games(p, args.nshards, args.shard)
        source = str(p)
    else:
        if args.nshards > 1:
            ap.error("--nshards is only supported with --store")
        if not args.archives.is_dir():
            ap.error(f"archives dir not found: {args.archives}")
        game_iter = iter_archive_games(args.archives)
        source = str(args.archives)

    args.out.mkdir(parents=True, exist_ok=True)
    counters: dict[int, Counter] = {b: Counter() for b in BANDS}
    band_games: dict[int, int] = {b: 0 for b in BANDS}

    seen = kept = skipped_short = skipped_norating = 0
    t0 = time.monotonic()
    for obj in game_iter:
        seen += 1
        moves = obj.get("Moves") or []
        if len(moves) < args.line_plies:
            skipped_short += 1
        else:
            we = _to_int(obj.get("WhiteElo"))
            be = _to_int(obj.get("BlackElo"))
            if we is None or be is None:
                skipped_norating += 1
            else:
                avg = (we + be) / 2.0
                line = " ".join(moves[: args.line_plies])
                for b in band_keys(avg):
                    counters[b][line] += 1
                    band_games[b] += 1
                kept += 1
        if seen % args.log_every == 0:
            rate = seen / (time.monotonic() - t0)
            tag = f"shard {args.shard}/{args.nshards} " if args.nshards > 1 else ""
            print(f"{tag}seen={seen} kept={kept} short={skipped_short} "
                  f"norating={skipped_norating} rate={rate:.0f}/s", flush=True)

    if args.nshards > 1:
        parts_dir.mkdir(parents=True, exist_ok=True)
        part = {
            "counters": {b: dict(counters[b]) for b in BANDS},
            "band_games": band_games,
            "stats": {"seen": seen, "kept": kept, "skipped_short": skipped_short,
                      "skipped_norating": skipped_norating},
            "source": source, "line_plies": args.line_plies,
        }
        part_path = parts_dir / f"part_{args.shard:03d}.pkl"
        with open(part_path, "wb") as f:
            pickle.dump(part, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"WROTE {part_path} (seen={seen} kept={kept}, {time.monotonic() - t0:.0f}s)", flush=True)
        return 0

    summary = {
        "source": source,
        "line_plies": args.line_plies,
        "games_seen": seen,
        "games_kept": kept,
        "skipped_short": skipped_short,
        "skipped_norating": skipped_norating,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "bands": {},
    }

    for b in BANDS:
        items = counters[b].most_common()
        total_mass = sum(c for _, c in items)
        if args.max_lines_per_band > 0 and len(items) > args.max_lines_per_band:
            items = items[: args.max_lines_per_band]
        kept_mass = sum(c for _, c in items)
        coverage = (kept_mass / total_mass) if total_mass else 0.0
        payload = {
            "band": b,
            "line_plies": args.line_plies,
            "n_games": band_games[b],
            "n_unique_lines": len(counters[b]),
            "n_lines_kept": len(items),
            "coverage": round(coverage, 6),
            "lines": [[line, c] for line, c in items],
        }
        out_path = args.out / f"openings_band{b}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f)
        summary["bands"][str(b)] = {
            "n_games": band_games[b],
            "n_unique_lines": len(counters[b]),
            "n_lines_kept": len(items),
            "coverage": round(coverage, 6),
            "file": out_path.name,
        }
        print(f"band {b}: games={band_games[b]} unique_lines={len(counters[b])} "
              f"kept={len(items)} coverage={coverage:.4f}", flush=True)

    with open(args.out / "opening_tables_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary["bands"]), flush=True)
    return 0


def _finalize(args, parts_dir: Path) -> int:
    part_files = sorted(parts_dir.glob("part_*.pkl"))
    if not part_files:
        print(f"no part_*.pkl in {parts_dir}", file=sys.stderr)
        return 1
    counters: dict[int, Counter] = {b: Counter() for b in BANDS}
    band_games: dict[int, int] = {b: 0 for b in BANDS}
    agg = {"seen": 0, "kept": 0, "skipped_short": 0, "skipped_norating": 0}
    source = line_plies = None
    for pf in part_files:
        with open(pf, "rb") as f:
            part = pickle.load(f)
        source = part["source"]
        line_plies = part["line_plies"]
        for b in BANDS:
            counters[b].update(part["counters"].get(b, {}))
            band_games[b] += part["band_games"].get(b, 0)
        for k in agg:
            agg[k] += part["stats"][k]
        print(f"merged {pf.name}: seen={part['stats']['seen']}", flush=True)

    summary = {"source": source, "line_plies": line_plies, "n_parts": len(part_files),
               "games_seen": agg["seen"], "games_kept": agg["kept"],
               "skipped_short": agg["skipped_short"], "skipped_norating": agg["skipped_norating"],
               "bands": {}}
    for b in BANDS:
        items = counters[b].most_common()
        total_mass = sum(c for _, c in items)
        if args.max_lines_per_band > 0 and len(items) > args.max_lines_per_band:
            items = items[: args.max_lines_per_band]
        kept_mass = sum(c for _, c in items)
        coverage = (kept_mass / total_mass) if total_mass else 0.0
        payload = {"band": b, "line_plies": line_plies, "n_games": band_games[b],
                   "n_unique_lines": len(counters[b]), "n_lines_kept": len(items),
                   "coverage": round(coverage, 6),
                   "lines": [[line, c] for line, c in items]}
        with open(args.out / f"openings_band{b}.json", "w") as f:
            json.dump(payload, f)
        summary["bands"][str(b)] = {"n_games": band_games[b], "n_unique_lines": len(counters[b]),
                                    "n_lines_kept": len(items), "coverage": round(coverage, 6),
                                    "file": f"openings_band{b}.json"}
        print(f"band {b}: games={band_games[b]} unique_lines={len(counters[b])} "
              f"kept={len(items)} coverage={coverage:.4f}", flush=True)
    with open(args.out / "opening_tables_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary["bands"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

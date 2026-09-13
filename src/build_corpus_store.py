#!/usr/bin/env python3
"""Convert the per-month ``*.tar.gz`` corpus archives into a single random-access
SQLite blob store for training.

Why this exists
---------------
``ChessGamesDataset`` reads one loose ``.pkl`` per game, and the DataLoader
shuffles globally every epoch, so training needs cheap random access to any one
game.  The full corpus is ~2.46M games at ~203KB each, which is ~500GB loose and
does not fit the ~318GB NVMe scratch disk.  The archives it is currently stored
in are single gzip streams, which cannot be seeked into.

Compressing each game independently solves both problems at once: the whole
corpus lands at roughly 11GB (per-game DEFLATE keeps ~97% of the ratio the solid
tar.gz stream gets, because the compressibility comes from long zero runs inside
each position tensor, not from redundancy across games), and any single game can
be fetched with one primary-key lookup.

Layout::

    <store_dir>/
        corpus.sqlite        # this script writes it
        split_manifest.json  # written by chess_rating_net.py on first run

Point ``--data_dir`` at ``<store_dir>``.  ``chess_rating_net.py`` uses the store
when ``corpus.sqlite`` is present and falls back to loose ``.pkl`` files
otherwise, so nothing changes for the existing 170k flat directory.

Usage
-----
Build (resumable, safe to re-run after an interruption)::

    python src/build_corpus_store.py build \
        --archives ~/Bacsain/thesis2/data/corpus_archive_transfer \
        --store /tmp/ratingnet_store --jobs 8

Verify every game round-trips byte-for-byte against the source archives::

    python src/build_corpus_store.py verify \
        --archives ~/Bacsain/thesis2/data/corpus_archive_transfer \
        --store /tmp/ratingnet_store

Inspect what is in a store::

    python src/build_corpus_store.py stat --store /tmp/ratingnet_store

Cost and safety notes
---------------------
* **Peak transient disk is the output only.**  Archives are streamed with
  ``tarfile.open(..., "r|gz")``, one member held in memory at a time, so a month
  is never extracted to disk.  You need room for the store (~11GB) and nothing
  else.
* **Resumable at archive granularity.**  Each archive is one transaction and is
  recorded in the ``archives`` table on success.  A crash rolls that archive back
  whole; re-running skips the archives already done.
* **The source archives are never modified.**  This script only reads them.
* Compression is the bottleneck at roughly 0.9ms/game single-threaded, so about
  40 minutes of CPU for 2.46M games.  ``--jobs`` spreads that across processes.
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
import tarfile
import time
import zlib
from pathlib import Path

SCHEMA_VERSION = 1
STORE_FILENAME = "corpus.sqlite"
COMPRESS_LEVEL = 6

# The schema every game pickle is expected to carry, per
# prototype/corpus-ops/fm-corpus-archive-month.py's own verification step.
EXPECTED_KEYS = {
    "Black", "BlackElo", "Clocks", "Moves",
    "Positions", "Result", "Time", "White", "WhiteElo",
}


# --------------------------------------------------------------------------- #
# store plumbing
# --------------------------------------------------------------------------- #

def store_path(store_dir: str) -> Path:
    return Path(store_dir) / STORE_FILENAME


def connect_rw(store_dir: str) -> sqlite3.Connection:
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(store_path(store_dir))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-65536")  # 64MB page cache while building
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS games (
            name TEXT PRIMARY KEY,
            blob BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS archives (
            archive      TEXT PRIMARY KEY,
            n_games      INTEGER NOT NULL,
            source_bytes INTEGER NOT NULL,
            stored_bytes INTEGER NOT NULL,
            completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    con.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    con.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('compression', ?)",
        (f"zlib:{COMPRESS_LEVEL}",),
    )
    con.commit()
    return con


def _compress(item: tuple[str, bytes]) -> tuple[str, bytes, int]:
    name, raw = item
    return name, zlib.compress(raw, COMPRESS_LEVEL), len(raw)


def iter_archive(path: Path):
    """Yield ``(member_name, raw_bytes)`` for every .pkl in a tar.gz, streaming.

    ``r|gz`` is a forward-only stream: no seeking, no index build, constant
    memory.  That is exactly right here since conversion is one sequential pass.
    Non-.pkl members (``.sampling_summary.json``) are skipped, matching
    ``os.listdir`` + ``endswith('.pkl')`` in the trainer.

    Every month's games reuse identical basenames (``game_0000000.pkl`` ..
    ``game_0029999.pkl`` in every archive - confirmed directly against real
    archives, not assumed), so the store key is prefixed with the archive's
    own month stem (``2019-07_game_0000000.pkl``), matching the ``{month}_``
    convention already used elsewhere in this project (the flatten step,
    the model_55.pth eval directory). Without this, the second archive ever
    built always collides with the first on `games.name` - caught by running
    this against real archives rather than the synthetic 2000-game test,
    where sequentially-generated fake ids never collided by construction.
    """
    month = path.name.removesuffix(".tar.gz")
    with tarfile.open(path, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".pkl"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            yield f"{month}_{os.path.basename(member.name)}", fh.read()


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def cmd_build(args: argparse.Namespace) -> int:
    archives = sorted(Path(args.archives).glob("*.tar.gz"))
    if not archives:
        print(f"no *.tar.gz found in {args.archives}", file=sys.stderr)
        return 1

    con = connect_rw(args.store)
    done = {row[0] for row in con.execute("SELECT archive FROM archives")}
    todo = [a for a in archives if a.name not in done]

    print(f"archives: {len(archives)} found, {len(done)} already done, {len(todo)} to do")
    if not todo:
        print("nothing to do")
        return _summarize(con)

    pool = mp.Pool(args.jobs) if args.jobs > 1 else None
    t_start = time.time()
    total_games = 0

    try:
        for i, arch in enumerate(todo, 1):
            t0 = time.time()
            src_bytes = 0
            out_bytes = 0
            n = 0
            rows: list[tuple[str, bytes]] = []

            stream = iter_archive(arch)
            mapped = (
                pool.imap(_compress, stream, chunksize=args.chunk)
                if pool is not None
                else map(_compress, stream)
            )

            try:
                for name, comp, raw_len in mapped:
                    rows.append((name, comp))
                    src_bytes += raw_len
                    out_bytes += len(comp)
                    n += 1
                    if len(rows) >= args.batch:
                        con.executemany("INSERT INTO games(name, blob) VALUES (?, ?)", rows)
                        rows.clear()
                if rows:
                    con.executemany("INSERT INTO games(name, blob) VALUES (?, ?)", rows)
            except sqlite3.IntegrityError as exc:
                con.rollback()
                print(
                    f"\nABORT on {arch.name}: duplicate game id ({exc}). "
                    f"Two archives contain the same .pkl basename; the corpus "
                    f"would silently lose a game. Nothing from this archive was "
                    f"written.",
                    file=sys.stderr,
                )
                return 2
            except Exception:
                con.rollback()
                raise

            if n == 0:
                con.rollback()
                print(f"ABORT on {arch.name}: no .pkl members found", file=sys.stderr)
                return 2

            # spot-check straight out of the store before trusting the archive,
            # mirroring what fm-corpus-archive-month.py does on the way in
            bad = _spot_check(con, _sample_names(con, arch, n, args.sample))
            if bad:
                con.rollback()
                print(f"ABORT on {arch.name}: {bad} sampled rows failed schema check", file=sys.stderr)
                return 2

            con.execute(
                "INSERT INTO archives(archive, n_games, source_bytes, stored_bytes, completed_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                (arch.name, n, src_bytes, out_bytes),
            )
            con.commit()
            # WAL-mode commits only land in the -wal file until checkpointed;
            # readers open with mode=ro&immutable=1 (GameBlobStore, and hence
            # check_corpus_store.py) never read the -wal file at all, so a
            # process killed after this commit but before a checkpoint would
            # leave this archive's games durably committed yet invisible to
            # every consumer with no error. Checkpointing per archive (not
            # just at the end) bounds the invisible window to at most the
            # one archive currently in flight, which resumability already
            # handles by simply redoing it (it isn't in `archives` yet).
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

            total_games += n
            dt = time.time() - t0
            ratio = src_bytes / out_bytes if out_bytes else 0
            print(
                f"[{i}/{len(todo)}] {arch.name}: {n:,} games, "
                f"{src_bytes/1e9:.2f}GB -> {out_bytes/1e6:.0f}MB ({ratio:.1f}x), "
                f"{dt:.0f}s ({n/dt:.0f} games/s)",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    print(f"\nconverted {total_games:,} games in {(time.time()-t_start)/60:.1f} min")
    return _summarize(con)


def _sample_names(con, arch, n, k):
    """Random sample of the names just inserted, via `ORDER BY rowid DESC LIMIT n`.

    Reliable only because this is single-writer with no deletes: the n most
    recent rowids are exactly this archive's rows. If concurrent writers or
    row deletion are ever introduced, this must sample from `iter_archive(arch)`
    output directly instead - do not "fix" this by trusting an outdated
    docstring that once described that approach as already in use here."""
    rnd = random.Random(hashlib.sha256(arch.name.encode()).hexdigest())
    rows = [r[0] for r in con.execute(
        "SELECT name FROM games ORDER BY rowid DESC LIMIT ?", (n,))]
    return rnd.sample(rows, min(k, len(rows)))


def _spot_check(con, names) -> int:
    bad = 0
    for name in names:
        row = con.execute("SELECT blob FROM games WHERE name=?", (name,)).fetchone()
        if row is None:
            bad += 1
            continue
        try:
            obj = pickle.loads(zlib.decompress(row[0]))
        except Exception:
            bad += 1
            continue
        if set(obj.keys()) != EXPECTED_KEYS:
            bad += 1
    return bad


def _summarize(con) -> int:
    n, = con.execute("SELECT COUNT(*) FROM games").fetchone()
    src, out = con.execute(
        "SELECT COALESCE(SUM(source_bytes),0), COALESCE(SUM(stored_bytes),0) FROM archives"
    ).fetchone()
    print(f"store now holds {n:,} games")
    if out:
        print(f"  source {src/1e9:.1f}GB -> stored {out/1e9:.2f}GB ({src/out:.1f}x)")
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def cmd_verify(args: argparse.Namespace) -> int:
    """Re-stream the archives and compare every game byte-for-byte with the store.

    This is the check that has to pass before the store is trusted for a training
    run: it proves the store returns exactly the bytes the archive holds, for
    every game, not for a sample.
    """
    archives = sorted(Path(args.archives).glob("*.tar.gz"))
    con = sqlite3.connect(f"file:{store_path(args.store)}?mode=ro", uri=True)

    if args.limit:
        archives = archives[: args.limit]

    checked = mismatched = missing = schema_bad = 0
    t0 = time.time()
    for arch in archives:
        for name, raw in iter_archive(arch):
            row = con.execute("SELECT blob FROM games WHERE name=?", (name,)).fetchone()
            if row is None:
                missing += 1
                if missing <= 5:
                    print(f"  MISSING {name} (from {arch.name})")
                continue
            got = zlib.decompress(row[0])
            if got != raw:
                mismatched += 1
                if mismatched <= 5:
                    print(f"  MISMATCH {name} (from {arch.name}): "
                          f"{len(raw)} source bytes vs {len(got)} stored bytes")
            else:
                obj = pickle.loads(got)
                if set(obj.keys()) != EXPECTED_KEYS:
                    schema_bad += 1
            checked += 1
        print(f"  {arch.name}: {checked:,} checked so far", flush=True)

    n_store, = con.execute("SELECT COUNT(*) FROM games").fetchone()
    print(f"\nchecked {checked:,} games in {(time.time()-t0)/60:.1f} min")
    print(f"  byte-identical mismatches : {mismatched}")
    print(f"  missing from store        : {missing}")
    print(f"  schema mismatches         : {schema_bad}")
    print(f"  rows in store             : {n_store:,}")
    count_mismatch = (not args.limit) and n_store != checked
    if count_mismatch:
        print(f"  WARNING: store holds {n_store - checked:+,} rows vs archives streamed "
              f"(these rows were never checked against a source archive)")
    ok = (mismatched == 0 and missing == 0 and schema_bad == 0 and not count_mismatch)
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# stat
# --------------------------------------------------------------------------- #

def cmd_stat(args: argparse.Namespace) -> int:
    p = store_path(args.store)
    if not p.exists():
        print(f"no store at {p}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    n, = con.execute("SELECT COUNT(*) FROM games").fetchone()
    print(f"store        : {p}")
    print(f"file size    : {p.stat().st_size/1e9:.2f} GB")
    print(f"games        : {n:,}")
    for k, v in con.execute("SELECT key, value FROM meta ORDER BY key"):
        print(f"meta.{k:<9}: {v}")
    print(f"archives     : {con.execute('SELECT COUNT(*) FROM archives').fetchone()[0]}")

    names = [r[0] for r in con.execute("SELECT name FROM games ORDER BY name LIMIT 3")]
    print(f"first names  : {names}")

    # the split manifest is computed over the sorted basenames, so print its hash
    all_names = [r[0] for r in con.execute("SELECT name FROM games ORDER BY name")]
    digest = hashlib.sha256("\n".join(all_names).encode("utf-8")).hexdigest()
    print(f"split sha256 : {digest}")
    print("  (this is the value chess_rating_net.py will record in split_manifest.json)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="convert archives into the store (resumable)")
    b.add_argument("--archives", required=True, help="directory of *.tar.gz month archives")
    b.add_argument("--store", required=True, help="output store directory")
    b.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                   help="parallel compression workers (default: half the cores)")
    b.add_argument("--batch", type=int, default=512, help="rows per executemany")
    b.add_argument("--chunk", type=int, default=32, help="games per worker chunk")
    b.add_argument("--sample", type=int, default=20, help="rows to spot-check per archive")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("verify", help="byte-for-byte check of the store against the archives")
    v.add_argument("--archives", required=True)
    v.add_argument("--store", required=True)
    v.add_argument("--limit", type=int, default=0, help="check only the first N archives")
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("stat", help="summarize a store")
    s.add_argument("--store", required=True)
    s.set_defaults(func=cmd_stat)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

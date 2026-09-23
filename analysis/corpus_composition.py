"""Per-time-control composition and 100-ply-truncation rate for the full corpus.

Reads every game record out of the corpus SQLite store (`--store`, default
`/tmp/ratingnet_store/corpus.sqlite`) and classifies each game's time control
using the exact same rule the training pipeline uses at load time
(`chess_rating_net.py`'s `ChessGamesDataset.__getitem__`):

    initial_time, increment = map(int, game_info["Time"].split("+"))
    estimated_duration = initial_time + 40 * increment
    time_control = categorize_time_control(estimated_duration)

Truncation is estimated from the stored ply count: `parse_game` (and its
one-pass/two-pass equivalents) stop writing positions/moves either when the
game ends naturally or when the ply cap (100) is reached, and no separate
"was this game longer than the cap" flag is persisted. A game whose stored
`Moves` list has exactly `--max-plies` entries is counted as truncated. This
is a proxy, not ground truth: a game that legitimately ends at exactly the
cap ply (rare) would be misclassified as truncated. That caveat is reported
alongside the number, not silently absorbed into it.

Usage (on the HPC host, where the store lives):
    python analysis/scripts/corpus_composition.py --store /tmp/ratingnet_store/corpus.sqlite \
        --out analysis/heldout_test_eval/corpus_composition.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sqlite3
import sys
import time
import zlib

TIME_CONTROLS = ["ultrabullet", "bullet", "blitz", "rapid", "classical"]


def categorize_time_control(estimated_duration: int) -> str:
    if estimated_duration < 29:
        return "ultrabullet"
    elif estimated_duration < 179:
        return "bullet"
    elif estimated_duration < 479:
        return "blitz"
    elif estimated_duration < 1499:
        return "rapid"
    else:
        return "classical"


def classify_blob(blob: bytes, max_plies: int) -> tuple[str, bool] | None:
    try:
        game_info = pickle.loads(zlib.decompress(blob))
        initial_time, increment = map(int, game_info["Time"].split("+"))
        estimated_duration = initial_time + 40 * increment
        tc = categorize_time_control(estimated_duration)
        n_plies = len(game_info["Moves"])
        return tc, n_plies >= max_plies
    except Exception:
        return None


def _worker(args: tuple[bytes, int]) -> tuple[str, bool] | None:
    blob, max_plies = args
    return classify_blob(blob, max_plies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="/tmp/ratingnet_store/corpus.sqlite")
    parser.add_argument("--out", default="analysis/heldout_test_eval/corpus_composition.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--max-plies", type=int, default=100)
    parser.add_argument("--log-every-chunks", type=int, default=5)
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.store}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM games")
    total = cur.fetchone()[0]
    print(f"total games in store: {total}", file=sys.stderr)

    tc_counts = {tc: 0 for tc in TIME_CONTROLS}
    tc_truncated = {tc: 0 for tc in TIME_CONTROLS}
    n_parsed = 0
    n_failed = 0

    t0 = time.time()
    pool = mp.Pool(args.workers)
    last_rowid = 0
    chunk_idx = 0
    try:
        while True:
            cur.execute(
                "SELECT rowid, blob FROM games WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, args.chunk_size),
            )
            rows = cur.fetchall()
            if not rows:
                break
            last_rowid = rows[-1][0]
            blobs = [(b, args.max_plies) for _, b in rows]
            for result in pool.imap_unordered(_worker, blobs, chunksize=200):
                if result is None:
                    n_failed += 1
                    continue
                tc, truncated = result
                tc_counts[tc] += 1
                if truncated:
                    tc_truncated[tc] += 1
                n_parsed += 1
            chunk_idx += 1
            if chunk_idx % args.log_every_chunks == 0:
                elapsed = time.time() - t0
                rate = n_parsed / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{elapsed:7.1f}s] {n_parsed}/{total} parsed "
                    f"({rate:.0f} games/s, {n_failed} failed)",
                    file=sys.stderr,
                )
    finally:
        pool.close()
        pool.join()

    n_total = sum(tc_counts.values())
    result = {
        "store": args.store,
        "n_total_games": n_total,
        "n_failed_to_parse": n_failed,
        "max_plies": args.max_plies,
        "elapsed_seconds": time.time() - t0,
        "by_time_control": {
            tc: {
                "count": tc_counts[tc],
                "pct_of_corpus": round(100.0 * tc_counts[tc] / n_total, 4) if n_total else None,
                "n_truncated_at_cap": tc_truncated[tc],
                "pct_truncated_within_tc": (
                    round(100.0 * tc_truncated[tc] / tc_counts[tc], 4) if tc_counts[tc] else None
                ),
            }
            for tc in TIME_CONTROLS
        },
        "overall_pct_truncated_at_cap": (
            round(100.0 * sum(tc_truncated.values()) / n_total, 4) if n_total else None
        ),
        "truncation_caveat": (
            "n_truncated_at_cap counts games whose stored ply count equals "
            "max_plies exactly, since the store does not retain a separate "
            "truncation flag or the pre-truncation game length. A game that "
            "legitimately ends at exactly the cap ply would be "
            "misclassified as truncated; this is expected to be rare."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

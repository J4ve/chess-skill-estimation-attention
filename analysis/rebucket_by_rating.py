"""Re-bucket an existing per-game held-out-test CSV by rating bracket.

Does NOT re-run the model. It reads the per-game CSV already produced by
``score_test_split.py`` (columns: game_id, white_err, black_err,
white_signed_err, black_signed_err, time_control, white_elo, black_elo) and
buckets by the five Chapter 3 rating brackets (0-1200, 1201-1600, 1601-2000,
2001-2400, 2401+).

A game contributes two samples, not one: (white_elo, white_err) and
(black_elo, black_err), since the two sides can fall in different brackets.
This mirrors how the model is scored per side; pooling every bracket's
samples back together reproduces the already-published overall MAE
(``overall_mae_rating_points`` in the arm's JSON) as a consistency check,
printed at the end of this script's output.

Usage:
    python analysis/scripts/rebucket_by_rating.py \
        --csv analysis/heldout_test_eval/baseline__best.csv \
        --out analysis/heldout_test_eval/baseline__best__rating_buckets.json
"""

from __future__ import annotations

import argparse
import csv
import json

BRACKETS = [
    ("0-1200", 0, 1200),
    ("1201-1600", 1201, 1600),
    ("1601-2000", 1601, 2000),
    ("2001-2400", 2001, 2400),
    ("2401+", 2401, None),
]


def bracket_for(elo: float) -> str:
    for name, lo, hi in BRACKETS:
        if elo >= lo and (hi is None or elo <= hi):
            return name
    raise ValueError(f"unbracketed elo: {elo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    sums = {name: 0.0 for name, _, _ in BRACKETS}
    counts = {name: 0 for name, _, _ in BRACKETS}
    total_sum = 0.0
    total_n = 0

    with open(args.csv) as f:
        for row in csv.DictReader(f):
            for elo_key, err_key in (("white_elo", "white_err"), ("black_elo", "black_err")):
                elo = float(row[elo_key])
                err = float(row[err_key])
                b = bracket_for(elo)
                sums[b] += err
                counts[b] += 1
                total_sum += err
                total_n += 1

    result = {
        "label": args.label or args.csv,
        "source_csv": args.csv,
        "by_rating_bracket": {
            name: {
                "mae_rating_points": round(sums[name] / counts[name], 4) if counts[name] else None,
                "n": counts[name],
            }
            for name, _, _ in BRACKETS
        },
        "overall_mae_rating_points_pooled": round(total_sum / total_n, 4) if total_n else None,
        "overall_n_pooled": total_n,
        "note": (
            "Each game contributes two samples (white side, black side), so "
            "n sums to 2x the game count and overall_mae_rating_points_pooled "
            "should match the arm's overall_mae_rating_points from its "
            "score_test_split.py JSON as a consistency check."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Paired bootstrap over per-game absolute errors on the frozen 255k TEST split.

Given two per-game-error CSVs (schema from ``score_test_split.py`` /
``chess_rating_net.test``), match rows on ``game_id``, form the per-game paired
delta ``mean_abs_err_A - mean_abs_err_B`` (per-game err = (white_err+black_err)/2,
matching ``analysis/scripts/paired_bootstrap.py``), and bootstrap the mean delta
with 10,000 resamples for a 95% percentile CI.

A negative delta means arm A has the lower error (A better). The CI "crosses
zero" verdict is the headline: if it does not cross zero, the difference is
distinguishable from resampling noise on the test set.

Usage:
    python analysis/scripts/paired_bootstrap_heldout.py \
        --pair tuned_attention_vs_baseline A=<attn.csv> B=<baseline.csv> \
        --pair baseline_lr3e4_vs_baseline A=<lr3e4.csv> B=<baseline.csv> \
        --out_json analysis/heldout_test_eval/bootstrap.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random

N_RESAMPLES = 10_000
SEED = 12345


def load(path: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            w = float(row["white_err"])
            b = float(row["black_err"])
            out[row["game_id"]] = (w + b) / 2.0
    return out


def bootstrap_ci(deltas: list[float], n_resamples: int = N_RESAMPLES):
    n = len(deltas)
    means = []
    for _ in range(n_resamples):
        sample = random.choices(deltas, k=n)
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    point = sum(deltas) / n
    frac_neg = sum(1 for m in means if m < 0) / n_resamples
    frac_pos = sum(1 for m in means if m > 0) / n_resamples
    return {
        "point_delta_mae": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_crosses_zero": lo <= 0.0 <= hi,
        "frac_resamples_A_better": frac_neg,
        "frac_resamples_B_better": frac_pos,
        "n_games": n,
    }


def parse_pair(tokens: list[str]) -> tuple[str, str, str]:
    name = tokens[0]
    a = b = None
    for tok in tokens[1:]:
        k, _, v = tok.partition("=")
        if k == "A":
            a = v
        elif k == "B":
            b = v
    if not (a and b):
        raise SystemExit(f"--pair {name}: need A=<csv> B=<csv>")
    return name, a, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", nargs="+", action="append", metavar="NAME A=csv B=csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    results = {"n_resamples": N_RESAMPLES, "seed": SEED, "pairs": {}}
    for tokens in args.pair:
        random.seed(SEED)  # identical resample draws per pair, reproducible
        name, a_path, b_path = parse_pair(tokens)
        a = load(a_path)
        b = load(b_path)
        common = sorted(set(a) & set(b))
        if len(common) != len(a) or len(common) != len(b):
            print(f"WARN {name}: |A|={len(a)} |B|={len(b)} common={len(common)}")
        deltas = [a[g] - b[g] for g in common]
        r = bootstrap_ci(deltas)
        r["A"] = a_path
        r["B"] = b_path
        r["mae_A"] = sum(a[g] for g in common) / len(common)
        r["mae_B"] = sum(b[g] for g in common) / len(common)
        results["pairs"][name] = r
        print(
            f"{name}: dMAE={r['point_delta_mae']:+.3f}  "
            f"95% CI [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]  "
            f"crosses_zero={r['ci_crosses_zero']}  "
            f"(MAE A={r['mae_A']:.3f} B={r['mae_B']:.3f}, n={r['n_games']})"
        )

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

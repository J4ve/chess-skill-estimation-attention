"""Paired bootstrap (attention vs baseline) over the seed-rerun per-game error dumps.

Input: models/<exp>_per_game_errors.csv, one row per test game with
white_err/black_err columns (HPC: ~/Bacsain/thesis2/models/, gitignored — not
committed here). Point results are recorded in analysis/seed-rerun-results.md.

Usage: python3 paired_bootstrap.py <dir containing the 10 abl_*_per_game_errors.csv files>
"""
import csv
import json
import random
import sys
import os

SEEDS = [0, 1, 2, 3, 4]
N_RESAMPLES = 10000


def load(data_dir, exp):
    path = os.path.join(data_dir, f"{exp}_per_game_errors.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            w = float(row["white_err"])
            b = float(row["black_err"])
            out[row["game_id"]] = (w + b) / 2.0
    return out


def bootstrap_ci(deltas, n_resamples=N_RESAMPLES):
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
    return point, lo, hi, frac_neg


def main(data_dir):
    random.seed(12345)
    baseline = {s: load(data_dir, f"abl_baseline_s{s}") for s in SEEDS}
    attention = {s: load(data_dir, f"abl_attention_s{s}") for s in SEEDS}

    ids0 = set(baseline[0].keys())
    for s in SEEDS:
        assert set(baseline[s].keys()) == ids0, f"baseline s{s} game set mismatch"
        assert set(attention[s].keys()) == ids0, f"attention s{s} game set mismatch"
    game_ids = sorted(ids0)

    results = {"n_games": len(game_ids), "per_seed": {}}
    for s in SEEDS:
        deltas = [attention[s][g] - baseline[s][g] for g in game_ids]
        results["per_seed"][s] = bootstrap_ci(deltas)

    base_avg = {g: sum(baseline[s][g] for s in SEEDS) / len(SEEDS) for g in game_ids}
    att_avg = {g: sum(attention[s][g] for s in SEEDS) / len(SEEDS) for g in game_ids}
    deltas_avg = [att_avg[g] - base_avg[g] for g in game_ids]
    results["seed_averaged_caveat_seed_blind"] = bootstrap_ci(deltas_avg)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

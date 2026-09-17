"""Turn per_side_scores.csv (S_att score per side) into p75/p95 cutoffs with
bootstrap 95% CIs, overall and per time control. Same percentile, bootstrap and
n-fallback method as the detector cutoffs (experiments/detector_cutoffs/cutoffs_from_scores.py).
Completes the run that produced the original provisional file: same seed,
same sample definition, just carried through to the full ~2,700-game sample."""
import csv, hashlib, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260915
N_BOOTSTRAP = 2000
MIN_SIDES = 300
PERCENTILES = (75, 95)
CHECKPOINT_MD5 = "fcb38b8a6a3ee04a6b1859dd5bc69b75"
SCORES_CSV = Path(sys.argv[1])
OUT_PATH = Path(sys.argv[2])


def percentile(sv, pct):
    k = (len(sv) - 1) * (pct / 100.0)
    f = int(k); c = min(f + 1, len(sv) - 1)
    if f == c:
        return sv[f]
    return sv[f] * (c - k) + sv[c] * (k - f)


def bootstrap_ci(values, pct, rng):
    n = len(values); est = []
    for _ in range(N_BOOTSTRAP):
        r = sorted(values[rng.randrange(n)] for _ in range(n))
        est.append(percentile(r, pct))
    est.sort()
    return percentile(est, 2.5), percentile(est, 97.5)


def summarize(values, rng):
    sv = sorted(values); out = {"n_sides": len(values)}
    for pct in PERCENTILES:
        lo, hi = bootstrap_ci(values, pct, rng)
        out[f"p{pct}"] = round(percentile(sv, pct), 2)
        out[f"p{pct}_ci"] = [round(lo, 2), round(hi, 2)]
    return out


def main():
    rng = random.Random(SEED)
    by_tc, overall = {}, []
    with SCORES_CSV.open() as f:
        for row in csv.DictReader(f):
            vals = [float(row["white_score"]), float(row["black_score"])]
            by_tc.setdefault(row["time_control"], []).extend(vals)
            overall.extend(vals)
    games = len(overall) // 2
    result = {
        "score": "s_att",
        "generated_date": time.strftime("%Y-%m-%d"),
        "checkpoint": "models/preflight_check_2m/best_model.pth",
        "checkpoint_md5": CHECKPOINT_MD5,
        "provisional": False,
        "source_description": (
            f"Stratified sample of held-out TEST-partition games from the thesis corpus "
            f"(analysis/heldout_test_eval/attn_tuned__best.csv on the HPC checkout; seed {SEED}; "
            "up to 600 games each for bullet, blitz and rapid, every available classical and "
            "ultrabullet game; months with unpacked game files only; at least 20 plies). "
            f"{games} games ({len(overall)} sides) scored on CPU with the exact model + "
            "AnomalyDetector code path the RatingNet web prototype uses. The model never "
            "trained on these games. Completes the run that produced the original "
            "provisional file (stopped early on 2026-09-15 at 1,200 games) at the same "
            "seed, so the first 1,200 games' scores are unchanged."
        ),
        "note": (
            "These cutoffs describe how S_att is distributed across ordinary, "
            "finished, rated human games in the held-out test set. They are not "
            "a cheat-detection threshold: in thesis evaluation S_att separated "
            "engine-substituted synthetic games only weakly (ROC-AUC 0.555), and "
            "a clean synthetic game scored 1097 on this same scale."
        ),
        "bootstrap_resamples": N_BOOTSTRAP,
        "min_sides_for_time_control_cutoff": MIN_SIDES,
        "overall": summarize(overall, rng),
        "by_time_control": {},
    }
    for tc, values in sorted(by_tc.items()):
        if len(values) < MIN_SIDES:
            result["by_time_control"][tc] = {"n_sides": len(values), "use_overall": True,
                                             "reason": f"fewer than {MIN_SIDES} sides"}
        else:
            e = summarize(values, rng); e["use_overall"] = False
            result["by_time_control"][tc] = e
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("overall", "by_time_control")}, indent=1))


main()

"""Turn per_side_scores.csv (detector score per side) into p75/p95 cutoffs with
bootstrap 95% CIs, overall and per time control. Same percentile, bootstrap and
n-fallback method as the S_att cutoffs (scratch/suspicion-cutoffs/cutoffs_from_scores.py)."""
import csv, hashlib, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260915
N_BOOTSTRAP = 2000
MIN_SIDES = 300
PERCENTILES = (75, 95)
DETECTOR_WEIGHTS = Path(sys.argv[1])


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
        out[f"p{pct}"] = round(percentile(sv, pct), 4)
        out[f"p{pct}_ci"] = [round(lo, 4), round(hi, 4)]
    return out


def main():
    rng = random.Random(SEED)
    by_tc, overall = {}, []
    with (HERE / "per_side_scores.csv").open() as f:
        for row in csv.DictReader(f):
            vals = [float(row["white_score"]), float(row["black_score"])]
            by_tc.setdefault(row["time_control"], []).extend(vals)
            overall.extend(vals)
    games = len(overall) // 2
    result = {
        "score": "detector_a3g_seed0",
        "score_description": "Trained per-move detector (thesis arm A3g, seed 0): game-level output in [0, 1] per side. Not the computed attention-weighted score S_att, whose cutoffs live in suspicion_cutoffs_s_att.json.",
        "generated_date": time.strftime("%Y-%m-%d"),
        "detector_weights": "src/models/detector_a3g_seed0.pt",
        "detector_weights_md5": hashlib.md5(DETECTOR_WEIGHTS.read_bytes()).hexdigest(),
        "checkpoint": "models/preflight_check_2m/best_model.pth",
        "checkpoint_md5": "fcb38b8a6a3ee04a6b1859dd5bc69b75",
        "provisional": False,
        "source_description": (
            f"Stratified sample of held-out TEST-partition games from the thesis corpus "
            f"(analysis/heldout_test_eval/attn_tuned__best.csv on the HPC checkout; seed {SEED}; "
            "up to 700 games each for bullet, blitz and rapid, every available classical and "
            "ultrabullet game; months with unpacked game files only; at least 20 plies). "
            f"{games} games ({len(overall)} sides) scored on CPU with the prototype's own "
            "src/detector.py and the frozen rating checkpoint, each side as the suspect against "
            "its PGN-header rating. The rating model never trained on these games, and the "
            "detector trained only on synthetic games."
        ),
        "note": (
            "These cutoffs describe how the detector score is distributed across ordinary, "
            "finished, rated human games in the held-out test set. They are not a cheat-detection "
            "threshold: the detector reached ROC-AUC about 0.75 on synthetic games from rating "
            "bands withheld from training, but only about 0.58 and 0.61 when 2 and 5 percent of "
            "moves were engine moves."
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
    out = sys.argv[2]
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("overall", "by_time_control")}, indent=1))


main()

"""Turn per_side_scores.csv (all four methods per side) into the combined
src/static/suspicion_cutoffs.json: p75/p95 with bootstrap 95% CIs, overall and
per time control, one entry per method. Same percentile, bootstrap, seed and
n-fallback method as experiments/detector_cutoffs/cutoffs_from_scores.py.

  python cutoffs_from_scores.py <repo root> <output json>
"""
import csv, hashlib, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260915
N_BOOTSTRAP = 2000
MIN_SIDES = 300
PERCENTILES = (75, 95)
ROOT = Path(sys.argv[1])
CHECKPOINT = {"checkpoint": "models/preflight_check_2m/best_model.pth", "checkpoint_md5": "fcb38b8a6a3ee04a6b1859dd5bc69b75"}

SAMPLE = (
    "Stratified sample of held-out TEST-partition games from the thesis corpus "
    "(analysis/heldout_test_eval/attn_tuned__best.csv on the HPC checkout; seed 20260915; up to 700 "
    "games each for bullet, blitz and rapid, every available classical and ultrabullet game; months "
    "with unpacked game files only; at least 20 plies): the same {games} games ({sides} sides) for all "
    "four methods, reused from the per-move detector's cutoffs run rather than re-sampled. Scored on "
    "CPU with the prototype's own code and the frozen rating checkpoint, each side as the suspect "
    "against its PGN-header rating. The rating model never trained on these games, and the trained "
    "detectors trained only on synthetic games."
)


def md5(path):
    return hashlib.md5((ROOT / path).read_bytes()).hexdigest()


METHODS = {
    "s_att": {
        "score_description": "Computed score S_att (first method tried): attention-weighted mean absolute gap, in rating points, between the rating model's per-move estimate and the side's baseline. No trained parameters.",
        "note": "These cutoffs describe how S_att is distributed across ordinary, finished, rated human games in the held-out test set. They are not a cheat-detection threshold: S_att separated engine-substituted synthetic games at about chance (ROC-AUC about 0.51 on rating bands withheld from training).",
        "files": {},
    },
    "lgbm_a0g": {
        "score_description": "Trained LightGBM detector (thesis arm A0g): game-level output in [0, 1] per side from 97 pooled per-game features.",
        "note": "These cutoffs describe how the LightGBM detector score is distributed across ordinary, finished, rated human games in the held-out test set. They are not a cheat-detection threshold: on synthetic games from rating bands withheld from training it reached ROC-AUC about 0.69, and about 0.54 and 0.57 when 2 and 5 percent of moves were engine moves.",
        "files": {"detector_model": "src/models/detector_lgbm_a0g.txt"},
    },
    "detector_a3g_seed0": {
        "score_description": "Trained per-move detector (thesis arm A3g, seed 0): game-level output in [0, 1] per side.",
        "note": "These cutoffs describe how the detector score is distributed across ordinary, finished, rated human games in the held-out test set. They are not a cheat-detection threshold: the detector reached ROC-AUC about 0.75 on synthetic games from rating bands withheld from training, but only about 0.58 and 0.61 when 2 and 5 percent of moves were engine moves.",
        "files": {"detector_weights": "src/models/detector_a3g_seed0.pt"},
    },
    "cnn_bilstm_a4": {
        "score_description": "Trained full CNN-BiLSTM detector (thesis arm A4): game-level output in [0, 1] per side, read from the board positions and clocks.",
        "note": "These cutoffs describe how the CNN-BiLSTM detector score is distributed across ordinary, finished, rated human games in the held-out test set. They are not a cheat-detection threshold: on synthetic games from rating bands withheld from training it reached ROC-AUC about 0.70, and about 0.55 and 0.57 when 2 and 5 percent of moves were engine moves.",
        "files": {"detector_weights": "src/models/detector_cnn_bilstm_a4.pt"},
    },
}


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


def summarize(values, rng, digits):
    sv = sorted(values); out = {"n_sides": len(values)}
    for pct in PERCENTILES:
        lo, hi = bootstrap_ci(values, pct, rng)
        out[f"p{pct}"] = round(percentile(sv, pct), digits)
        out[f"p{pct}_ci"] = [round(lo, digits), round(hi, digits)]
    return out


def main():
    with (HERE / "per_side_scores.csv").open() as f:
        rows = list(csv.DictReader(f))
    result = {
        "schema_note": "One entry per suspicion-score method, keyed by method id. Each entry names its own score in \"score\"; api._load_method_cutoffs ignores an entry whose score does not match its key.",
        "generated_date": time.strftime("%Y-%m-%d"),
        "methods": {},
    }
    for method, meta in METHODS.items():
        rng = random.Random(SEED)  # same resampling stream per method, as in the single-method runs
        digits = 2 if method == "s_att" else 4
        by_tc, overall = {}, []
        for row in rows:
            vals = [float(row[f"white_{method}"]), float(row[f"black_{method}"])]
            by_tc.setdefault(row["time_control"], []).extend(vals)
            overall.extend(vals)
        entry = {"score": method, "score_description": meta["score_description"],
                 "generated_date": result["generated_date"]}
        for key, path in meta["files"].items():
            entry[key] = path
            entry[f"{key}_md5"] = md5(path)
        entry.update(CHECKPOINT)
        entry.update({
            "provisional": False,
            "source_description": SAMPLE.format(games=len(rows), sides=len(overall)),
            "note": meta["note"],
            "bootstrap_resamples": N_BOOTSTRAP,
            "min_sides_for_time_control_cutoff": MIN_SIDES,
            "overall": summarize(overall, rng, digits),
            "by_time_control": {},
        })
        for tc, values in sorted(by_tc.items()):
            if len(values) < MIN_SIDES:
                entry["by_time_control"][tc] = {"n_sides": len(values), "use_overall": True,
                                                "reason": f"fewer than {MIN_SIDES} sides"}
            else:
                e = summarize(values, rng, digits); e["use_overall"] = False
                entry["by_time_control"][tc] = e
        result["methods"][method] = entry
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    for m, e in result["methods"].items():
        print(m, e["overall"], {tc: (v.get("p75"), v.get("p95")) for tc, v in e["by_time_control"].items()})


main()

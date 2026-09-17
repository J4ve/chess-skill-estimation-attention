"""Write tests/regression/expected_sample_scores.json, the browser regression
suite's expected score and label for every bundled sample under every method (local).

Trained methods' expected scores are their parity-checked HPC code path values
(hpc_scores.json); this script asserts the app still agrees to 1e-4 before
writing. S_att had no separate HPC port to check against, so its expected
score is the app's own value at the time of writing. Labels are the app's,
against src/static/suspicion_cutoffs.json. Needs RATINGNET_CHECKPOINT.

Re-run after any deliberate model or cutoffs change, and review the diff.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import api  # noqa: E402

api.MODEL, api.MODEL_PARAMS, api.DEVICE, api.CHECKPOINT_PATH = api._load_model()
api.DETECTOR = api._load_detector()
api.LGBM_DETECTOR = api._load_lgbm_detector()
api.CNN_BILSTM_DETECTOR = api._load_cnn_bilstm_detector()
api.METHOD_CUTOFFS = api._load_method_cutoffs(api.SUSPICION_CUTOFFS_PATH)

manifest = json.load(open(os.path.join(ROOT, "src/static/samples/manifest.json")))
hpc = json.load(open(os.path.join(HERE, "hpc_scores.json")))["hpc"]
gids = {(e["source"], e["side"]): e["gid"] for e in json.load(open(os.path.join(HERE, "index.json")))}

out = {"note": __doc__.strip().splitlines()[0], "tolerance": 1e-4, "samples": []}
for sample in manifest["samples"]:
    pgn_file = os.path.basename(sample["pgn_path"])
    pgn = open(os.path.join(ROOT, "src/static", sample["pgn_path"])).read()
    result = api._run_inference(pgn, None, None, api.DEFAULT_TOP_K)
    methods = {}
    for method_id, entry in result["suspicion_methods"].items():
        assert entry["available"], (sample["id"], method_id)
        m = {"scale": entry["scale"]}
        for side in ("white", "black"):
            app_score = entry[f"{side}_score"]
            if method_id == api.COMPUTED_SCORE_ID:
                m[f"{side}_score"], m["source"] = app_score, "app"
            else:
                expected = hpc[gids[(pgn_file, side)]][method_id]
                assert abs(expected - app_score) < 1e-4, (sample["id"], method_id, side, expected, app_score)
                m[f"{side}_score"], m["source"] = round(expected, 6), "hpc_code_path"
            m[f"{side}_label"] = entry[f"{side}_label"]
        methods[method_id] = m
    out["samples"].append({"id": sample["id"], "title": sample.get("title"), "pgn_path": sample["pgn_path"],
                           "time_control_bucket": (result["suspicion_methods"]["s_att"]["cutoffs_used"] or {}).get("time_control"),
                           "methods": methods})
path = os.path.join(ROOT, "tests/regression/expected_sample_scores.json")
json.dump(out, open(path, "w"), indent=1)
open(path, "a").write("\n")
print(len(out["samples"]), "samples written")

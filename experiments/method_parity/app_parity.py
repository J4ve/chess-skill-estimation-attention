"""App side of the LightGBM (A0g) and CNN-BiLSTM (A4) parity checks (local).

Runs every kit PGN through api._run_inference and compares each method's
per-side score with hpc_score.py's thesis code path (and, for corpus games,
the original thesis GPU runs). Run from this directory with index.json,
hpc_scores.json and corpus_pgn/ copied from the HPC scratch directory, and
RATINGNET_CHECKPOINT set. Writes parity_rows.json and prints markdown tables.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import api  # noqa: E402

api.MODEL, api.MODEL_PARAMS, api.DEVICE, api.CHECKPOINT_PATH = api._load_model()
api.DETECTOR = api._load_detector()
api.LGBM_DETECTOR = api._load_lgbm_detector()
api.CNN_BILSTM_DETECTOR = api._load_cnn_bilstm_detector()
SAMPLES = os.path.join(HERE, "..", "..", "src", "static", "samples")

idx = json.load(open("index.json"))
hpc = json.load(open("hpc_scores.json"))
cache, rows = {}, []
for e in idx:
    path = os.path.join("corpus_pgn" if e["kind"] == "corpus" else SAMPLES, e["source"])
    if path not in cache:
        cache[path] = api._run_inference(open(path).read(), None, None, 5)
    methods = cache[path]["suspicion_methods"]
    for method in ("lgbm_a0g", "cnn_bilstm_a4", "detector_a3g_seed0"):
        h = hpc["hpc"][e["gid"]][method]
        a = methods[method][f"{e['side']}_score"]
        t = hpc["thesis"].get(e["gid"], {}).get(method)
        rows.append({"method": method, "kind": e["kind"], "game": e["source"].replace(".pgn", ""), "side": e["side"],
                     "hpc": h, "app": a, "abs_diff": abs(h - a), "thesis_gpu": t})
json.dump(rows, open("parity_rows.json", "w"), indent=1)

for method in ("lgbm_a0g", "cnn_bilstm_a4", "detector_a3g_seed0"):
    mr = [r for r in rows if r["method"] == method]
    print(f"\n### {method}\n")
    print("| kind | game | suspect side | HPC code path | app code path | abs diff | thesis run (GPU) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in mr:
        t = "" if r["thesis_gpu"] is None else f"{r['thesis_gpu']:.6f}"
        print(f"| {r['kind']} | {r['game']} | {r['side']} | {r['hpc']:.6f} | {r['app']:.6f} | {r['abs_diff']:.1e} | {t} |")
    mx = max(r["abs_diff"] for r in mr)
    mt = max((abs(r["hpc"] - r["thesis_gpu"]) for r in mr if r["thesis_gpu"] is not None), default=float("nan"))
    print(f"\nmax |HPC - app| = {mx:.2e} over {len(mr)} side-scores; max |HPC CPU - thesis GPU| = {mt:.2e}")
bad = [r for r in rows if r["abs_diff"] >= 1e-4]
assert not bad, bad

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import api, detector
api.MODEL, api.MODEL_PARAMS, api.DEVICE, api.CHECKPOINT_PATH = api._load_model()
api.DETECTOR = detector.load_detector()
idx = json.load(open("index.json")); hpc = json.load(open("hpc_scores.json"))
cache = {}
rows = []
for e in idx:
    src = e["source"]
    path = os.path.join("corpus_pgn" if e["kind"] == "corpus" else ".", src)
    if path not in cache:
        cache[path] = api._run_inference(open(path).read(), None, None, 5)
    r = cache[path]
    app = r[f"{e['side']}_suspicion_score"]
    assert r["suspicion_score_kind"] == "detector"
    h = hpc[e["gid"]]
    rows.append((e["kind"], src.replace(".pgn", ""), e["side"], h, app, abs(h - app), e.get("thesis_score")))
print(f"| kind | game | suspect side | HPC code path | app code path | abs diff | thesis run (GPU) |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for k, s, side, h, a, d, t in rows:
    print(f"| {k} | {s} | {side} | {h:.6f} | {a:.6f} | {d:.1e} | {'' if t is None else f'{t:.6f}'} |")
mx = max(r[5] for r in rows); mt = max(abs(r[3]-r[6]) for r in rows if r[6] is not None)
print(f"\nmax |HPC - app| = {mx:.2e} over {len(rows)} side-scores; max |HPC CPU - thesis GPU| = {mt:.2e}")
json.dump(rows, open("parity_rows.json", "w"))
assert mx < 1e-4

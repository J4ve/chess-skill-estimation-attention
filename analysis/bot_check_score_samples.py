"""Score the prototype's bundled sample PGNs through its own /predict/pgn path."""
import glob, json, os, sys
sys.path.insert(0, "proto/src")  # a copy of prototype/src at submodule 4c7d7c5
os.environ["RATINGNET_CHECKPOINT"] = os.path.expanduser("~/Bacsain/thesis2/models/preflight_check_2m/best_model.pth")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from fastapi.testclient import TestClient
import api
out = {}
with TestClient(api.app) as c:
    for f in sorted(glob.glob("proto/src/static/samples/*.pgn")):
        r = c.post("/predict/pgn", json={"pgn": open(f).read()}).json()
        h = r["headers"]
        out[os.path.basename(f)] = {k: r.get(k) for k in ["white_actual_rating", "black_actual_rating", "white_baseline", "black_baseline",
            "white_baseline_source", "black_baseline_source", "white_final_rating", "black_final_rating", "white_suspicion_score",
            "black_suspicion_score", "white_suspicion_label", "black_suspicion_label", "suspicion_cutoffs_used"]} | {"TimeControl": h.get("TimeControl"), "n_ply": len(r["per_move"])}
json.dump(out, open("sample_scores.json", "w"), indent=1)
for k, v in out.items():
    print(k, v["TimeControl"], v["white_actual_rating"], v["black_actual_rating"], v["white_final_rating"], v["black_final_rating"],
          v["white_suspicion_score"], v["black_suspicion_score"], v["white_suspicion_label"], v["black_suspicion_label"])
print(json.dumps(next(iter(out.values()))["suspicion_cutoffs_used"]))

"""Score the parity kit with the thesis code paths (HPC, CPU) and write hpc_scores.json.

A0g: extra_arms_common.pooled_features (train_anomaly_detector.pool_game) and the
re-fit booster from a0g_refit.py (which reproduces the thesis A0g scores exactly).
A4:  extra_arm_a4 stage_cache (frozen CNN trunk, float16 cache) + load_cache +
     make_model + Batcher + predict with analysis/extra_arms/A4/model_best.pt.
A3g: experiments/detector_parity/hpc_score.py's path, for the regression suite.
Also records each corpus game's score from the original thesis GPU runs.
"""
import json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "analysis/scripts"))
OUT = os.path.join(T, "scratch/arm-selector/parity")
import extra_arms_common as C
C.FEAT_DIR = os.path.join(OUT, "feat")
C.CORPUS_DIR = os.path.join(OUT, "corpus")
C.OUT_ROOT = os.path.join(OUT, "arms")  # caches land here, never in analysis/
import extra_arm_a4 as A4
import extra_arm_a3 as A3
import lightgbm as lgb

torch.set_num_threads(8)
data = C.Data()
res = {}

# --- A0g ---
X, keys, _, _, _ = C.pooled_features(data, smoke=False)
train_keys = json.load(open(os.path.join(T, "scratch/arm-selector/a0g/feature_keys.json")))
assert keys == train_keys, "pooled feature keys differ from training"
booster = lgb.Booster(model_file=os.path.join(T, "scratch/arm-selector/a0g/model.txt"))
a0g = booster.predict(X)

# --- A4 ---
A4.stage_cache(smoke=False, workers=2)
E, K, offs = A4.load_cache(data, smoke=False)
dev = torch.device("cpu")
model, _, _ = A4.make_model(dev)
model.load_state_dict(torch.load(os.path.join(T, "analysis/extra_arms/A4/model_best.pt"), map_location="cpu"))
bat = A4.Batcher(E, K, offs, data)
a4, _, _, _ = A4.predict(model, bat, np.arange(data.n), dev)

# --- A3g ---
M = A3.build_ply_matrix(data)
zs = np.load(os.path.join(T, "analysis/extra_arms/A3g_seed0/scores.npz"), allow_pickle=True)
M = (M - zs["feat_mu"]) / zs["feat_sd"]; M[~np.isfinite(M)] = 0.0
m3 = A3.make_model(M.shape[1], 128, 0.3)
m3.load_state_dict(torch.load(os.path.join(T, "analysis/extra_arms/A3g_seed0/model.pt"), map_location="cpu"))
a3g, _, _, _ = A3.predict(m3, A3.Batcher(M, data, dev), np.arange(data.n))

for i in range(data.n):
    res[data.gid[i]] = {"lgbm_a0g": float(a0g[i]), "cnn_bilstm_a4": float(a4[i]), "detector_a3g_seed0": float(a3g[i])}

# thesis GPU-run scores for the corpus games
index = json.load(open(os.path.join(OUT, "index.json")))
thesis = {}
for arm, key in (("A0g", "lgbm_a0g"), ("A4", "cnn_bilstm_a4"), ("A3g_seed0", "detector_a3g_seed0")):
    z = np.load(os.path.join(T, f"analysis/extra_arms/{arm}/scores.npz"), allow_pickle=True)
    pos = {g: j for j, g in enumerate(z["gid"].astype(str))}
    for e in index:
        if e["kind"] == "corpus":
            thesis.setdefault(e["gid"], {})[key] = float(z["score"][pos[e["orig_gid"]]])
json.dump({"hpc": res, "thesis": thesis}, open(os.path.join(OUT, "hpc_scores.json"), "w"), indent=1)
print(len(res), "scored")

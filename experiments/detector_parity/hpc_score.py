"""Score the fake corpus with the thesis code path (CPU) and write hpc_scores.json."""
import json, os, sys
import numpy as np
import torch
T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "analysis/scripts"))
OUT = os.path.join(T, "scratch/detector-default/parity")
import extra_arms_common as C
C.FEAT_DIR = os.path.join(OUT, "feat")
import extra_arm_a3 as A3
data = C.Data()
M = A3.build_ply_matrix(data)
zs = np.load(os.path.join(T, "analysis/extra_arms/A3g_seed0/scores.npz"), allow_pickle=True)
mu, sd = zs["feat_mu"], zs["feat_sd"]
M = (M - mu) / sd
M[~np.isfinite(M)] = 0.0
model = A3.make_model(M.shape[1], 128, 0.3)
model.load_state_dict(torch.load(os.path.join(T, "analysis/extra_arms/A3g_seed0/model.pt"), map_location="cpu"))
dev = torch.device("cpu")
bat = A3.Batcher(M, data, dev)
g, _, _, _ = A3.predict(model, bat, np.arange(data.n))
res = {data.gid[i]: float(g[i]) for i in range(data.n)}
json.dump(res, open(os.path.join(OUT, "hpc_scores.json"), "w"), indent=1)
print(len(res), "scored")

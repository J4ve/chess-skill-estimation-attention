"""Why does one corpus game's A4 score differ between the CPU parity run and the thesis GPU run?"""
import json, os, sys
import numpy as np
import torch
T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "analysis/scripts"))
OUT = os.path.join(T, "scratch/arm-selector/parity")
import extra_arms_common as C
C.FEAT_DIR = os.path.join(OUT, "feat"); C.CORPUS_DIR = os.path.join(OUT, "corpus"); C.OUT_ROOT = os.path.join(OUT, "arms")
import extra_arm_a4 as A4
data = C.Data()
index = {e["gid"]: e for e in json.load(open(os.path.join(OUT, "index.json")))}
E, K, offs = A4.load_cache(data, False)
res = {}
for i in range(data.n):
    e = index[data.gid[i]]
    if e["kind"] != "corpus":
        continue
    cell, fn = e["orig_gid"].split("/")
    z = np.load(os.path.join(T, "analysis/extra_arms/A4_cache", f"{cell}.npz"))
    g = z["gid"].astype(str); j = int(np.where(g == e["orig_gid"])[0][0])
    start = int(np.concatenate([[0], np.cumsum(z["nply"])])[j]); n = int(z["nply"][j])
    thesis_emb = z["emb"][start:start + n].astype(np.float32); cpu_emb = E[offs[i]:offs[i] + n].astype(np.float32)
    res[e["orig_gid"]] = {"n": n, "max_emb_diff": float(np.abs(thesis_emb - cpu_emb).max()),
                          "max_clk_diff": float(np.abs(z["clk"][start:start + n] - K[offs[i]:offs[i] + n]).max())}
    E[offs[i]:offs[i] + n] = z["emb"][start:start + n]  # swap in the thesis GPU embeddings
for name, dev in (("cpu_with_thesis_emb", torch.device("cpu")), ("gpu_with_thesis_emb", torch.device("cuda"))):
    model, _, _ = A4.make_model(dev)
    model.load_state_dict(torch.load(os.path.join(T, "analysis/extra_arms/A4/model_best.pt"), map_location=dev))
    bat = A4.Batcher(E, K, offs, data)
    idx = np.array([i for i in range(data.n) if index[data.gid[i]]["kind"] == "corpus"])
    s, _, _, _ = A4.predict(model, bat, idx, dev)
    for i, v in zip(idx, s):
        res[index[data.gid[i]]["orig_gid"]][name] = float(v)
print(json.dumps(res, indent=1))

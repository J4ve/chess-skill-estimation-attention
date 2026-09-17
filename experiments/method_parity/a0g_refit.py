"""Re-fit arm A0g's LightGBM detector (HPC) and save the booster.

The thesis run (analysis/extra_arms/A0g) kept only its scores, not the model.
This re-runs extra_arm_a0.fit_lgbm unchanged, on the same cached pooled
features and grouped split, and checks the re-fit reproduces the stored
A0g scores.npz before saving the model, so the saved booster is the thesis
model rather than a look-alike. Nothing under analysis/ is written.
"""
import json, os, sys
import numpy as np

T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "analysis/scripts"))
import extra_arms_common as C  # noqa: E402
import extra_arm_a0 as A0  # noqa: E402

OUT = os.path.join(T, "scratch/arm-selector/a0g")
os.makedirs(OUT, exist_ok=True)

data = C.Data()
split = data.split("grouped")
cache = os.path.join(C.OUT_ROOT, "cache", "pooled_features.npz")
z = np.load(cache, allow_pickle=True)
assert np.array_equal(z["gid"].astype(str), data.gid), "pooled feature cache does not match corpus order"
X, keys = z["X"], list(z["keys"].astype(str))
clf = A0.fit_lgbm(X, data.y, split, C.SPLIT_SEED, smoke=False)
score = clf.predict_proba(X)[:, 1]

ref = np.load(os.path.join(C.OUT_ROOT, "A0g", "scores.npz"), allow_pickle=True)
assert np.array_equal(ref["gid"].astype(str), data.gid)
assert np.array_equal(ref["split"].astype(str), split.astype(str))
diff = np.abs(score - ref["score"])
report = {"best_iteration": int(clf.best_iteration_), "n_features": len(keys),
          "max_abs_diff_vs_thesis_scores": float(diff.max()), "mean_abs_diff": float(diff.mean()),
          "n_games": int(len(score))}
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(OUT, "refit_report.json"), "w"), indent=1)
clf.booster_.save_model(os.path.join(OUT, "model.txt"), num_iteration=clf.best_iteration_)
json.dump(clf.booster_.dump_model(num_iteration=clf.best_iteration_), open(os.path.join(OUT, "model_dump.json"), "w"))
json.dump(keys, open(os.path.join(OUT, "feature_keys.json"), "w"))
print("saved")

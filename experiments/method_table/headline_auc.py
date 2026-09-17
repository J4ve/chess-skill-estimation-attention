"""Headline AUCs for the suspicion-method table (HPC, read-only): S_att on the grouped split
(from A0g scores.npz, same conditions and bootstrap as extra_arms_common), plus the stored
A0g and A4 results for cross-checking the model provenance files. Prints JSON after log lines."""
import json, os, sys
import numpy as np
T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "analysis/scripts"))
import extra_arms_common as C
data = C.Data()
KEYS = ("seen", "withheld_band", "withheld_rate_vs_clean", "withheld_both",
        "withheld_band_rate02_vs_clean", "withheld_band_rate05_vs_clean", "withheld_band_rate20_vs_clean",
        "withheld_band_rate40_vs_clean", "withheld_band_rate60_vs_clean")
out = {}
z = np.load(os.path.join(C.OUT_ROOT, "A0g/scores.npz"), allow_pickle=True)
assert np.array_equal(z["gid"].astype(str), data.gid)
split = z["split"].astype(str)
conds = C.conditions(data, split)
out["s_att_grouped"] = {k: C.auc_ci(data.y[conds[k]], z["s_att"][conds[k]], k, 1000) for k in KEYS}
for k in KEYS: out["s_att_grouped"][k].pop("n_pos"); out["s_att_grouped"][k].pop("n_neg")
sm, sy = C.strength_mask(data, split)
out["s_att_grouped"]["strength_check"] = C.auc(sy[sm], z["s_att"][sm])
for arm, path in (("A0g", ("variants", "A0g")), ("A4", ("game_level",)), ("A3g_seed0", None)):
    r = json.load(open(os.path.join(C.OUT_ROOT, arm, "results.json")))
    if path is None:
        continue
    g = r
    for p in path: g = g[p]
    out[arm] = {k: {"auc": g["conditions"][k]["auc"], "ci95": g["conditions"][k].get("ci95"), "n": g["conditions"][k]["n"]} for k in KEYS}
    out[arm]["strength_check"] = g["strength_check"]["auc"]
    out[arm]["split"] = r["split"]
    if arm == "A4":
        out[arm]["localization_instance_logit_withheld_band"] = {k: r["localization_instance_logit"]["withheld_band"][k] for k in ("hit1", "rand_hit1", "ply_auc")}
        out[arm]["best_val_auc"] = r["best_val_auc"]
    if arm == "A0g":
        out[arm]["best_iteration"] = r["best_iteration"]
print(json.dumps(out, indent=1))

"""A0 anchor (and A0g grouped-split sensitivity): the v2 LightGBM detector, unchanged.

Plan: analysis/anomaly-extra-arms-plan.md section 5 (A0). With --split v2 it
also writes the reproduction gate marker A0_GATE_PASS or A0_GATE_FAIL.

  python extra_arm_a0.py --split v2        # A0
  python extra_arm_a0.py --split grouped   # A0g
  python extra_arm_a0.py --smoke           # tiny subset, pipeline check only
"""
import argparse
import os

import numpy as np

import extra_arms_common as C


def fit_lgbm(X, y, split, seed, smoke):
    import lightgbm as lgb
    tr = split == "train"; va = split == "val"
    npos, nneg = int(y[tr].sum()), int((y[tr] == 0).sum())
    clf = lgb.LGBMClassifier(
        n_estimators=(50 if smoke else 2000), learning_rate=0.02, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_samples=30, scale_pos_weight=nneg / max(npos, 1),
        random_state=seed, n_jobs=4, verbose=-1)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return clf


def lgbm_arm(arm, split_kind, smoke, transform=None, extra_scores=None):
    """Shared by A0/A0g and A2: pooled features -> LightGBM -> reports.

    transform(X, keys, data, split) -> dict variant -> X' lets A2 swap in
    band-normalized features (one model, several withheld transforms)."""
    def body(d):
        data = C.Data(smoke=smoke)
        split = data.split(split_kind)
        X, keys, s_att, _, _ = C.pooled_features(data, smoke)
        variants = transform(X, keys, data, split) if transform else {arm: X}
        train_variant = next(iter(variants))
        clf = fit_lgbm(variants[train_variant], data.y, split, C.SPLIT_SEED, smoke)
        n_boot = 50 if smoke else C.N_BOOT
        a0 = None
        if arm != "A0" and split_kind == "v2":  # paired diff only on identical splits
            try:
                a0s, a0g = C.load_scores("A0", smoke)
                if np.array_equal(a0g, data.gid):
                    a0 = a0s
            except FileNotFoundError:
                pass
        res = C.base_results(arm, smoke, split_kind, {
            "model": "LightGBM (anchor settings)", "n_features": len(keys), "features": keys,
            "best_iteration": int(clf.best_iteration_ or 0),
            "feature_importance_gain": dict(sorted(zip(keys, clf.booster_.feature_importance("gain").tolist()),
                                                   key=lambda kv: -kv[1])),
            "split_counts": {k: int((split == k).sum()) for k in np.unique(split)},
            "variants": {}})
        save = {"gid": data.gid, "split": split.astype(str), "y": data.y, "s_att": s_att}
        for vname, Xv in variants.items():
            score = clf.predict_proba(Xv)[:, 1]
            save["score" if vname == train_variant else f"score_{vname}"] = score
            res["variants"][vname] = C.game_level_report(data, split, score, n_boot, a0_score=a0)
        if arm == "A0":
            res["S_att"] = C.game_level_report(data, split, s_att, n_boot, a0_score=save["score"])
            res["S_att_localization"] = C.localization_report(data, split, C.s_att_ply_score(data))
        C.atomic_savez(os.path.join(d, "scores.npz"), **save)
        if arm == "A0" and split_kind == "v2":
            head = res["variants"]["A0"]["conditions"]
            got = {k: head[k]["auc"] for k in C.REF_A0}
            ok = all(abs(got[k] - C.REF_A0[k]) <= C.GATE_TOL for k in ("seen", "withheld_band"))
            res["gate"] = {"reference": C.REF_A0, "reproduced": got, "tolerance": C.GATE_TOL,
                           "checked": ["seen", "withheld_band"], "pass": bool(ok)}
            for m in ("A0_GATE_PASS", "A0_GATE_FAIL"):
                if os.path.exists(os.path.join(d, m)):
                    os.remove(os.path.join(d, m))
            if smoke:
                ok = True  # subset cannot reproduce; smoke only checks the pipeline
            with open(os.path.join(d, "A0_GATE_PASS" if ok else "A0_GATE_FAIL"), "w") as f:
                f.write(repr(got) + "\n")
        C.write_json(os.path.join(d, "results.json"), res)
    C.run_arm(arm, smoke, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v2", choices=["v2", "grouped"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.split == "grouped":
        C.require_gate(a.smoke)
    lgbm_arm("A0" if a.split == "v2" else "A0g", a.split, a.smoke)


if __name__ == "__main__":
    main()

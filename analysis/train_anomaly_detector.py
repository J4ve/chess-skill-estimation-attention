"""Train + evaluate the supervised anomaly (cheat) detector and compare it
against the computed suspicion score.

Preliminary / future-work experiment.  Inputs are the .npz files written by
extract_detector_features.py:

  rating_cheap.npz  (+ rating_cheap_meta.json)   per-ply rating + cheap features
  engine.npz        (optional)                    per-ply Stockfish features

Split discipline (game level, seed 42):
  * held-out Maia bands 1300 and 1700  -> never in train/val
  * held-out substitution rate r015    -> never in train/val
  * remaining cells -> 70/15/15 train/val/test, stratified by (band,engine,rate)

Models (LightGBM game-level classifier on pooled per-ply summary stats):
  full        : rating + cheap features (all games)
  plus_engine : rating + cheap + Stockfish features (engine subset only)

Reports game-level ROC-AUC pooled and per band / engine / rate / held-out cell,
directly against the computed score's S_att / S_max / S_mean on the identical
games, plus feature importances and precision @ 1% / 5% FPR (with the
class-prevalence caveat).  Writes detector_results.json.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HELDOUT_BANDS = (1300, 1700)
# v1 corpus used rates {0,5,15,30,60}; v2 uses {0,2,5,10,20,40,60} (no 15).
# 10 is v2's analogous mid-grid choice (sits between light 5% and moderate 20%,
# same relative position as v1's 15 between 5% and 30%).
HELDOUT_RATE = 10
PCTS = (90, 99)


# ----------------------------------------------------------------------------
def load_npz_games(path):
    z = np.load(path, allow_pickle=True)
    gs = z["game_start"]
    n = len(z["game_id"])
    per_game = {}
    per_ply = {}
    for k in z.files:
        arr = z[k]
        if k == "game_start":
            continue
        if len(arr) == n:
            per_game[k] = arr
        else:
            per_ply[k] = arr
    return per_game, per_ply, gs, n


def slc(per_ply, gs, i):
    a, b = int(gs[i]), int(gs[i + 1])
    return {k: v[a:b] for k, v in per_ply.items()}


def _stats(x, prefix, out, pcts=PCTS, with_std=True):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        out[f"{prefix}_mean"] = np.nan
        out[f"{prefix}_max"] = np.nan
        for p in pcts:
            out[f"{prefix}_p{p}"] = np.nan
        if with_std:
            out[f"{prefix}_std"] = np.nan
        return
    out[f"{prefix}_mean"] = float(x.mean())
    out[f"{prefix}_max"] = float(x.max())
    for p in pcts:
        out[f"{prefix}_p{p}"] = float(np.percentile(x, p))
    if with_std:
        out[f"{prefix}_std"] = float(x.std())


# ----------------------------------------------------------------------------
def computed_scores(pp, n_real):
    """S_att / S_max / S_mean exactly as eval_anomaly_v3.aggregations:
    all real plies, suspect-column deviation d_t, attention over all plies."""
    d = np.asarray(pp["d_t"][:n_real], dtype=np.float64)
    a = np.asarray(pp["alpha"][:n_real], dtype=np.float64)
    s = a.sum()
    a = a / s if s > 1e-8 else np.full_like(a, 1.0 / max(len(a), 1))
    s_att = float((a * d).sum())
    s_max = float(d.max()) if d.size else np.nan
    s_mean = float(d.mean()) if d.size else np.nan
    return s_att, s_max, s_mean


def pool_game(pp, n_real, eng_pp=None):
    """Pooled game-level feature vector (dict)."""
    o = {}
    ply = pp["ply_idx"][:n_real].astype(np.float64)
    is_sus = pp["is_suspect_move"][:n_real].astype(bool)
    sub = pp["move_is_substituted"][:n_real].astype(bool)

    def both(name, arr):
        arr = np.asarray(arr, dtype=np.float64)[:n_real]
        _stats(arr, f"{name}_all", o)
        _stats(arr[is_sus], f"{name}_sus", o)

    both("d_t", pp["d_t"])
    ad = np.asarray(pp["alpha"][:n_real], np.float64) * np.asarray(pp["d_t"][:n_real], np.float64)
    both("alphad", ad)
    both("r_hat_suspect", pp["r_hat_suspect"])
    both("run_std", pp["run_std"])
    both("abs_first_diff", np.abs(pp["first_diff"][:n_real]))
    both("abs_second_diff", np.abs(pp["second_diff"][:n_real]))
    both("material_balance", pp["material_balance"])
    both("clock_remaining", pp["clock_remaining"])
    both("clock_delta", pp["clock_delta"])

    o["frac_capture"] = float(np.nanmean(pp["is_capture"][:n_real])) if n_real else np.nan
    o["frac_check"] = float(np.nanmean(pp["is_check"][:n_real])) if n_real else np.nan
    o["n_plies"] = float(n_real)
    o["n_suspect_moves"] = float(is_sus.sum())
    o["r_hat_final"] = float(pp["r_hat_suspect"][n_real - 1]) if n_real else np.nan
    o["run_std_final"] = float(pp["run_std"][n_real - 1]) if n_real else np.nan
    # last-quarter mean deviation (late-game drift)
    q = max(1, n_real // 4)
    o["d_t_lastq_mean"] = float(np.mean(pp["d_t"][n_real - q:n_real]))

    if eng_pp is not None:
        m = min(n_real, len(eng_pp["cp_loss"]))
        cp = np.asarray(eng_pp["cp_loss"][:m], dtype=np.float64)
        t1 = np.asarray(eng_pp["top1_match"][:m], dtype=np.float64)
        t3 = np.asarray(eng_pp["top3_match"][:m], dtype=np.float64)
        ae = np.asarray(eng_pp["abs_eval"][:m], dtype=np.float64)
        sw = np.asarray(eng_pp["eval_swing"][:m], dtype=np.float64)
        sus_m = is_sus[:m]
        _stats(cp, "cp_loss_all", o)
        _stats(cp[sus_m], "cp_loss_sus", o)
        _stats(ae, "abs_eval_all", o)
        _stats(np.abs(sw), "abs_eval_swing_all", o)
        _stats(np.abs(sw[sus_m]), "abs_eval_swing_sus", o)
        for lab, mask in (("all", np.ones(m, bool)), ("sus", sus_m)):
            cpm = cp[mask]
            cpm = cpm[np.isfinite(cpm)]
            if cpm.size:
                o[f"top1_match_{lab}"] = float(np.nanmean(t1[mask]))
                o[f"top3_match_{lab}"] = float(np.nanmean(t3[mask]))
                o[f"frac_cp_le2_{lab}"] = float(np.mean(cpm <= 2.0))
                o[f"frac_cp_le10_{lab}"] = float(np.mean(cpm <= 10.0))
                o[f"frac_cp_ge100_{lab}"] = float(np.mean(cpm >= 100.0))
                o[f"n_cp_le10_{lab}"] = float(np.sum(cpm <= 10.0))
            else:
                for kk in (f"top1_match_{lab}", f"top3_match_{lab}", f"frac_cp_le2_{lab}",
                           f"frac_cp_le10_{lab}", f"frac_cp_ge100_{lab}", f"n_cp_le10_{lab}"):
                    o[kk] = np.nan
        o["has_engine"] = 1.0
    return o


# ----------------------------------------------------------------------------
def make_split(bands, engines, rates, seed=42):
    n = len(bands)
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * n, dtype=object)
    is_hb = np.isin(bands, HELDOUT_BANDS)
    is_hr = rates == HELDOUT_RATE
    split[is_hb & is_hr] = "heldout_both"
    split[is_hb & ~is_hr] = "heldout_band"
    split[~is_hb & is_hr] = "heldout_rate"
    pool = np.where(~is_hb & ~is_hr)[0]
    # stratify by (band,engine,rate)
    key = np.array([f"{bands[i]}_{engines[i]}_{rates[i]}" for i in pool])
    for k in np.unique(key):
        idx = pool[key == k]
        rng.shuffle(idx)
        n_tr = int(round(0.70 * len(idx)))
        n_va = int(round(0.15 * len(idx)))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"
    return split


def auc_safe(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def prec_at_fpr(clean_scores, y_eval, s_eval, fpr):
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    clean_scores = clean_scores[np.isfinite(clean_scores)]
    if clean_scores.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(clean_scores, 1.0 - fpr))
    flag = np.asarray(s_eval) >= thr
    if flag.sum() == 0:
        return float("nan"), thr
    tp = int(np.sum(flag & (np.asarray(y_eval) == 1)))
    fp = int(np.sum(flag & (np.asarray(y_eval) == 0)))
    return (tp / (tp + fp) if (tp + fp) else float("nan")), thr


# ----------------------------------------------------------------------------
def breakdowns(meta, y, score, mask_eval, tag):
    """Per band/engine/rate AUC on the eval subset given by mask_eval.
    Per-rate AUC uses that rate's positives vs all held-out negatives (r00)."""
    out = {}
    idx = np.where(mask_eval)[0]
    yy = y[idx]; ss = score[idx]
    b = meta["band"][idx]; e = meta["engine"][idx]; r = meta["rate"][idx]
    out["pooled"] = auc_safe(yy, ss)
    out["n"] = int(len(idx))
    out["prevalence_sub"] = float(np.mean(yy)) if len(yy) else float("nan")
    for name, vals in (("band", b), ("engine", e), ("rate", r)):
        d = {}
        for v in sorted(set(vals.tolist())):
            m = vals == v
            d[str(v)] = auc_safe(yy[m], ss[m])
        out[f"by_{name}"] = d
    # per-rate vs r00 negatives
    d = {}
    neg = r == 0
    for rt in sorted(set(r.tolist())):
        if rt == 0:
            continue
        m = neg | (r == rt)
        d[str(rt)] = auc_safe(yy[m], ss[m])
    out["by_rate_vs_r00"] = d
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_ply", action="store_true")
    args = ap.parse_args()

    rc_pg, rc_pp, rc_gs, n = load_npz_games(os.path.join(args.feat_dir, "rating_cheap.npz"))
    print(f"rating_cheap: {n} games")
    eng_path = os.path.join(args.feat_dir, "engine.npz")
    have_eng = os.path.exists(eng_path)
    eng_index = {}
    if have_eng:
        eg_pg, eg_pp, eg_gs, en = load_npz_games(eng_path)
        for i, gid in enumerate(eg_pg["game_id"]):
            eng_index[str(gid)] = i
        print(f"engine: {en} games")

    band = rc_pg["band"].astype(int)
    engine = rc_pg["engine"].astype(str)
    rate = rc_pg["rate"].astype(int)
    gid = rc_pg["game_id"].astype(str)
    y = rc_pg["game_label"].astype(int)
    nply = rc_pg["n_plies"].astype(int)

    split = make_split(band, engine, rate, seed=args.seed)
    from collections import Counter
    print("split counts:", dict(Counter(split.tolist())))

    # ---- pooled features + computed scores ----
    feats_full = []
    feats_eng = []
    s_att = np.full(n, np.nan); s_max = np.full(n, np.nan); s_mean = np.full(n, np.nan)
    has_eng = np.zeros(n, bool)
    for i in range(n):
        pp = slc(rc_pp, rc_gs, i)
        nr = int(nply[i])
        sa, sm, sme = computed_scores(pp, nr)
        s_att[i], s_max[i], s_mean[i] = sa, sm, sme
        epp = None
        if have_eng and gid[i] in eng_index:
            epp = slc(eg_pp, eg_gs, eng_index[gid[i]])
            has_eng[i] = True
        feats_full.append(pool_game(pp, nr, eng_pp=None))
        feats_eng.append(pool_game(pp, nr, eng_pp=epp) if epp is not None else None)
        if (i + 1) % 5000 == 0:
            print(f"  pooled {i+1}/{n}")

    full_keys = sorted({k for d in feats_full for k in d})
    Xfull = np.array([[d.get(k, np.nan) for k in full_keys] for d in feats_full], dtype=np.float64)
    eng_keys = sorted({k for d in feats_eng if d for k in d})
    Xeng = np.full((n, len(eng_keys)), np.nan)
    for i, d in enumerate(feats_eng):
        if d:
            Xeng[i] = [d.get(k, np.nan) for k in eng_keys]

    meta = {"band": band, "engine": engine, "rate": rate, "gid": gid}
    results = {"n_games": int(n), "split_counts": {k: int(v) for k, v in Counter(split.tolist()).items()},
               "heldout_bands": list(HELDOUT_BANDS), "heldout_rate": HELDOUT_RATE,
               "models": {}, "computed_score_baseline": {}}

    try:
        import lightgbm as lgb
        HAVE_LGB = True
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        HAVE_LGB = False
    print(f"classifier: {'LightGBM' if HAVE_LGB else 'sklearn GradientBoosting'}")
    results["classifier"] = "lightgbm" if HAVE_LGB else "sklearn_gbc"

    def fit_eval(X, keys, member_mask, name):
        tr = (split == "train") & member_mask
        va = (split == "val") & member_mask
        te = (split == "test") & member_mask
        hb = (split == "heldout_band") & member_mask
        hr = (split == "heldout_rate") & member_mask
        hbth = (split == "heldout_both") & member_mask
        npos, nneg = int(y[tr].sum()), int((y[tr] == 0).sum())
        spw = (nneg / max(npos, 1))
        print(f"\n[{name}] train={tr.sum()} val={va.sum()} test={te.sum()} "
              f"hb={hb.sum()} hr={hr.sum()} hboth={hbth.sum()}  pos/neg train={npos}/{nneg}")
        if HAVE_LGB:
            clf = lgb.LGBMClassifier(
                n_estimators=2000, learning_rate=0.02, num_leaves=31,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, min_child_samples=30, scale_pos_weight=spw,
                random_state=args.seed, n_jobs=4, verbose=-1)
            clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            imp = dict(zip(keys, clf.booster_.feature_importance(importance_type="gain").tolist()))
        else:
            from sklearn.impute import SimpleImputer
            imp_t = SimpleImputer(strategy="median")
            Xtr = imp_t.fit_transform(X[tr])
            clf = GradientBoostingClassifier(n_estimators=400, learning_rate=0.03,
                                             max_depth=3, subsample=0.8, random_state=args.seed)
            clf.fit(Xtr, y[tr])
            X = imp_t.transform(X)  # transform all rows for scoring
            imp = dict(zip(keys, clf.feature_importances_.tolist()))

        score = np.full(n, np.nan)
        ev = member_mask & np.isin(split, ["test", "heldout_band", "heldout_rate", "heldout_both"])
        score[ev] = clf.predict_proba(X[ev])[:, 1]

        m = {"features": keys, "n_features": len(keys),
             "feature_importance_gain": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
             "eval": {}}
        for tag, mk in (("test", te), ("heldout_band", hb), ("heldout_rate", hr),
                        ("heldout_both", hbth),
                        ("test_plus_heldout", te | hb | hr | hbth)):
            if mk.sum() == 0:
                continue
            m["eval"][tag] = breakdowns(meta, y, score, mk, tag)
        # precision @ FPR on test (clean = r00 test negatives)
        clean = score[(te) & (rate == 0)]
        for fpr in (0.01, 0.05):
            p, thr = prec_at_fpr(clean, y[te], score[te], fpr)
            m["eval"].setdefault("test", {}).setdefault("precision_at_fpr", {})[str(fpr)] = {
                "precision": p, "threshold": thr}
        m["eval"]["test"]["note_prevalence"] = (
            "eval split is substituted-heavy by construction: prevalence_sub above; "
            "precision@FPR is optimistic vs a realistic low base rate")
        results["models"][name] = m
        return score

    sc_full = fit_eval(Xfull, full_keys, np.ones(n, bool), "full")
    if have_eng and has_eng.sum() > 200:
        sc_eng = fit_eval(Xeng, eng_keys, has_eng, "plus_engine")

    # ---- computed-score baseline on identical eval masks ----
    for sname, sarr in (("S_att", s_att), ("S_max", s_max), ("S_mean", s_mean)):
        d = {}
        for tag, mk in (("test", split == "test"),
                        ("heldout_band", split == "heldout_band"),
                        ("heldout_rate", split == "heldout_rate"),
                        ("heldout_both", split == "heldout_both"),
                        ("test_plus_heldout", np.isin(split, ["test", "heldout_band", "heldout_rate", "heldout_both"]))):
            if mk.sum() == 0:
                continue
            d[tag] = breakdowns(meta, y, sarr, mk, tag)
        results["computed_score_baseline"][sname] = d

    # ---- optional per-ply model ----
    if args.per_ply and have_eng:
        results["per_ply_model"] = per_ply_model(rc_pp, rc_gs, eg_pp, eg_gs, eng_index,
                                                 gid, nply, y, band, engine, rate, split, meta, args.seed)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {args.out_json}")

    # ---- console summary ----
    print("\n===== SUMMARY: game-level ROC-AUC =====")
    for name, m in results["models"].items():
        print(f"\n[{name}]  (features={m['n_features']})")
        for tag, e in m["eval"].items():
            if isinstance(e, dict) and "pooled" in e:
                print(f"  {tag:20s} pooled AUC={e['pooled']:.4f}  n={e['n']}  prev_sub={e['prevalence_sub']:.3f}")
    print("\n[computed score baseline]")
    for sname, d in results["computed_score_baseline"].items():
        row = " ".join(f"{t}={d[t]['pooled']:.4f}" for t in d if "pooled" in d[t])
        print(f"  {sname:7s} {row}")
    print("\ntop-12 gain features (full):")
    for k, v in list(results["models"]["full"]["feature_importance_gain"].items())[:12]:
        print(f"  {k:28s} {v:.1f}")


def per_ply_model(rc_pp, rc_gs, eg_pp, eg_gs, eng_index, gid, nply, y,
                  band, engine, rate, split, meta, seed):
    """Per-ply LightGBM on suspect moves (engine subset), aggregated to game."""
    try:
        import lightgbm as lgb
    except Exception:
        return {"skipped": "lightgbm unavailable"}
    n = len(gid)
    rows_X, rows_y, rows_g = [], [], []
    feat_names = ["d_t", "alpha", "alphad", "run_std", "abs_d1", "abs_d2",
                  "material", "clock_delta", "clock_rem", "ply_pos", "is_capture", "is_check",
                  "cp_loss", "top1", "top3", "abs_eval", "abs_swing"]
    for i in range(n):
        if gid[i] not in eng_index:
            continue
        a, b = int(rc_gs[i]), int(rc_gs[i + 1])
        nr = int(nply[i])
        ea, eb = int(eg_gs[eng_index[gid[i]]]), int(eg_gs[eng_index[gid[i]] + 1])
        m = min(nr, eb - ea)
        is_sus = rc_pp["is_suspect_move"][a:a + m].astype(bool)
        dd = rc_pp["d_t"][a:a + m].astype(np.float64)
        al = rc_pp["alpha"][a:a + m].astype(np.float64)
        rs = rc_pp["run_std"][a:a + m].astype(np.float64)
        d1 = np.abs(rc_pp["first_diff"][a:a + m].astype(np.float64))
        d2 = np.abs(rc_pp["second_diff"][a:a + m].astype(np.float64))
        mat = rc_pp["material_balance"][a:a + m].astype(np.float64)
        cd = rc_pp["clock_delta"][a:a + m].astype(np.float64)
        cr = rc_pp["clock_remaining"][a:a + m].astype(np.float64)
        cap = rc_pp["is_capture"][a:a + m].astype(np.float64)
        chk = rc_pp["is_check"][a:a + m].astype(np.float64)
        pp_ = rc_pp["ply_idx"][a:a + m].astype(np.float64) / max(nr - 1, 1)
        sub = rc_pp["move_is_substituted"][a:a + m].astype(np.int64)
        cp = eg_pp["cp_loss"][ea:ea + m].astype(np.float64)
        t1 = eg_pp["top1_match"][ea:ea + m].astype(np.float64)
        t3 = eg_pp["top3_match"][ea:ea + m].astype(np.float64)
        ae = eg_pp["abs_eval"][ea:ea + m].astype(np.float64)
        sw = np.abs(eg_pp["eval_swing"][ea:ea + m].astype(np.float64))
        for t in range(m):
            if not is_sus[t]:
                continue
            rows_X.append([dd[t], al[t], al[t]*dd[t], rs[t], d1[t], d2[t], mat[t],
                           cd[t], cr[t], pp_[t], cap[t], chk[t],
                           cp[t], t1[t], t3[t], ae[t], sw[t]])
            rows_y.append(int(sub[t]))
            rows_g.append(i)
    X = np.array(rows_X, dtype=np.float64)
    yy = np.array(rows_y)
    gg = np.array(rows_g)
    spl = split[gg]
    tr = spl == "train"; va = spl == "val"
    npos, nneg = int(yy[tr].sum()), int((yy[tr] == 0).sum())
    clf = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.02, num_leaves=31,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             scale_pos_weight=nneg / max(npos, 1), random_state=seed,
                             n_jobs=4, verbose=-1)
    clf.fit(X[tr], yy[tr], eval_set=[(X[va], yy[va])], eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    ply_auc = auc_safe(yy[~tr & ~va], clf.predict_proba(X[~tr & ~va])[:, 1])
    p = clf.predict_proba(X)[:, 1]
    # aggregate to game
    gscore_mean = np.full(len(y), np.nan)
    gscore_topk = np.full(len(y), np.nan)
    for i in np.unique(gg):
        pv = p[gg == i]
        gscore_mean[i] = float(np.mean(pv))
        gscore_topk[i] = float(np.mean(np.sort(pv)[-5:]))
    out = {"per_ply_auc_heldout_games": ply_auc,
           "feature_names": feat_names,
           "feature_importance_gain": dict(sorted(
               zip(feat_names, clf.booster_.feature_importance("gain").tolist()),
               key=lambda kv: -kv[1])),
           "game_agg": {}}
    for tag, mk in (("test", split == "test"), ("heldout_band", split == "heldout_band"),
                    ("heldout_rate", split == "heldout_rate"), ("heldout_both", split == "heldout_both")):
        mk = mk & np.isfinite(gscore_mean)
        if mk.sum() == 0:
            continue
        out["game_agg"][tag] = {
            "mean_agg_auc": auc_safe(y[mk], gscore_mean[mk]),
            "top5_agg_auc": auc_safe(y[mk], gscore_topk[mk]),
            "n": int(mk.sum())}
    return out


if __name__ == "__main__":
    main()

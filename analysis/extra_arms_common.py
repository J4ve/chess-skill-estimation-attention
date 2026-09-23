"""Shared data loading, splits and metrics for the extra anomaly-detection arms.

Pre-registered in analysis/anomaly-extra-arms-plan.md. Every definition here
(splits, conditions, bootstrap, precision at FPR, localization) follows that
plan; change the plan's section 9 "Deviations" before changing anything here.

Split and pooled features come from train_anomaly_detector.py (the HPC copy
that produced analysis/anomaly-trained-detector-v2-results.json), imported
unchanged so the A0 anchor is the same code path.
"""
import json
import os
import sys
import time
import traceback
import zlib

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_anomaly_detector as tad  # noqa: E402

THESIS = os.path.expanduser("~/Bacsain/thesis2")
FEAT_DIR = os.path.join(THESIS, "analysis/v2_features")
OUT_ROOT = os.path.join(THESIS, "analysis/extra_arms")
SMOKE_ROOT = os.path.join(THESIS, "analysis/extra_arms_smoke")
CORPUS_DIR = os.path.join(THESIS, "data/anomaly_corpus_v2")
CHECKPOINT = os.path.join(THESIS, "models/preflight_check_2m/best_model.pth")

SPLIT_SEED = 42
BOOT_SEED = 20260915
N_BOOT = 1000
HELDOUT_BANDS = tad.HELDOUT_BANDS
HELDOUT_RATE = tad.HELDOUT_RATE
assert HELDOUT_RATE == 10 and tuple(HELDOUT_BANDS) == (1300, 1700)

# published v2 LightGBM figures (anomaly-trained-detector-v2-results.json)
REF_A0 = {"seen": 0.8073178833477713, "withheld_band": 0.6853976664452743,
          "withheld_rate_vs_clean": 0.6849786484857955, "withheld_both": 0.6292162452988597}
GATE_TOL = 0.005
REFERENCE_LEVEL = 0.80  # shown as a reference line only, never a criterion
EVAL_SPLITS = ("test", "heldout_band", "heldout_rate", "heldout_both")
LOC_MIN_PLY = 16  # generator substitution eligibility (ply 17, 0-based 16)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------
# output dir + markers
# ----------------------------------------------------------------------------
def out_dir(arm, smoke):
    d = os.path.join(SMOKE_ROOT if smoke else OUT_ROOT, arm)
    os.makedirs(d, exist_ok=True)
    return d


def run_arm(arm, smoke, body):
    """Run body(out_dir); write DONE, or FAILED with the traceback."""
    d = out_dir(arm, smoke)
    for m in ("DONE", "FAILED"):
        if os.path.exists(os.path.join(d, m)):
            os.remove(os.path.join(d, m))
    with open(os.path.join(d, "STARTED"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S %z\n"))
    try:
        body(d)
    except BaseException:
        tb = traceback.format_exc()
        with open(os.path.join(d, "FAILED"), "w") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S %z\n") + tb)
        print(tb, flush=True)
        sys.exit(1)
    with open(os.path.join(d, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S %z\n"))
    log(f"{arm}: DONE -> {d}")


def require_gate(smoke, wait=True):
    """Downstream arms run only after A0 reproduced (plan section 5)."""
    d = os.path.join(SMOKE_ROOT if smoke else OUT_ROOT, "A0")
    while True:
        if os.path.exists(os.path.join(d, "A0_GATE_PASS")):
            return
        if os.path.exists(os.path.join(d, "A0_GATE_FAIL")):
            raise RuntimeError("A0_GATE_FAIL present: A0 did not reproduce; arm not run")
        if os.path.exists(os.path.join(d, "FAILED")):
            raise RuntimeError("A0 FAILED; arm not run")
        if not wait:
            raise RuntimeError("A0 gate marker missing")
        time.sleep(60)


def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Data:
    """rating_cheap.npz loaded once: per-game meta, flat per-ply arrays."""

    def __init__(self, smoke=False, smoke_frac=0.03):
        path = os.path.join(FEAT_DIR, "rating_cheap.npz")
        log(f"loading {path}")
        pg, pp, gs, n = tad.load_npz_games(path)
        self.full_n = n
        self.band = pg["band"].astype(int)
        self.engine = pg["engine"].astype(str)
        self.rate = pg["rate"].astype(int)
        self.gid = pg["game_id"].astype(str)
        self.y = pg["game_label"].astype(int)
        self.nply = pg["n_plies"].astype(int)
        self.game_index = np.array([int(g.split("game_")[-1].split(".")[0]) for g in self.gid])
        self.pp = pp
        self.gs = gs.astype(np.int64)
        self.sel = np.arange(n)
        if smoke:
            # a few games per cell, deterministic; plies stay addressed via gs
            rng = np.random.default_rng(7)
            cell = np.array([g.split("/")[0] for g in self.gid])
            keep = []
            for c in np.unique(cell):
                idx = np.where(cell == c)[0]
                keep.extend(rng.choice(idx, max(8, int(len(idx) * smoke_frac)), replace=False))
            self.sel = np.sort(np.array(keep))
            for k in ("band", "engine", "rate", "gid", "y", "nply", "game_index"):
                setattr(self, k, getattr(self, k)[self.sel])
        self.n = len(self.sel)
        self.start = self.gs[self.sel]
        log(f"games={self.n} (smoke={smoke})")

    def ply_slice(self, i):
        a = int(self.start[i])
        return {k: v[a:a + int(self.nply[i])] for k, v in self.pp.items()}

    def split(self, kind="v2"):
        if kind == "v2":
            return tad.make_split(self.band, self.engine, self.rate, seed=SPLIT_SEED)
        if kind == "grouped":
            return grouped_split(self.band, self.rate, self.game_index, seed=SPLIT_SEED)
        raise ValueError(kind)


def grouped_split(band, rate, game_index, seed=SPLIT_SEED):
    """Same withheld definitions; 70/15/15 by (band, game index) so no twin
    (same band + index across rate cells) crosses train/val/test."""
    n = len(band)
    split = np.array(["train"] * n, dtype=object)
    is_hb = np.isin(band, HELDOUT_BANDS)
    is_hr = rate == HELDOUT_RATE
    split[is_hb & is_hr] = "heldout_both"
    split[is_hb & ~is_hr] = "heldout_band"
    split[~is_hb & is_hr] = "heldout_rate"
    pool = ~is_hb & ~is_hr
    rng = np.random.default_rng(seed)
    for b in sorted(set(band[pool].tolist())):
        idxs = np.unique(game_index[pool & (band == b)])
        rng.shuffle(idxs)
        n_tr = int(round(0.70 * len(idxs)))
        n_va = int(round(0.15 * len(idxs)))
        assign = {}
        for j, gi in enumerate(idxs):
            assign[int(gi)] = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
        m = np.where(pool & (band == b))[0]
        split[m] = [assign[int(game_index[i])] for i in m]
    return split


def pooled_features(data, smoke):
    """The anchor's 97 pooled features + S_att/S_max/S_mean, cached."""
    cdir = os.path.join(SMOKE_ROOT if smoke else OUT_ROOT, "cache")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, "pooled_features.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if len(z["gid"]) == data.n and np.array_equal(z["gid"].astype(str), data.gid):
            log(f"pooled features from cache {path}")
            return z["X"], list(z["keys"].astype(str)), z["s_att"], z["s_max"], z["s_mean"]
    log("pooling features (anchor pool_game)")
    feats = []
    s_att = np.full(data.n, np.nan); s_max = np.full(data.n, np.nan); s_mean = np.full(data.n, np.nan)
    for i in range(data.n):
        pp = data.ply_slice(i)
        nr = int(data.nply[i])
        s_att[i], s_max[i], s_mean[i] = tad.computed_scores(pp, nr)
        feats.append(tad.pool_game(pp, nr, eng_pp=None))
        if (i + 1) % 20000 == 0:
            log(f"  pooled {i+1}/{data.n}")
    keys = sorted({k for d in feats for k in d})
    X = np.array([[d.get(k, np.nan) for k in keys] for d in feats], dtype=np.float64)
    atomic_savez(path, X=X, keys=np.array(keys), s_att=s_att, s_max=s_max, s_mean=s_mean, gid=data.gid)
    return X, keys, s_att, s_max, s_mean


def load_scores(arm, smoke, key="score"):
    p = os.path.join(SMOKE_ROOT if smoke else OUT_ROOT, arm, "scores.npz")
    z = np.load(p, allow_pickle=True)
    return z[key], z["gid"].astype(str)


# ----------------------------------------------------------------------------
# conditions (plan section 4.1)
# ----------------------------------------------------------------------------
def conditions(data, split):
    r, b = data.rate, data.band
    hb = np.isin(b, HELDOUT_BANDS)
    c = {
        "seen": split == "test",
        "withheld_band": split == "heldout_band",
        "withheld_rate_vs_clean": ((r == HELDOUT_RATE) & np.isin(split, ["heldout_rate", "heldout_both"]))
        | ((r == 0) & np.isin(split, ["test", "heldout_band"])),
        "withheld_both": split == "heldout_both",
        "withheld_rate": split == "heldout_rate",
        "withheld_both_vs_clean": hb & ((r == HELDOUT_RATE) | ((r == 0) & (split == "heldout_band"))),
    }
    for rt in (2, 5, 20, 40, 60):
        c[f"withheld_band_rate{rt:02d}_vs_clean"] = (split == "heldout_band") & ((r == 0) | (r == rt))
    for rt in (2, 5, 20, 40, 60):
        c[f"seen_rate{rt:02d}_vs_clean"] = (split == "test") & ((r == 0) | (r == rt))
    return c


HEADLINE = ("seen", "withheld_band", "withheld_rate_vs_clean", "withheld_both")


def strength_mask(data, split):
    """hard-negative vs rate 0 on test + heldout_band; label = is hard-negative."""
    m = np.isin(split, ["test", "heldout_band"]) & np.isin(data.rate, [-1, 0])
    return m, (data.rate == -1).astype(int)


# ----------------------------------------------------------------------------
# statistics (plan section 4.2)
# ----------------------------------------------------------------------------
def _clean(s):
    s = np.asarray(s, dtype=np.float64).copy()
    bad = ~np.isfinite(s)
    if bad.any():
        s[bad] = np.nanmin(s[~bad]) - 1.0 if (~bad).any() else 0.0
    return s


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _clean(s)))


def boot_indices(y, name, n_boot):
    """Stratified bootstrap indices, deterministic per condition name."""
    rng = np.random.default_rng([BOOT_SEED, zlib.crc32(name.encode())])
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        out.append(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))
    return out


def auc_ci(y, s, name, n_boot, ref=None):
    """AUC with percentile CI; paired difference against ref scores if given."""
    y = np.asarray(y)
    res = {"auc": auc(y, s), "n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    if len(np.unique(y)) < 2:
        return res
    s = _clean(s)
    idx = boot_indices(y, name, n_boot)
    a = np.array([roc_auc_score(y[i], s[i]) for i in idx])
    res["ci95"] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    if ref is not None:
        rs = _clean(ref)
        d = np.array([roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], rs[i]) for i in idx])
        res["diff_vs_A0"] = float(res["auc"] - auc(y, rs))
        res["diff_vs_A0_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return res


def precision_at_fpr(val_neg_scores, y, s, fprs=(0.01, 0.05)):
    out = {}
    v = _clean(val_neg_scores)
    s = _clean(s)
    for f in fprs:
        thr = float(np.quantile(v, 1.0 - f))
        flag = s >= thr
        tp = int(np.sum(flag & (y == 1))); fp = int(np.sum(flag & (y == 0)))
        out[str(f)] = {
            "threshold_from_val": thr,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "tpr": tp / max(int((y == 1).sum()), 1),
            "realized_fpr": fp / max(int((y == 0).sum()), 1),
            "n_flagged": int(flag.sum()),
        }
    return out


def game_level_report(data, split, score, n_boot, a0_score=None, prevalence_note=True):
    """All section 4.1/4.2 metrics for one per-game score vector."""
    conds = conditions(data, split)
    val_neg = score[(split == "val") & (data.y == 0)]
    rep = {"conditions": {}, "strength_check": None}
    for name, m in conds.items():
        if m.sum() == 0:
            continue
        y = data.y[m]
        ref = a0_score[m] if a0_score is not None else None
        nb = n_boot if (name in HEADLINE or "withheld_band_rate" in name or name == "withheld_both_vs_clean") else max(n_boot // 5, 20)
        r = auc_ci(y, score[m], name, nb, ref=ref)
        r["prevalence_pos"] = float(y.mean())
        r["precision_at_fpr"] = precision_at_fpr(val_neg, y, score[m])
        rep["conditions"][name] = r
    sm, sy = strength_mask(data, split)
    if sm.sum() and len(np.unique(sy[sm])) == 2:
        rep["strength_check"] = auc_ci(sy[sm], score[sm], "strength_check", max(n_boot // 5, 20))
        rep["strength_check"]["definition"] = "AUC(hard-negative vs rate 0), test + heldout_band"
    return rep


# ----------------------------------------------------------------------------
# move-level localization (plan section 4.4)
# ----------------------------------------------------------------------------
def localization_report(data, split, ply_score, ply_offsets=None, engine=None):
    """ply_score: flat per-ply array aligned with data.pp (full-corpus offsets)."""
    conds = conditions(data, split)
    sub_all = data.pp["move_is_substituted"]
    sus_all = data.pp["is_suspect_move"]
    ply_all = data.pp["ply_idx"]
    rep = {}
    for name in ("seen", "withheld_band", "withheld_rate", "withheld_both",
                 "withheld_band_rate02_vs_clean", "withheld_band_rate05_vs_clean"):
        m = conds[name] & (data.y == 1)
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        acc = {k: [] for k in ("hit1", "p3", "p5", "rand_hit1", "rand_p3", "rand_p5")}
        ys, ss = [], []
        for i in idx:
            a = int(data.start[i]); b = a + int(data.nply[i])
            cand = (sus_all[a:b] > 0.5) & (ply_all[a:b] >= LOC_MIN_PLY)
            if cand.sum() == 0:
                continue
            sub = sub_all[a:b][cand] > 0.5
            if sub.sum() == 0:
                continue
            sc = np.asarray(ply_score[a:b], dtype=np.float64)[cand]
            order = np.argsort(-sc, kind="stable")
            nc = len(sc); frac = sub.sum() / nc
            acc["hit1"].append(float(sub[order[0]]))
            acc["p3"].append(float(sub[order[:min(3, nc)]].mean()))
            acc["p5"].append(float(sub[order[:min(5, nc)]].mean()))
            acc["rand_hit1"].append(frac); acc["rand_p3"].append(frac); acc["rand_p5"].append(frac)
            ys.append(sub); ss.append(sc)
        if not ys:
            continue
        yy = np.concatenate(ys); sc = np.concatenate(ss)
        rep[name] = {k: float(np.mean(v)) for k, v in acc.items()}
        rep[name]["n_games"] = len(ys)
        rep[name]["ply_auc"] = auc(yy.astype(int), sc)
        rep[name]["ply_auc_random"] = 0.5
    if engine is not None:
        rep["engine_proximity"] = engine_proximity(data, split, ply_score, engine)
    return rep


def load_engine(data):
    path = os.path.join(FEAT_DIR, "engine.npz")
    z = np.load(path, allow_pickle=True)
    egid = z["game_id"].astype(str)
    egs = z["game_start"].astype(np.int64)
    pos = {g: j for j, g in enumerate(egid)}
    return {"start": np.array([egs[pos[g]] for g in data.gid]),
            "analyzed": z["analyzed"], "top1": z["top1_match"]}


def engine_proximity(data, split, ply_score, eng):
    """AUC of per-move score for Stockfish top1_match among non-substituted
    analysed candidate plies of evaluation games (probe only, never an input)."""
    ev = np.isin(split, EVAL_SPLITS)
    ys, ss = [], []
    for i in np.where(ev)[0]:
        a = int(data.start[i]); n = int(data.nply[i]); ea = int(eng["start"][i])
        an = eng["analyzed"][ea:ea + n] > 0.5
        keep = an & (data.pp["move_is_substituted"][a:a + n] < 0.5) & (data.pp["is_suspect_move"][a:a + n] > 0.5) \
            & (data.pp["ply_idx"][a:a + n] >= LOC_MIN_PLY)
        t1 = eng["top1"][ea:ea + n]
        keep &= np.isfinite(t1)
        if keep.sum():
            ys.append(t1[keep] > 0.5); ss.append(np.asarray(ply_score[a:a + n], np.float64)[keep])
    if not ys:
        return None
    yy = np.concatenate(ys).astype(int); sc = np.concatenate(ss)
    return {"auc_top1_match": auc(yy, sc), "n_plies": int(len(yy)), "frac_top1": float(yy.mean()),
            "definition": "non-substituted analysed suspect plies (ply>=16) of test+withheld games"}


def s_att_ply_score(data):
    """Reference per-move score: normalized attention times deviation."""
    return (data.pp["alpha"].astype(np.float64) * data.pp["d_t"].astype(np.float64))


def base_results(arm, smoke, split_kind, extra=None):
    r = {"arm": arm, "smoke": bool(smoke), "split": split_kind,
         "plan": "analysis/anomaly-extra-arms-plan.md",
         "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
         "reference_level": REFERENCE_LEVEL,
         "reference_level_note": "reference line only, not a target or criterion",
         "deviations": []}
    if extra:
        r.update(extra)
    return r

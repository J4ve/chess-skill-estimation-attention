"""A1 player-level aggregation (no training).

Answers a DIFFERENT question from every other arm: "is this player cheating",
not "is this game substituted". Plan: analysis/anomaly-extra-arms-plan.md
section 5 (A1).

A player = k games drawn without replacement from one cell's evaluation pool
(k = 1, 5, 10, 20). Cheater players come from one (band, engine, rate>0) cell;
clean players from one band's rate 0 games, deduplicated by (band, game index)
because the lc0 and stockfish16 rate 0 cells hold identical games. Per-game
scores (S_att, A0 probability) aggregated by mean and by max. CI: hierarchical
bootstrap (resample games within each pool, redraw players).

  python extra_arm_a1.py [--smoke]
"""
import argparse
import os

import numpy as np
from sklearn.metrics import roc_auc_score

import extra_arms_common as C

KS = (1, 5, 10, 20)
P_CHEAT = 500
P_CLEAN = 1000
N_HBOOT = 200
RATES = (2, 5, 20, 40, 60)


def pools(data, split):
    """condition -> {"cheat": [(cell_key, rate, idx)], "clean": [(band, idx)], "hardneg": [...]}"""
    b, e, r, gi = data.band, data.engine, data.rate, data.game_index
    hb = np.isin(b, C.HELDOUT_BANDS)

    def clean_pool(mask):
        out = []
        for band in sorted(set(b[mask].tolist())):
            m = np.where(mask & (b == band))[0]
            _, first = np.unique(gi[m], return_index=True)  # dedupe identical r00 copies
            out.append((band, m[np.sort(first)]))
        return out

    def cheat_pool(mask, rates):
        out = []
        for band in sorted(set(b[mask].tolist())):
            for eng in ("lc0", "stockfish16"):
                for rt in rates:
                    m = np.where(mask & (b == band) & (e == eng) & (r == rt))[0]
                    if len(m):
                        out.append((f"{band}_{eng}_r{rt:02d}", rt, m))
        return out

    def hn_pool(mask):
        return [(band, np.where(mask & (b == band) & (r == -1))[0])
                for band in sorted(set(b[mask & (r == -1)].tolist()))]

    test = split == "test"
    hbs = split == "heldout_band"
    return {
        "seen": {"cheat": cheat_pool(test, RATES), "clean": clean_pool(test & (r == 0)),
                 "hardneg": hn_pool(test)},
        "withheld_band": {"cheat": cheat_pool(hbs, RATES), "clean": clean_pool(hbs & (r == 0)),
                          "hardneg": hn_pool(hbs)},
        "withheld_rate": {"cheat": cheat_pool((split == "heldout_rate"), (C.HELDOUT_RATE,)),
                          "clean": clean_pool(test & (r == 0) & ~hb)},
        "withheld_both": {"cheat": cheat_pool((split == "heldout_both"), (C.HELDOUT_RATE,)),
                          "clean": clean_pool(hbs & (r == 0))},
    }


def draw(pool_idx, n_players, kmax, rng, resample):
    """(n_players, kmax) game indices; each row without replacement over pool
    positions; prefixes of a row are uniform k-subsets for every k <= kmax."""
    src = rng.choice(pool_idx, len(pool_idx)) if resample else pool_idx
    kmax = min(kmax, len(src))
    keys = rng.random((n_players, len(src)))
    top = np.argpartition(keys, kmax - 1, axis=1)[:, :kmax]
    return src[top]


def simulate(cond_pools, scores, rng, resample):
    """-> {score_name: {k: {agg: (labels, values, rates)}}}"""
    out = {s: {k: {"mean": [[], [], []], "max": [[], [], []]} for k in KS} for s in scores}
    groups = [("cheat", c[2], 1, c[1], P_CHEAT) for c in cond_pools["cheat"]] + \
             [("clean", c[1], 0, 0, P_CLEAN) for c in cond_pools["clean"]]
    for kind, idx, lab, rt, npl in groups:
        g = draw(idx, npl, max(KS), rng, resample)
        for sname, sv in scores.items():
            vals = sv[g]
            for k in KS:
                kk = min(k, vals.shape[1])
                for agg, fn in (("mean", np.mean), ("max", np.max)):
                    o = out[sname][k][agg]
                    o[0].append(np.full(npl, lab)); o[1].append(fn(vals[:, :kk], axis=1)); o[2].append(np.full(npl, rt))
    return out


def summarize(sim):
    res = {}
    for sname, d in sim.items():
        res[sname] = {}
        for k, dd in d.items():
            res[sname][k] = {}
            for agg, (ls, vs, rs) in dd.items():
                y = np.concatenate(ls); v = np.concatenate(vs); rt = np.concatenate(rs)
                per_rate = {}
                for r0 in sorted(set(rt[y == 1].tolist())):
                    m = (y == 0) | (rt == r0)
                    per_rate[str(r0)] = float(roc_auc_score(y[m], v[m]))
                res[sname][k][agg] = {"auc": float(roc_auc_score(y, v)), "per_rate": per_rate}
    return res


def hardneg_check(cond_pools, scores, rng):
    if not cond_pools.get("hardneg"):
        return None
    p = {"cheat": [(f"hardneg_{bd}", -1, idx) for bd, idx in cond_pools["hardneg"] if len(idx)],
         "clean": cond_pools["clean"]}
    return summarize(simulate(p, scores, rng, resample=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C.require_gate(a.smoke)

    def body(d):
        global N_HBOOT, P_CHEAT, P_CLEAN
        if a.smoke:
            N_HBOOT, P_CHEAT, P_CLEAN = 5, 50, 100
        data = C.Data(smoke=a.smoke)
        a0, a0gid = C.load_scores("A0", a.smoke)
        s_att, _ = C.load_scores("A0", a.smoke, key="s_att")
        split_saved, _ = C.load_scores("A0", a.smoke, key="split")
        assert np.array_equal(a0gid, data.gid)
        split = data.split("v2")
        assert np.array_equal(split.astype(str), split_saved.astype(str))
        scores = {"S_att": C._clean(s_att), "A0": C._clean(a0)}
        P = pools(data, split)
        res = C.base_results("A1", a.smoke, "v2", {
            "question": "player level: is this player cheating (NOT: is this game substituted)",
            "ks": KS, "players_per_cheater_cell": P_CHEAT, "players_per_clean_band": P_CLEAN,
            "hierarchical_bootstrap_reps": N_HBOOT, "seed": C.BOOT_SEED,
            "pool_sizes": {c: {"cheat_cells": len(p["cheat"]),
                               "cheat_games_per_cell": sorted({len(x[2]) for x in p["cheat"]}),
                               "clean_games_per_band": {str(bd): len(ix) for bd, ix in p["clean"]}}
                           for c, p in P.items()},
            "conditions": {}})
        for ci, (cname, cp) in enumerate(P.items()):
            C.log(f"A1 {cname}")
            rng = np.random.default_rng([C.BOOT_SEED, ci])
            point = summarize(simulate(cp, scores, rng, resample=False))
            boots = [summarize(simulate(cp, scores, rng, resample=True)) for _ in range(N_HBOOT)]
            for sname in point:
                for k in point[sname]:
                    for agg in point[sname][k]:
                        arr = np.array([bb[sname][k][agg]["auc"] for bb in boots])
                        point[sname][k][agg]["ci95"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            res["conditions"][cname] = point
            hn = hardneg_check(cp, scores, np.random.default_rng([C.BOOT_SEED, 100 + ci]))
            if hn is not None:
                res["conditions"][cname + "_hardneg_vs_clean_players"] = hn
        C.write_json(os.path.join(d, "results.json"), res)

    C.run_arm("A1", a.smoke, body)


if __name__ == "__main__":
    main()

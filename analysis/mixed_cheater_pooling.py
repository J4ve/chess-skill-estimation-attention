"""Player-level pooling against a MIXED cheater: a player whose k games are
only PARTLY drawn from a cheating cell, the rest from that band's clean games.

Post hoc check, not pre-registered. A1's own simulated cheater
(analysis/extra_arm_a1.py) draws every one of a player's k games from
a single nonzero-substitution cell, so every game is a cheating game. Nobody
asked what happens to player-level AUC when only a fraction f of the games
are actually cheated. This script answers that, reusing A1's pooling/draw/
simulate/summarize functions by import: only the "how do we build a cheater's
hand of games" step is new (simulate_mixed below); everything downstream
(aggregation, bootstrap CI, AUC) is A1's own code, unchanged.

Score sources (same ones A1 and its A3g rerun used):
  - A0: the v2 LightGBM detector's per-game probability (A0's own scores.npz)
  - S_att: the computed attention x deviation score, same file
  - A3g: the BiGRU + gated-attention MIL sequence detector's per-game score
    (A3g_seed0/scores.npz), scored on ITS OWN grouped, twin-free split

Data: only per-game arrays are read (band/engine/rate/game_id from
rating_cheap.npz, scores/split/gid from each detector's scores.npz). np.load
on an .npz is lazy per-key, so pointing this at the full HPC files reads only
the few small arrays actually used and never touches the 19.7M-row per-ply
arrays those files also carry.

  python mixed_cheater_pooling.py [--smoke]
    [--rating-cheap PATH] [--a0-scores PATH] [--a3g-scores PATH]
    [--out-dir DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extra_arm_a1 as A1  # noqa: E402
import extra_arms_common as C  # noqa: E402

DEFAULT_THESIS = os.path.expanduser("~/Bacsain/thesis2")
FRACS = (0.1, 0.25, 0.5, 0.75, 1.0)
KS = (1, 5, 10, 20, 50, 100)
CONDITIONS = ("seen", "withheld_band", "withheld_rate", "withheld_both")
HEADLINE = "withheld_band"
BOOT_SEED = C.BOOT_SEED  # 20260915, shared with A1 so the clean-player draws line up


# ----------------------------------------------------------------------------
# data loading (per-game arrays only; bypasses C.Data, which would also pull
# in the 19.7M-row per-ply arrays rating_cheap.npz carries)
# ----------------------------------------------------------------------------
class GameMeta:
    def __init__(self, band, engine, rate, gid):
        self.band = np.asarray(band).astype(int)
        self.engine = np.asarray(engine).astype(str)
        self.rate = np.asarray(rate).astype(int)
        self.gid = np.asarray(gid).astype(str)
        self.game_index = np.array([int(g.split("game_")[-1].split(".")[0]) for g in self.gid])


def load_meta(rating_cheap_path):
    z = np.load(rating_cheap_path, allow_pickle=True)
    return GameMeta(z["band"], z["engine"], z["rate"], z["game_id"])


def load_detector_scores(path, keys):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in keys}, z["gid"].astype(str)


# ----------------------------------------------------------------------------
# mixing: how many of a player's k games actually come from the cheating cell
# ----------------------------------------------------------------------------
def mix_count(f, k):
    """Games drawn from the cheating cell for fraction f, hand size k.
    floor(f*k + 0.5), not round()/np.round() (banker's rounding: round(0.5)=0,
    round(2.5)=2), and never 0 when f > 0 -- "some" games means at least one."""
    m = int(np.floor(f * k + 0.5))
    return max(1, min(m, k))


def mix_grid(fracs=FRACS, ks=KS):
    """(f, k) -> {n_cheat_games, f_eff}. f_eff can exceed the nominal f (most
    visibly at k=1, where every f collapses to a single, fully-cheated game)."""
    out = {}
    for f in fracs:
        for k in ks:
            m = mix_count(f, k)
            out[(f, k)] = {"n_cheat_games": m, "f_eff": m / k}
    return out


# ----------------------------------------------------------------------------
# simulate_mixed: the one new function. Everything else (draw, summarize) is
# A1's, imported unchanged.
# ----------------------------------------------------------------------------
def simulate_mixed(cond_pools, scores, fracs, ks, rng_cheat, rng_fill, rng_clean, resample):
    """-> {f: {score_name: {k: {agg: (labels, values, rates)}}}}, i.e. A1's own
    per-f sim shape, ready for A1.summarize(out[f]).

    Cheat side: for each (band, engine, rate) cell, draw P_CHEAT players' worth
    of cheat-cell games AND the same band's clean games (one A1.draw call each,
    kmax = max(ks), so every k is a prefix of the same draw -- common random
    numbers across k, and across f since the mixing only changes how many of
    the two prefixes are kept).
    Clean side: A1's own clean-player draw, verbatim, shared across every f
    (a clean player has no cheat games regardless of f).
    """
    kmax = max(ks)
    out = {f: {s: {k: {"mean": [[], [], []], "max": [[], [], []]} for k in ks} for s in scores}
           for f in fracs}
    clean_by_band = dict(cond_pools["clean"])

    for cell_key, rt, idx in cond_pools["cheat"]:
        band = int(cell_key.split("_")[0])
        fidx = clean_by_band.get(band)
        if fidx is None or len(fidx) == 0:
            continue  # no band-matched filler pool; cannot build a mixed player
        cheat_draw = A1.draw(idx, A1.P_CHEAT, kmax, rng_cheat, resample)
        fill_draw = A1.draw(fidx, A1.P_CHEAT, kmax, rng_fill, resample)
        for f in fracs:
            for k in ks:
                m = mix_count(f, k)
                m = min(m, cheat_draw.shape[1])
                n_fill = min(k - m, fill_draw.shape[1])
                cg, fg = cheat_draw[:, :m], fill_draw[:, :n_fill]
                for sname, sv in scores.items():
                    combo = np.concatenate([sv[cg], sv[fg]], axis=1)
                    for agg, fn in (("mean", np.mean), ("max", np.max)):
                        o = out[f][sname][k][agg]
                        o[0].append(np.full(A1.P_CHEAT, 1))
                        o[1].append(fn(combo, axis=1))
                        o[2].append(np.full(A1.P_CHEAT, rt))

    for band, idx in cond_pools["clean"]:
        g = A1.draw(idx, A1.P_CLEAN, kmax, rng_clean, resample)
        for sname, sv in scores.items():
            vals = sv[g]
            for f in fracs:
                for k in ks:
                    kk = min(k, vals.shape[1])
                    for agg, fn in (("mean", np.mean), ("max", np.max)):
                        o = out[f][sname][k][agg]
                        o[0].append(np.full(A1.P_CLEAN, 0))
                        o[1].append(fn(vals[:, :kk], axis=1))
                        o[2].append(np.full(A1.P_CLEAN, 0))
    return out


def summarize_with_rate_ci(sim_point, sim_boots):
    """A1.summarize() attaches ci95 to the overall AUC only; the per-rate
    breakdown the task asks for needs its own percentile CI, taken over the
    same bootstrap replicates."""
    point = A1.summarize(sim_point)
    boots = [A1.summarize(b) for b in sim_boots]
    for sname in point:
        for k in point[sname]:
            for agg in point[sname][k]:
                p = point[sname][k][agg]
                arr = np.array([bb[sname][k][agg]["auc"] for bb in boots])
                p["ci95"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
                for r0 in list(p["per_rate"]):
                    rarr = np.array([bb[sname][k][agg]["per_rate"].get(r0, float("nan")) for bb in boots])
                    rarr = rarr[np.isfinite(rarr)]
                    if len(rarr) >= 20:
                        p["per_rate"][r0] = {
                            "auc": p["per_rate"][r0],
                            "ci95": [float(np.percentile(rarr, 2.5)), float(np.percentile(rarr, 97.5))],
                        }
                    else:
                        p["per_rate"][r0] = {"auc": p["per_rate"][r0], "ci95": None}
    return point


# ----------------------------------------------------------------------------
# training overlap: does this evaluation pool share (band, game_index) twins
# with games the detector trained on?
# ----------------------------------------------------------------------------
def train_overlap(meta, split, idx_by_group):
    train_keys = set(zip(meta.band[split == "train"].tolist(), meta.game_index[split == "train"].tolist()))

    def frac(idx):
        if len(idx) == 0:
            return None
        keys = list(zip(meta.band[idx].tolist(), meta.game_index[idx].tolist()))
        return sum(1 for k in keys if k in train_keys) / len(keys)

    return {name: frac(idx) for name, idx in idx_by_group.items()}


# ----------------------------------------------------------------------------
# scaling check: does matching the all-cheating (f=1) separation need ~1/f^2
# as many games as the rough "averaging shrinks noise by sqrt(k)" model predicts?
# ----------------------------------------------------------------------------
def dprime_from_auc(auc):
    """AUC = Phi(d'/sqrt(2)) under equal-variance Gaussian scores -> invert."""
    if auc is None or not np.isfinite(auc) or auc <= 0 or auc >= 1:
        return float("nan")
    return float(np.sqrt(2.0) * norm.ppf(auc))


def scaling_check(mixed_summary, fracs, ks, grid, target_k=20):
    """Two tests against the pooled mean-aggregation AUC curve:
    (a) fixed-k ratio: d'(f,k)/d'(1,k) vs the naive prediction f_eff
        (mean-aggregated score shifts linearly with the cheat fraction, so this
        part of the "1/f^2" reasoning should hold almost by construction);
    (b) games-needed: smallest k in the grid whose f-curve AUC reaches the
        f=1, k=target_k target, vs the naive target_k/f_eff^2 prediction (this
        part additionally assumes the target's own separation grows like
        sqrt(k), which the reference-note k=1..100 curves show saturates well
        before k=100 -- so this is the part expected to fail for small f).
    """
    out = {"dprime_ratio": {}, "games_needed": {}}
    target_k = target_k if target_k in ks else max(ks)
    target_auc = mixed_summary[1.0][target_k]["mean"]["auc"]
    for f in fracs:
        out["dprime_ratio"][str(f)] = {}
        for k in ks:
            d_f = dprime_from_auc(mixed_summary[f][k]["mean"]["auc"])
            d_1 = dprime_from_auc(mixed_summary[1.0][k]["mean"]["auc"])
            ratio = d_f / d_1 if (d_1 == d_1 and d_1 != 0) else float("nan")
            out["dprime_ratio"][str(f)][str(k)] = {
                "observed_ratio": ratio, "predicted_ratio_f_eff": grid[(f, k)]["f_eff"],
            }
        f_eff_t = grid[(f, target_k)]["f_eff"]
        predicted_k = target_k / (f_eff_t ** 2) if f_eff_t > 0 else float("inf")
        reached = [k for k in ks if mixed_summary[f][k]["mean"]["auc"] >= target_auc]
        out["games_needed"][str(f)] = {
            "target_k": target_k,
            "target_auc_f1": target_auc,
            "predicted_k_from_1_over_f_eff_sq": predicted_k,
            "smallest_k_in_grid_reaching_target": min(reached) if reached else None,
            "reached_within_grid": bool(reached),
        }
    return out


# ----------------------------------------------------------------------------
def run_source(name, meta, score_arrays, split, fracs, ks, n_hboot, conditions):
    """One detector's full mixed-cheater sweep across evaluation conditions."""
    result = {"conditions": {}}
    P = A1.pools(meta, split)
    for ci, cname in enumerate(conditions):
        cp = P[cname]
        C.log(f"{name} {cname}")
        rc = np.random.default_rng([BOOT_SEED, ci, 1])
        rf = np.random.default_rng([BOOT_SEED, ci, 2])
        rl = np.random.default_rng([BOOT_SEED, ci, 3])
        point = simulate_mixed(cp, score_arrays, fracs, ks, rc, rf, rl, resample=False)
        boots = []
        for _ in range(n_hboot):
            boots.append(simulate_mixed(cp, score_arrays, fracs, ks, rc, rf, rl, resample=True))
        cond_out = {}
        for f in fracs:
            sim_boots_f = [b[f] for b in boots]
            cond_out[str(f)] = summarize_with_rate_ci(point[f], sim_boots_f)
        result["conditions"][cname] = cond_out

        idx_by_group = {"cheat_games": np.concatenate([c[2] for c in cp["cheat"]]) if cp["cheat"] else np.array([], int),
                         "clean_games": np.concatenate([c[1] for c in cp["clean"]]) if cp["clean"] else np.array([], int)}
        result.setdefault("train_overlap", {})[cname] = train_overlap(meta, split, idx_by_group)
    return result


def check_f1_against_published(mixed_result, published, source_key, condition, k_grid):
    """f=1.0 (every game cheated) must reproduce A1's / A3g_player_level's own
    published numbers for the k values both grids share. Bit-exact only when
    the k grid matches exactly (see module docstring on argpartition order);
    otherwise report the observed gap so a natural mismatch is not read as a
    bug."""
    out = {}
    for k in k_grid:
        for agg in ("mean", "max"):
            try:
                mine = mixed_result["conditions"][condition]["1.0"][source_key][k][agg]["auc"]
                theirs = published["conditions"][condition][source_key][str(k)][agg]["auc"]
            except KeyError:
                continue
            out[f"k{k}_{agg}"] = {"mixed_f1": mine, "published": theirs, "abs_diff": abs(mine - theirs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rating-cheap", default=os.path.join(DEFAULT_THESIS, "analysis/v2_features/rating_cheap.npz"))
    ap.add_argument("--a0-scores", default=os.path.join(DEFAULT_THESIS, "analysis/extra_arms/A0/scores.npz"))
    ap.add_argument("--a3g-scores", default=os.path.join(DEFAULT_THESIS, "analysis/extra_arms/A3g_seed0/scores.npz"))
    ap.add_argument("--a1-published", default="analysis/extra_arms_results/A1/results.json",
                     help="path relative to CWD; run this script from the repo root. Only present "
                          "in the private thesis repo, not the public fork -- the f1 sanity check "
                          "is skipped (not failed) when this file is absent.")
    ap.add_argument("--a3g-published", default="analysis/extra_arms_results/A3g_player_level/results.json")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "mixed_cheater_pooling"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    fracs, ks, n_hboot = FRACS, KS, A1.N_HBOOT
    if a.smoke:
        fracs, ks, n_hboot = (0.5, 1.0), (1, 5), 3
        A1.P_CHEAT, A1.P_CLEAN = 40, 80

    os.makedirs(a.out_dir, exist_ok=True)
    grid = mix_grid(fracs, ks)

    meta = load_meta(a.rating_cheap)
    a0_arr, a0_gid = load_detector_scores(a.a0_scores, ("s_att", "score", "split"))
    a3g_arr, a3g_gid = load_detector_scores(a.a3g_scores, ("score", "split"))
    assert np.array_equal(meta.gid, a0_gid), "A0 scores.npz gid order does not match rating_cheap"
    assert np.array_equal(meta.gid, a3g_gid), "A3g scores.npz gid order does not match rating_cheap"

    a0_split = a0_arr["split"].astype(str)
    a3g_split = a3g_arr["split"].astype(str)
    a0_scores = {"S_att": C._clean(a0_arr["s_att"]), "A0": C._clean(a0_arr["score"])}
    a3g_scores = {"A3g": C._clean(a3g_arr["score"].astype(np.float64))}

    t0 = time.time()
    result = {
        "arm": "mixed_cheater_pooling",
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "post_hoc_disclaimer": (
            "Post hoc check on synthetic players, run after the pre-registered A1/A3g "
            "player-level results were already reported. Not a pre-registered arm; treat "
            "as exploratory."
        ),
        "question": "player level: how does pooling degrade when only a FRACTION f of a "
                    "player's k games are actually cheating games (the rest are that "
                    "player's clean games from the same band)? Contrast with A1/A3g's own "
                    "all-cheating (f=1.0) player.",
        "smoke": bool(a.smoke),
        "fracs": list(fracs), "ks": list(ks),
        "players_per_cheater_cell": A1.P_CHEAT, "players_per_clean_band": A1.P_CLEAN,
        "hierarchical_bootstrap_reps": n_hboot, "seed": BOOT_SEED,
        "headline_condition": HEADLINE,
        "mix_grid": {f"f{f}_k{k}": v for (f, k), v in grid.items()},
        "sources": {},
    }

    conditions = CONDITIONS
    result["sources"]["A0_family"] = run_source("A0_family", meta, a0_scores, a0_split, fracs, ks, n_hboot, conditions)
    result["sources"]["A3g"] = run_source("A3g", meta, a3g_scores, a3g_split, fracs, ks, n_hboot, conditions)

    # f=1.0 sanity check against the published, unmodified A1 / A3g_player_level runs
    if os.path.exists(a.a1_published) and os.path.exists(a.a3g_published):
        a1_pub = json.load(open(a.a1_published))
        a3g_pub = json.load(open(a.a3g_published))
        result["f1_check_vs_published"] = {
            "A0": check_f1_against_published(result["sources"]["A0_family"], a1_pub, "A0", HEADLINE, (1, 5, 10, 20)),
            "S_att": check_f1_against_published(result["sources"]["A0_family"], a1_pub, "S_att", HEADLINE, (1, 5, 10, 20)),
            "A3g": check_f1_against_published(result["sources"]["A3g"], a3g_pub, "A3g", HEADLINE, ks),
            "note": "k<=20 values from this script's own kmax=100 draw will not be bit-exact "
                    "to the published kmax=20 (A0) run: np.argpartition's first-k slots are the "
                    "k smallest keys but are not internally sorted, so a kmax=100 draw's first-20 "
                    "columns are not guaranteed to be the same 20 players as a kmax=20 draw. "
                    "Agreement is judged by falling inside the published CI, not by exact equality.",
        }

    # scaling check (pooled mean-aggregation curve, headline condition) on both detectors
    result["scaling_check"] = {}
    for sk, src_key, cond_key in (("A0", "A0_family", "A0"), ("A3g", "A3g", "A3g")):
        cond = result["sources"][src_key]["conditions"][HEADLINE]
        summary_by_f = {f: cond[str(f)][cond_key] for f in fracs}
        result["scaling_check"][sk] = scaling_check(summary_by_f, fracs, ks, grid)

    result["elapsed_seconds"] = time.time() - t0
    out_path = os.path.join(a.out_dir, "results.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2, default=C._json_default)
    os.replace(tmp, out_path)
    C.log(f"wrote {out_path} ({result['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()

"""A1 player-level aggregation, pointed at A3g's per-game scores.

Reuses analysis/scripts/extra_arm_a1.py's pooling/simulate/summarize logic
by import, not by copy: only the score source and the evaluation split
differ. A1 originally aggregated S_att and A0 probabilities under the v2
split; this run aggregates A3g's own game-level score
(analysis/extra_arms/A3g_seed0/scores.npz, key "score") under the split
saved in that same file, which is the grouped (twin-free) split A3g itself
was scored on.

Firstmate addition 2026-09-17 (captain approved): extends the k grid from
{1,5,10,20} to {1,5,10,20,50,100} to show where the AUC curve saturates.
Nothing else changes: same 500 cheater players/cell, 1,000 clean players/
band, seed 20260915, 200 hierarchical bootstrap reps, same RATES, same
evaluation groups. draw() (imported unchanged from extra_arm_a1) already
caps a player's k to the pool size without replacement when a pool is
smaller than the requested k; pool_size_notes() below makes any such cap
explicit in the results file instead of leaving it implicit.

Plan: analysis/anomaly-extra-arms-plan.md section 5 (A1).

  python extra_arm_a1_on_a3g.py [--smoke]
"""
import argparse
import os

import numpy as np

import extra_arm_a1 as A1
import extra_arms_common as C

KS = (1, 5, 10, 20, 50, 100)
A1.KS = KS  # only the k grid changes; A1's pools/draw/simulate/summarize/hardneg_check are reused verbatim

SRC_ARM = "A3g_seed0"
ARM = "A3g_player_level"


def pool_size_notes(P, ks):
    """Flag every (condition, pool) whose available game count is below the
    largest requested k, with the actual n, instead of silently capping."""
    notes = []
    for cname, cp in P.items():
        for key, rt, idx in cp.get("cheat", []):
            n = len(idx)
            short = [k for k in ks if k > n]
            if short:
                notes.append({"condition": cname, "pool": "cheat", "cell": key,
                              "n_available": int(n), "k_over_n": short})
        for band, idx in cp.get("clean", []):
            n = len(idx)
            short = [k for k in ks if k > n]
            if short:
                notes.append({"condition": cname, "pool": "clean", "band": int(band),
                              "n_available": int(n), "k_over_n": short})
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C.require_gate(a.smoke)
    if a.smoke:
        A1.N_HBOOT, A1.P_CHEAT, A1.P_CLEAN = 5, 50, 100

    def body(d):
        data = C.Data(smoke=a.smoke)
        a3g_score, a3g_gid = C.load_scores(SRC_ARM, a.smoke, key="score")
        a3g_split, _ = C.load_scores(SRC_ARM, a.smoke, key="split")
        assert np.array_equal(a3g_gid, data.gid), "A3g scores.npz gid order does not match rating_cheap.npz"
        split = a3g_split.astype(str)
        scores = {"A3g": C._clean(a3g_score)}
        P = A1.pools(data, split)
        res = C.base_results(ARM, a.smoke, "grouped", {
            "question": "player level: is this player cheating (NOT: is this game substituted)",
            "source_arm": SRC_ARM,
            "source_arm_split": "grouped",
            "method": "identical to A1 (analysis/scripts/extra_arm_a1.py), scored on A3g instead of S_att/A0",
            "ks": list(KS), "players_per_cheater_cell": A1.P_CHEAT, "players_per_clean_band": A1.P_CLEAN,
            "hierarchical_bootstrap_reps": A1.N_HBOOT, "seed": C.BOOT_SEED,
            "k_extension_note": "k=50,100 added 2026-09-17 by captain request, same sampling/seed/bootstrap "
                                 "as k=1,5,10,20; draw() caps a player's k to the pool size without replacement "
                                 "or fabrication when the pool is smaller than the requested k (see k_cap_notes)",
            "pool_sizes": {c: {"cheat_cells": len(p["cheat"]),
                               "cheat_games_per_cell": sorted({len(x[2]) for x in p["cheat"]}),
                               "clean_games_per_band": {str(bd): len(ix) for bd, ix in p["clean"]}}
                           for c, p in P.items()},
            "k_cap_notes": pool_size_notes(P, KS),
            "conditions": {}})
        for ci, (cname, cp) in enumerate(P.items()):
            C.log(f"{ARM} {cname}")
            rng = np.random.default_rng([C.BOOT_SEED, ci])
            point = A1.summarize(A1.simulate(cp, scores, rng, resample=False))
            boots = [A1.summarize(A1.simulate(cp, scores, rng, resample=True)) for _ in range(A1.N_HBOOT)]
            for sname in point:
                for k in point[sname]:
                    for agg in point[sname][k]:
                        arr = np.array([bb[sname][k][agg]["auc"] for bb in boots])
                        point[sname][k][agg]["ci95"] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            res["conditions"][cname] = point
            hn = A1.hardneg_check(cp, scores, np.random.default_rng([C.BOOT_SEED, 100 + ci]))
            if hn is not None:
                res["conditions"][cname + "_hardneg_vs_clean_players"] = hn
        C.write_json(os.path.join(d, "results.json"), res)

    C.run_arm(ARM, a.smoke, body)


if __name__ == "__main__":
    main()

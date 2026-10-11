"""Cluster-bootstrap OLS: what does the live estimate's ply-to-ply change
actually track? Resamples whole GAMES (not plies), since plies within a
game are not independent -- a plain per-ply bootstrap or per-ply OLS
standard error would be absurdly narrow given ~63 plies/game.

Nested models, per response, per arm:
  M0 (nuisance):      ply_band + tc_bucket + mover
  M1 (+ move quality): M0 + sqrt(cp_loss) + signed_eval_change (pawns)
  M2 (+ time spent):   M1 + time_spent_capped   (subset with time available)

Reports R^2 at each stage, the resulting delta-R^2, and 1000x cluster
bootstrap 95% CIs on every coefficient (percentile method, matching
analysis/paired_bootstrap_heldout.py's convention elsewhere in this repo).

Bootstrap is done via per-game sufficient statistics (X_g^T X_g, X_g^T y_g,
sum y_g, sum y_g^2, n_g), each precomputed once; a resample is then a
multinomial-weighted sum of those, not a refit on ~180k concatenated rows,
which is what makes 1000 resamples x 8 responses x 2 arms x 3 models fast.
"""
import json
import numpy as np
import pandas as pd

N_RESAMPLES = 1000
SEED = 20261011
TIME_SPENT_CAP_S = 120.0


def build_design(df, include_time):
    cols = {}
    cols["intercept"] = np.ones(len(df))
    for band in ["11-20", "21-40", "41-60", "61-80", "81-100"]:  # ref: 1-10
        cols[f"ply_{band}"] = (df["ply_band"] == band).astype(float).values
    for bucket in ["bullet", "rapid", "classical", "ultrabullet"]:  # ref: blitz
        cols[f"tc_{bucket}"] = (df["tc_bucket"] == bucket).astype(float).values
    cols["mover_white"] = (df["mover"] == "white").astype(float).values
    cols["cp_loss_sqrt"] = np.sqrt(df["cp_loss"].clip(lower=0)).values
    cols["signed_eval_change_pawns"] = (df["signed_eval_change"] / 100.0).values
    if include_time:
        cols["time_spent_capped"] = df["time_spent"].clip(upper=TIME_SPENT_CAP_S).values
    X = np.column_stack(list(cols.values()))
    return X, list(cols.keys())


def per_game_sufficient_stats(df, y_col, design_fn):
    sub = df.dropna(subset=[y_col]).copy()
    X, names = design_fn(sub)
    y = sub[y_col].values
    games = sub["game_id"].values
    unique_games = np.unique(games)
    g_index = {g: i for i, g in enumerate(unique_games)}
    row_game_idx = np.array([g_index[g] for g in games])

    k = X.shape[1]
    n_games = len(unique_games)
    A = np.zeros((n_games, k, k))
    b = np.zeros((n_games, k))
    sum_y = np.zeros(n_games)
    sum_y2 = np.zeros(n_games)
    n_rows = np.zeros(n_games)

    for gi in range(n_games):
        mask = row_game_idx == gi
        Xg, yg = X[mask], y[mask]
        A[gi] = Xg.T @ Xg
        b[gi] = Xg.T @ yg
        sum_y[gi] = yg.sum()
        sum_y2[gi] = (yg ** 2).sum()
        n_rows[gi] = mask.sum()

    return {
        "names": names,
        "unique_games": unique_games,
        "A": A,
        "b": b,
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "n_rows": n_rows,
        "n_rows_total": int(len(sub)),
    }


def fit_from_weights(stats, w):
    A_w = np.einsum("g,gij->ij", w, stats["A"])
    b_w = np.einsum("g,gi->i", w, stats["b"])
    n = float(np.dot(w, stats["n_rows"]))
    total_y = float(np.dot(w, stats["sum_y"]))
    total_y2 = float(np.dot(w, stats["sum_y2"]))
    beta, *_ = np.linalg.lstsq(A_w, b_w, rcond=None)
    ss_res = total_y2 - 2.0 * beta.dot(b_w) + beta.dot(A_w).dot(beta)
    ybar = total_y / n
    ss_tot = total_y2 - n * ybar ** 2
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, r2


def cluster_bootstrap_ols(df, y_col, design_fn, n_resamples=N_RESAMPLES, seed=SEED):
    stats = per_game_sufficient_stats(df, y_col, design_fn)
    n_games = len(stats["unique_games"])
    w1 = np.ones(n_games)
    beta0, r2_0 = fit_from_weights(stats, w1)

    rng = np.random.RandomState(seed)
    betas = np.zeros((n_resamples, len(stats["names"])))
    r2s = np.zeros(n_resamples)
    for r in range(n_resamples):
        counts = rng.multinomial(n_games, np.full(n_games, 1.0 / n_games))
        beta_b, r2_b = fit_from_weights(stats, counts.astype(float))
        betas[r] = beta_b
        r2s[r] = r2_b

    ci_lo = np.percentile(betas, 2.5, axis=0)
    ci_hi = np.percentile(betas, 97.5, axis=0)
    r2_lo, r2_hi = np.percentile(r2s, [2.5, 97.5])

    return {
        "n_rows": stats["n_rows_total"],
        "n_games": int(n_games),
        "r2": {"point": r2_0, "ci95": [float(r2_lo), float(r2_hi)]},
        "coefficients": {
            name: {"point": float(beta0[i]), "ci95": [float(ci_lo[i]), float(ci_hi[i])]}
            for i, name in enumerate(stats["names"])
        },
    }


def nuisance_design(include_time):
    def fn(d):
        X, names = build_design(d, include_time=include_time)
        keep = [i for i, n in enumerate(names) if n not in ("cp_loss_sqrt", "signed_eval_change_pawns")]
        return X[:, keep], [names[i] for i in keep]

    return fn


def main():
    df = pd.read_csv("merged_per_ply.csv")
    # Drop ply 1 (no previous value to diff against) and rows the engine
    # could not score (Stockfish timeout/failure -> NaN cp_loss/eval).
    df = df[df["ply"] > 1]
    df_scored = df.dropna(subset=["cp_loss", "signed_eval_change"]).copy()
    df_timed = df_scored.dropna(subset=["time_spent"]).copy()

    results = {
        "n_games_total": int(df["game_id"].nunique()),
        "n_rows_total": int(len(df)),
        "n_rows_scored": int(len(df_scored)),
        "n_rows_timed": int(len(df_timed)),
        "time_spent_cap_s": TIME_SPENT_CAP_S,
        "n_resamples": N_RESAMPLES,
        "bootstrap_seed": SEED,
        "by_arm": {},
    }

    for arm, resp_prefix in [("served", "served"), ("mirror_arm_a_gapweight", "mirror")]:
        arm_results = {}
        for response in ["own", "opp", "mean", "own_minus_opp"]:
            y_col = f"delta_{response}_{resp_prefix}"

            # Sequence A, on the full scored sample (no time requirement):
            # how much does move quality add over the ply/TC/mover baseline.
            m0_scored = cluster_bootstrap_ols(df_scored, y_col, nuisance_design(False))
            m1_scored = cluster_bootstrap_ols(df_scored, y_col, lambda d: build_design(d, include_time=False))

            # Sequence B, on the SAME rows (df_timed) for all three nested
            # models: this is the only valid way to compare move quality's
            # and time's marginal R^2, since R^2 is not comparable across
            # models fit on different row sets. df_timed drops exactly one
            # row per game (ply 2; see time_spent note in the merge script).
            m0_timed = cluster_bootstrap_ols(df_timed, y_col, nuisance_design(False))
            m1_timed = cluster_bootstrap_ols(df_timed, y_col, lambda d: build_design(d, include_time=False))
            m2_timed = cluster_bootstrap_ols(df_timed, y_col, lambda d: build_design(d, include_time=True))

            arm_results[response] = {
                "y_col": y_col,
                "scored_sample": {
                    "nuisance_only": m0_scored,
                    "plus_move_quality": m1_scored,
                    "delta_r2_move_quality": m1_scored["r2"]["point"] - m0_scored["r2"]["point"],
                },
                "timed_sample_nested": {
                    "nuisance_only": m0_timed,
                    "plus_move_quality": m1_timed,
                    "plus_move_quality_and_time": m2_timed,
                    "delta_r2_move_quality": m1_timed["r2"]["point"] - m0_timed["r2"]["point"],
                    "delta_r2_time": m2_timed["r2"]["point"] - m1_timed["r2"]["point"],
                },
            }
            print(
                f"[{arm}] {response}: scored R2 nuisance={m0_scored['r2']['point']:.5f} "
                f"+quality={m1_scored['r2']['point']:.5f} | "
                f"timed R2 nuisance={m0_timed['r2']['point']:.5f} "
                f"+quality={m1_timed['r2']['point']:.5f} "
                f"+time={m2_timed['r2']['point']:.5f} "
                f"(dR2_quality={arm_results[response]['timed_sample_nested']['delta_r2_move_quality']:.5f}, "
                f"dR2_time={arm_results[response]['timed_sample_nested']['delta_r2_time']:.5f})",
                flush=True,
            )
        results["by_arm"][arm] = arm_results

    # Robustness: same nested sequence (scored sample), restricted to
    # ply >= 11 (the project's own critical_moves.DEFAULT_MIN_PLY cutoff,
    # i.e. ply > 10 one-indexed), since opening swings of hundreds of
    # points may dominate SS_tot and keep every R^2 tiny regardless of what
    # is added.
    df_scored_minply = df_scored[df_scored["ply"] > 10].copy()
    minply_results = {}
    for arm, resp_prefix in [("served", "served"), ("mirror_arm_a_gapweight", "mirror")]:
        arm_minply = {}
        for response in ["own", "opp", "mean", "own_minus_opp"]:
            y_col = f"delta_{response}_{resp_prefix}"
            m0 = cluster_bootstrap_ols(df_scored_minply, y_col, nuisance_design(False), n_resamples=500)
            m1 = cluster_bootstrap_ols(
                df_scored_minply, y_col, lambda d: build_design(d, include_time=False), n_resamples=500
            )
            arm_minply[response] = {
                "nuisance_only": m0,
                "plus_move_quality": m1,
                "delta_r2_move_quality": m1["r2"]["point"] - m0["r2"]["point"],
            }
        minply_results[arm] = arm_minply
    results["ply_gt_10_only"] = {"n_rows": int(len(df_scored_minply)), "by_arm": minply_results}

    with open("regression_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote regression_results.json")


if __name__ == "__main__":
    main()

"""Regenerates the two numbers the attention-front-loading caveat in
move-explanation-results.md depends on, so they are not ad-hoc snippets:

  1. Observed mean attention weight by ply, over every sampled game's
     offline/full-game pass (served checkpoint only; this is also the
     live curve's final-ply value, see move_explanation_curves.py).
  2. A uniform-attention baseline at the same per-game lengths: if
     attention at query t were split evenly over its t causally-valid
     keys (chess_rating_net.py's attention module, forward_causal), the
     mean weight key k would receive, averaged over all T queries the
     same way api.py's attention_weights = ...mean(dim=1) does, is
     mean_{t=k..T}(1/t) = (H_T - H_{k-1}) / T, H = harmonic numbers.

Also reports the fraction of Stockfish positions that reached the full
search depth, from the engine pass's own recorded depth_before/depth_after.

Usage: python move_explanation_attention_diagnostics.py
  (run from the directory containing hpc_out/curves_3000.jsonl and
  hpc_out/engine_3000.jsonl; writes attention_diagnostics.json there)
"""
import json

import numpy as np


def load_curves(path="hpc_out/curves_3000.jsonl"):
    curves = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            curves[d["game_id"]] = d
    return curves


def observed_mean_attention_by_ply(curves):
    by_ply = {}
    for c in curves.values():
        attn = c["served"].get("offline_attention")
        if not attn:
            continue
        for i, a in enumerate(attn):
            by_ply.setdefault(i + 1, []).append(a)
    return {ply: float(np.mean(vals)) for ply, vals in by_ply.items()}, {
        ply: len(vals) for ply, vals in by_ply.items()
    }


def uniform_baseline_by_ply(curves):
    lengths = [
        len(c["served"]["offline_attention"])
        for c in curves.values()
        if c["served"].get("offline_attention")
    ]
    lengths = np.array(lengths)
    max_T = int(lengths.max())
    H = np.cumsum(1.0 / np.arange(1, max_T + 1))

    def uniform_weight(k, T):
        Hk_1 = H[k - 2] if k >= 2 else 0.0
        return (H[T - 1] - Hk_1) / T

    acc, cnt = {}, {}
    for T in lengths:
        T = int(T)
        for k in range(1, T + 1):
            acc[k] = acc.get(k, 0.0) + uniform_weight(k, T)
            cnt[k] = cnt.get(k, 0) + 1
    return {k: acc[k] / cnt[k] for k in acc}


def depth_and_terminal_stats(sf_depth, path="hpc_out/engine_3000.jsonl"):
    """Every position's depth, cross-checked against whether it is a
    genuinely terminal (checkmate/stalemate) position: depth_before[t] for
    t=0..n-1 covers positions 0..n-1; depth_after[-1] is the one additional
    (final) position per game not covered by depth_before. A position short
    of the configured depth is "terminal" here if it is that final position
    AND its eval is mate-magnitude (|eval| >= 9999, the MATE_CP_CLAMP
    boundary): Stockfish correctly returns an exact, zero-search score for a
    position with no legal moves, which is not a 1-second-cap failure."""
    n_total = 0
    n_at_depth = 0
    n_short_terminal = 0
    n_short_nonterminal = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                continue
            db = d.get("depth_before") or []
            da = d.get("depth_after") or []
            eb = d.get("eval_before_white_pov") or []
            ea = d.get("eval_after_white_pov") or []
            depths = list(db) + ([da[-1]] if da else [])
            evals = list(eb) + ([ea[-1]] if ea else [])
            n = len(depths)
            for i, (x, e) in enumerate(zip(depths, evals)):
                if x is None:
                    continue
                n_total += 1
                if x >= sf_depth:
                    n_at_depth += 1
                    continue
                is_last = i == n - 1
                is_mate = e is not None and abs(e) >= 9999
                if is_last and is_mate:
                    n_short_terminal += 1
                else:
                    n_short_nonterminal += 1
    return {
        "n_total": n_total,
        "n_at_configured_depth": n_at_depth,
        "n_short_terminal_checkmate_or_stalemate": n_short_terminal,
        "n_short_nonterminal_time_cap": n_short_nonterminal,
        "fraction_at_configured_depth": n_at_depth / n_total if n_total else None,
        "fraction_short_explained_by_terminal_positions": (
            n_short_terminal / (n_short_terminal + n_short_nonterminal)
            if (n_short_terminal + n_short_nonterminal)
            else None
        ),
    }


def main():
    curves = load_curves()
    observed, n_games_at_ply = observed_mean_attention_by_ply(curves)
    uniform = uniform_baseline_by_ply(curves)

    sf_depth = json.load(open("hpc_out/engine_3000.jsonl.meta.json"))["sf_depth"]
    depth_stats = depth_and_terminal_stats(sf_depth)

    out = {
        "observed_mean_attention_by_ply": observed,
        "n_games_at_ply": n_games_at_ply,
        "uniform_baseline_mean_attention_by_ply": uniform,
        "ply1_ply90_ratio_observed": observed[1] / observed[90],
        "ply1_ply90_ratio_uniform": uniform[1] / uniform[90],
        "stockfish": {"configured_depth": sf_depth, **depth_stats},
    }
    with open("attention_diagnostics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"ply1={observed[1]:.5f} ply2={observed[2]:.5f} ply90={observed[90]:.5f}")
    print(f"uniform ply1={uniform[1]:.5f} ply2={uniform[2]:.5f} ply90={uniform[90]:.5f}")
    print(f"observed ratio={out['ply1_ply90_ratio_observed']:.2f} uniform ratio={out['ply1_ply90_ratio_uniform']:.2f}")
    print(
        f"stockfish depth: {depth_stats['n_at_configured_depth']}/{depth_stats['n_total']} "
        f"({100*depth_stats['fraction_at_configured_depth']:.4f}%) reached depth {sf_depth}; "
        f"of the shortfall, {depth_stats['n_short_terminal_checkmate_or_stalemate']} are terminal "
        f"(checkmate/stalemate) positions and {depth_stats['n_short_nonterminal_time_cap']} are "
        "genuine 1-second-cap shortfalls"
    )
    print("Wrote attention_diagnostics.json")


if __name__ == "__main__":
    main()

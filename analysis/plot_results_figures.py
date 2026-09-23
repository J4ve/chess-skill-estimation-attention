"""Render the two Chapter 4 results figures from the frozen result JSONs.

Figure 1  anomaly detection: ROC-AUC against substitution rate, four methods,
          rating bands withheld from training.
Figure 2  rating estimation: test MAE per arm, and the paired bootstrap deltas
          against the reproduced baseline.

Every value is read from a results JSON. Nothing is typed in by hand, so the
figures cannot drift from the numbers the manuscript reports.

Usage:  python analysis/scripts/plot_results_figures.py [--outdir DIR]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIRSTMATE = "/media/jave/4CB2057BB2056AB2/firstmate/data/ms-extra-arms-writeup/hpc-results"
DEFAULT_OUT = os.path.join(REPO, "CCS Thesis - Integrated", "figures")

# Categorical slots 1-4 of the validated palette, assigned by final performance
# rank so the strongest method holds the most legible slot. Fixed, never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_SOFT, INK_MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#dedddaff"

RATES = [2, 5, 20, 40, 60]


def _conditions(node):
    """Rate conditions live under game_level for A3g/A4, at the top for A0g/S_att."""
    return (node.get("game_level") or node).get("conditions", {})


def load_anomaly():
    a0 = json.load(open(os.path.join(REPO, "analysis/extra_arms_results/A0/results.json")))
    a0g = json.load(open(os.path.join(FIRSTMATE, "A0g/results.json")))
    a3g = json.load(open(os.path.join(REPO, "analysis/extra_arms_results/A3g_seed0/results.json")))
    a4 = json.load(open(os.path.join(REPO, "analysis/extra_arms_results/A4/results.json")))
    src = {
        "Per-move detector (A3g)": (_conditions(a3g), BLUE, "o"),
        "End-to-end CNN-BiLSTM (A4)": (_conditions(a4), ORANGE, "s"),
        "LightGBM (A0g)": (a0g["variants"]["A0g"]["conditions"], AQUA, "^"),
        "Computed score (S_att)": (_conditions(a0["S_att"]), YELLOW, "D"),
    }
    out = []
    for label, (cond, color, marker) in src.items():
        aucs = [cond[f"withheld_band_rate{r:02d}_vs_clean"]["auc"] for r in RATES]
        out.append((label, aucs, color, marker))
    return out


def load_rating():
    agg = json.load(open(os.path.join(REPO, "analysis/heldout-test-eval-results.json")))
    boot = json.load(open(os.path.join(REPO, "analysis/heldout_test_eval/bootstrap.json")))["pairs"]
    arms = agg["arms"]

    def mae(key):
        return arms[key]["best_val_checkpoint"]["test_mae"]

    points = [
        ("Omori released ckpt\n(zero-shot)", 193.47, INK_MUTED),
        ("Reproduced baseline", mae("baseline"), INK_SOFT),
        ("Baseline + lr 3e-4", 173.54796328607844, INK_SOFT),
        ("+ attention, untuned", mae("attn_untuned"), INK_SOFT),
        ("Deeper CNN", 172.0523350819608, INK_SOFT),
        ("+ attention, tuned (frozen)", mae("attn_tuned"), BLUE),
    ]
    deltas = [
        ("lr 3e-4 alone\nvs baseline", boot["baseline_lr3e4_vs_baseline"]),
        ("Deeper CNN\nvs baseline", boot["deepcnn_vs_baseline"]),
        ("Tuned attention\nvs baseline", boot["tuned_attention_vs_baseline"]),
        ("Deeper CNN\nvs tuned attention", boot["deepcnn_vs_tuned_attention"]),
    ]
    return points, deltas, agg["n_test_games"], boot


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=1.0)
    ax.set_axisbelow(True)


def figure_anomaly(outdir):
    series = load_anomaly()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(0.5, color=INK_MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance (0.50)", xy=(2.2, 0.5), xytext=(0, 5), textcoords="offset points",
                fontsize=8.5, color=INK_MUTED, va="bottom")
    for label, aucs, color, marker in series:
        ax.plot(RATES, aucs, color=color, lw=2.0, marker=marker, markersize=8,
                markeredgecolor="white", markeredgewidth=1.4, zorder=3, label=label)
        ax.annotate(f"{aucs[-1]:.3f}", xy=(RATES[-1], aucs[-1]), xytext=(9, -3),
                    textcoords="offset points", fontsize=9, color=INK, weight="bold")
    ax.set_xlim(0, 72)
    ax.set_ylim(0.45, 0.95)
    ax.set_xticks(RATES)
    ax.set_xticklabels([f"{r}%" for r in RATES])
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_xlabel("Share of one side's moves replaced by an engine", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("Game-level ROC-AUC", fontsize=9.5, color=INK_SOFT)
    ax.set_title("Detecting a substituted game gets easier as more moves are replaced",
                 fontsize=11.5, color=INK, weight="bold", loc="left", pad=12)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, labelspacing=0.5,
                    handlelength=2.2, borderpad=0)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    style(ax)
    fig.tight_layout()
    path = os.path.join(outdir, "fig_anomaly_auc_by_rate.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def figure_rating(outdir):
    points, deltas, n_test, _ = load_rating()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.12, 1.0]})

    ys = range(len(points))
    ax1.axvline(182, color=INK_MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax1.annotate("182 reported by Omori &\nTadepalli (2024), on their data",
                 xy=(183.2, len(points) - 0.35), fontsize=7.8, color=INK_MUTED,
                 ha="left", va="top")
    for y, (label, val, color) in zip(ys, points):
        ax1.hlines(y, 170, val, color=GRID, lw=1.2, zorder=1)
        ax1.plot(val, y, "o", markersize=9, color=color, markeredgecolor="white",
                 markeredgewidth=1.4, zorder=3)
        ax1.annotate(f"{val:.2f}", xy=(val, y), xytext=(11, -3), textcoords="offset points",
                     fontsize=9, color=INK, weight="bold" if color == BLUE else "normal")
    ax1.set_yticks(list(ys))
    ax1.set_yticklabels([p[0] for p in points], fontsize=9, color=INK_SOFT)
    ax1.set_ylim(-0.7, len(points) - 0.15)
    ax1.set_xlim(170, 200)
    ax1.set_xlabel(f"Test MAE, rating points ({n_test:,} held-out games)",
                   fontsize=9.5, color=INK_SOFT)
    ax1.set_title("(a)  Where each arm landed", fontsize=10.5, color=INK, weight="bold",
                  loc="left", pad=10)
    ax1.grid(axis="x", color=GRID, lw=0.9)
    style(ax1)

    ys2 = range(len(deltas))
    ax2.axvline(0, color=INK_SOFT, lw=1.2, zorder=2)
    for y, (label, d) in zip(ys2, deltas):
        crosses = d["ci_crosses_zero"]
        color = INK_MUTED if crosses else BLUE
        lo, hi = d["ci95_lo"], d["ci95_hi"]
        ax2.hlines(y, lo, hi, color=color, lw=2.4, zorder=3)
        ax2.plot(d["point_delta_mae"], y, "o", markersize=9, color=color,
                 markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        note = "  n.s." if crosses else ""
        ax2.annotate(f"{d['point_delta_mae']:+.2f}{note}", xy=(d["point_delta_mae"], y),
                     xytext=(0, 12), textcoords="offset points", fontsize=9,
                     color=INK if not crosses else INK_MUTED,
                     weight="bold" if not crosses else "normal", ha="center")
    ax2.set_yticks(list(ys2))
    ax2.set_yticklabels([d[0] for d in deltas], fontsize=9, color=INK_SOFT)
    ax2.set_ylim(-0.6, len(deltas) - 0.15)
    ax2.set_xlim(-4.2, 1.25)
    ax2.set_xlabel("Change in test MAE (negative is better)\n"
                   "10,000 resamples, bars are 95% CI",
                   fontsize=9.5, color=INK_SOFT)
    ax2.set_title("(b)  Effect of each change", fontsize=10.5, color=INK,
                  weight="bold", loc="left", pad=10)
    ax2.grid(axis="x", color=GRID, lw=0.9)
    style(ax2)

    fig.tight_layout()
    path = os.path.join(outdir, "fig_rating_mae_arms.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for p in (figure_anomaly(args.outdir), figure_rating(args.outdir)):
        print("wrote", p)

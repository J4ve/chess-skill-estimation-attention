"""Renders the mixed-cheater-pooling chart (AUC vs. games pooled, one line per
cheat fraction f, mean aggregation, headline withheld-band condition) from the
results.json written by mixed_cheater_pooling.py.

  python plot_mixed_cheater_pooling.py [--results PATH] [--out PATH]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# dataviz skill categorical palette, fixed order, light mode
F_COLORS = {
    0.1: "#2a78d6",   # slot 1 blue
    0.25: "#eb6834",  # slot 2 orange
    0.5: "#1baf7a",   # slot 3 aqua
    0.75: "#eda100",  # slot 4 yellow
    1.0: "#e34948",   # slot 8 red (all-cheating, the reference case)
}
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def draw_panel(ax, ks, series_by_f, title, target_auc, target_k):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":", zorder=1)
    if target_auc is not None:
        ax.axhline(target_auc, color=MUTED, linewidth=1, linestyle="--", zorder=1)
        ax.text(ks[0], target_auc + 0.01, f"f=1.0 @ k={target_k} ({target_auc:.3f})",
                color=SECONDARY, fontsize=8, va="bottom")

    for f in sorted(series_by_f):
        vals = series_by_f[f]
        ax.plot(ks, vals, color=F_COLORS[f], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=3, label=f"f = {f}")
        ax.text(ks[-1] * 1.05, vals[-1], f"{vals[-1]:.2f}", color=F_COLORS[f],
                fontsize=8, va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.35)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("games pooled per player (k)", color=SECONDARY, fontsize=9)
    ax.set_ylabel("player-level AUC (mean aggregation)", color=SECONDARY, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, fontweight="bold", loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "mixed_cheater_pooling", "results.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "mixed_cheater_pooling", "auc_vs_games.png"))
    ap.add_argument("--condition", default=None, help="defaults to the results file's headline_condition")
    a = ap.parse_args()

    d = json.load(open(a.results))
    cond = a.condition or d["headline_condition"]
    ks = d["ks"]
    fracs = d["fracs"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)

    panels = [
        ("A0_family", "A0", "LightGBM detector (A0)"),
        ("A3g", "A3g", "Sequence detector (A3g)"),
    ]
    for ax, (src_key, score_key, title) in zip(axes, panels):
        cond_data = d["sources"][src_key]["conditions"][cond]
        series_by_f = {}
        for f in fracs:
            series_by_f[f] = [cond_data[str(f)][score_key][str(k)]["mean"]["auc"] for k in ks]
        target = cond_data["1.0"][score_key][str(max(k for k in ks if k <= 20))]["mean"]["auc"]
        target_k = max(k for k in ks if k <= 20)
        draw_panel(ax, ks, series_by_f, title, target, target_k)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(fracs), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, labelcolor=INK)
    fig.suptitle(f"Player-level AUC for a mixed cheater, by cheat fraction f ({cond.replace('_', ' ')})",
                 color=INK, fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

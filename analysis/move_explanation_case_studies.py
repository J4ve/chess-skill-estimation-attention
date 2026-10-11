"""Select the two illustrative TEST games and build their figures + JSON.

Selection rules (fixed before looking at the plots, so the picks are not
cherry-picked post hoc):
  - Accurate game: among sampled TEST games with >= 40 plies, the smallest
    final-ply mean |live estimate - true Elo| across both sides.
  - Swing game: among the same >= 40-ply pool, the largest (max - min) of
    the mean-of-both-sides live estimate, restricted to ply > 15 so opening
    volatility (see the attention front-loading note) cannot win this by
    construction.

Each figure is three panels sharing a ply x-axis: the live (prefix-only)
rating curve for both sides against the true ratings, with markers on the
largest centipawn-loss moves (ply >= 10 only, matching
prototype/src/critical_moves.py's own DEFAULT_MIN_PLY); the Stockfish
evaluation track (White POV); and the model's own offline/full-game
attention weight, which is also the LIVE curve's value at the final ply
(same forward pass -- see move_explanation_curves.py's docstring). The
model never sees Stockfish; every panel/caption says so.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BLUE = "#2a78d6"    # White
ORANGE = "#eb6834"  # Black
GRAY = "#8a8a86"
VIOLET = "#6250d6"
MIN_PLY_FOR_CRITICAL = 10  # matches prototype/src/critical_moves.py DEFAULT_MIN_PLY
EVAL_CLIP_CP = 1000


def load_data():
    curves = {}
    with open("hpc_out/curves_3000.jsonl") as f:
        for line in f:
            d = json.loads(line)
            curves[d["game_id"]] = d
    engine = {}
    with open("hpc_out/engine_3000.jsonl") as f:
        for line in f:
            d = json.loads(line)
            engine[d["game_id"]] = d
    return curves, engine


def select_games(curves, engine):
    candidates = []
    for gid, c in curves.items():
        e = engine.get(gid)
        if e is None or "error" in e:
            continue
        n = min(c["n_plies"], e["n_plies"])
        if n < 40:
            continue
        we, be = c.get("white_elo"), c.get("black_elo")
        if we is None or be is None:
            continue
        we, be = float(we), float(be)
        sw, sb = c["served"]["live_white"][:n], c["served"]["live_black"][:n]
        final_err = (abs(sw[-1] - we) + abs(sb[-1] - be)) / 2.0
        mean_curve = [(w + b) / 2.0 for w, b in zip(sw, sb)]
        tail = mean_curve[15:]
        swing = (max(tail) - min(tail)) if tail else 0.0
        candidates.append((gid, n, final_err, swing))

    accurate_gid = min(candidates, key=lambda x: x[2])[0]
    swing_gid = max(candidates, key=lambda x: x[3])[0]
    return accurate_gid, swing_gid


def build_case_json(gid, c, e):
    n = min(c["n_plies"], e["n_plies"])
    sw, sb = c["served"]["live_white"][:n], c["served"]["live_black"][:n]
    attn = (c["served"].get("offline_attention") or [None] * n)[:n]
    rows = []
    for i in range(n):
        rows.append(
            dict(
                ply=i + 1,
                mover=e["mover"][i],
                san=e["san"][i],
                live_white=sw[i],
                live_black=sb[i],
                eval_before_white_pov=e["eval_before_white_pov"][i],
                eval_after_white_pov=e["eval_after_white_pov"][i],
                cp_loss=e["cp_loss"][i],
                phase=e["phase"][i],
                attention_weight=attn[i],
            )
        )
    return {
        "game_id": gid,
        "n_plies": n,
        "white_elo": float(c.get("white_elo")),
        "black_elo": float(c.get("black_elo")),
        "time_control": c.get("time_control"),
        "note": (
            "The model never sees a chess engine. 'eval_*_white_pov' is Stockfish "
            "16 (depth 17, 1.0s cap), shown only as an external yardstick to "
            "interpret the model's live (prefix-only) rating estimate. "
            "'live_white'/'live_black' are the prefix-recompute estimate -- "
            "what the prototype shows as the game is streamed -- not the "
            "offline bidirectional per-ply output."
        ),
        "per_ply": rows,
    }


def make_figure(gid, case, out_path, title_label):
    rows = case["per_ply"]
    plies = [r["ply"] for r in rows]
    live_w = [r["live_white"] for r in rows]
    live_b = [r["live_black"] for r in rows]
    we, be = case["white_elo"], case["black_elo"]

    eval_track = [r["eval_before_white_pov"] for r in rows]
    last_after = rows[-1]["eval_after_white_pov"]
    eval_plies = [0] + plies
    eval_track_full = [eval_track[0] if eval_track[0] is not None else 0.0] + eval_track
    if last_after is not None:
        eval_plies = eval_plies + [plies[-1]]
        eval_track_full = eval_track_full + [last_after]

    def clip(v):
        if v is None:
            return 0.0
        return max(-EVAL_CLIP_CP, min(EVAL_CLIP_CP, v))

    eval_clipped = [clip(v) for v in eval_track_full]

    attn = [r["attention_weight"] or 0.0 for r in rows]

    # Largest centipawn-loss moves, ply >= 10 only (see module docstring).
    scored = [r for r in rows if r["ply"] >= MIN_PLY_FOR_CRITICAL and r["cp_loss"] is not None]
    top_moves = sorted(scored, key=lambda r: -r["cp_loss"])[:5]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.0]}
    )
    fig.suptitle(title_label, fontsize=12, fontweight="bold")

    # Panel 1: live rating curves vs true ratings.
    ax1.plot(plies, live_w, color=BLUE, linewidth=2, label="White, live estimate")
    ax1.plot(plies, live_b, color=ORANGE, linewidth=2, label="Black, live estimate")
    ax1.axhline(we, color=BLUE, linewidth=1, linestyle="--", alpha=0.6, label=f"White true rating ({we:.0f})")
    ax1.axhline(be, color=ORANGE, linewidth=1, linestyle="--", alpha=0.6, label=f"Black true rating ({be:.0f})")
    for r in top_moves:
        mover_color = BLUE if r["mover"] == "white" else ORANGE
        y = r["live_white"] if r["mover"] == "white" else r["live_black"]
        ax1.scatter([r["ply"]], [y], marker="v", s=70, color=mover_color, edgecolor="black", linewidth=0.6, zorder=5)
    # Stagger the top-3 labels' vertical offset (and alternate which side of
    # the marker they sit on) so two markers close in ply don't collide.
    offsets = [(0, 16), (18, 30), (-18, 30)]
    for r, (dx, dy) in zip(top_moves[:3], offsets):
        y = r["live_white"] if r["mover"] == "white" else r["live_black"]
        cp_label = f"-{r['cp_loss']:.0f}cp, capped" if r["cp_loss"] >= 2000.0 else f"-{r['cp_loss']:.0f}cp"
        ax1.annotate(
            f"{r['san']} ({cp_label})",
            xy=(r["ply"], y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            color="#333333",
            zorder=6,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
        )
    ax1.set_ylabel("Rating estimate")
    ax1.legend(loc="best", fontsize=8, ncol=2, frameon=False)
    ax1.set_title("Live (prefix-only) rating estimate, both sides, vs true rating", fontsize=9.5, loc="left")

    # Panel 2: Stockfish evaluation track, White POV.
    ax2.fill_between(eval_plies, eval_clipped, 0, where=[v >= 0 for v in eval_clipped], color=BLUE, alpha=0.5, linewidth=0)
    ax2.fill_between(eval_plies, eval_clipped, 0, where=[v < 0 for v in eval_clipped], color=ORANGE, alpha=0.5, linewidth=0)
    ax2.axhline(0, color=GRAY, linewidth=1)
    ax2.set_ylabel("Eval (cp, White POV)")
    ax2.set_ylim(-EVAL_CLIP_CP * 1.05, EVAL_CLIP_CP * 1.05)
    ax2.set_title(
        f"Stockfish 16 evaluation (depth 17, clipped at +/-{EVAL_CLIP_CP}cp) -- "
        "an external yardstick; the model never sees this",
        fontsize=9.5, loc="left",
    )

    # Panel 3: attention weight, with the front-loaded opening shaded out.
    ax3.fill_between(plies, attn, 0, color=VIOLET, alpha=0.7, linewidth=0, step="mid")
    ax3.axvspan(0.5, MIN_PLY_FOR_CRITICAL - 0.5, color=GRAY, alpha=0.18, linewidth=0)
    ax3.set_ylabel("Attention wt.")
    ax3.set_xlabel("Ply")
    ax3.set_title(
        f"Attention weight (shaded: ply < {MIN_PLY_FOR_CRITICAL}, mechanically front-loaded, "
        "excluded from critical-move ranking)",
        fontsize=9.5, loc="left",
    )
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    for ax in (ax1, ax2, ax3):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    curves, engine = load_data()
    accurate_gid, swing_gid = select_games(curves, engine)
    print("accurate game:", accurate_gid)
    print("swing game:", swing_gid)

    for gid, label, out_prefix in [
        (accurate_gid, "Most accurate final estimate", "accurate"),
        (swing_gid, "Largest live-estimate swing", "swing"),
    ]:
        c, e = curves[gid], engine[gid]
        case = build_case_json(gid, c, e)
        with open(f"case_{out_prefix}.json", "w") as f:
            json.dump(case, f, indent=2)
        make_figure(
            gid,
            case,
            f"move-explanation-{out_prefix}.png",
            f"{label}: {gid} (White {case['white_elo']}, Black {case['black_elo']})",
        )

    with open("case_study_selection.json", "w") as f:
        json.dump({"accurate_game_id": accurate_gid, "swing_game_id": swing_gid}, f, indent=2)


if __name__ == "__main__":
    main()

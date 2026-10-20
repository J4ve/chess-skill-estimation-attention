# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy",
#     "chess",
# ]
# ///
"""Interactive diagrams for the thesis defense.

Ten self-contained diagrams, one per section, each built only from the JSON
files in ``data/`` beside it (fetched from this repository when
the notebook runs without that folder, as on molab). Nothing here loads the model, so the notebook
runs in a browser with no GPU and no cluster. Rebuild the data with
``presentation/build_data.py``; see ``presentation/README.md`` to run or export.

To lift one diagram into a presentation notebook, copy the Setup section's two
cells once and then the cells of the diagram you want. Every global is prefixed
with its diagram number, so the groups do not collide.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="How the thesis works")


@app.cell
def _():
    import json
    import urllib.request

    import altair as alt
    import chess
    import chess.svg
    import marimo as mo
    import numpy as np
    import pandas as pd

    return alt, chess, json, mo, np, pd, urllib


@app.cell
def _(alt, json, mo, urllib):
    RAW_DATA_URL = (
        "https://raw.githubusercontent.com/J4ve/chess-skill-estimation-attention/"
        "main/notebooks/data/"
    )

    def load(name):
        """Read one precomputed data file.

        ``mo.notebook_location()`` is a filesystem path when the notebook runs
        locally and a URL when it runs in the browser, and only the first kind
        has ``read_text``. Locally the files are read from ``data/``, which is
        where the build script writes them; in the browser they are read from
        ``public/``, the path the export publishes them at. ``public`` is a
        symbolic link to ``data``, so a checkout that cannot create symbolic
        links still works.
        """
        location = mo.notebook_location()
        if hasattr(location, "read_text"):
            local = location / "data" / name
            if local.exists():
                return json.loads(local.read_text())
            # A notebook mirrored on its own (molab, a single downloaded file)
            # has no data folder beside it, so read the copy published here.
            return json.loads(
                urllib.request.urlopen(RAW_DATA_URL + name).read().decode()
            )
        return json.loads(
            urllib.request.urlopen(str(location / "public" / name)).read().decode()
        )

    # Categorical slots from the project's validated palette, assigned by
    # entity and never cycled.
    C = {
        "a": "#2a78d6",      # slot 1, blue
        "b": "#eb6834",      # slot 2, orange
        "c": "#4a3aa7",      # slot 7, violet
        "warn": "#e34948",   # slot 8, red
        "ok": "#1baf7a",     # slot 3, aqua
        "muted": "#6b6a66",
        "grid": "#d9d8d3",
        "ink": "#52514e",
    }

    def style(chart):
        """Recessive axes and grid, applied to the outermost chart only."""
        return chart.configure_axis(
            gridColor=C["grid"], domainColor=C["grid"], tickColor=C["grid"],
            labelColor=C["ink"], titleColor=C["ink"], labelFontSize=11, titleFontSize=11,
        ).configure_legend(labelColor=C["ink"], titleColor=C["ink"], labelFontSize=11,
                           titleFontSize=11).configure_view(strokeWidth=0)

    def head(number, title, caption):
        return mo.md(
            f"## {number}. {title}\n\n"
            f"<div style='font-size:0.95rem;opacity:0.85'>{caption}</div>"
        )

    def src(*lines):
        body = "<br>".join(lines)
        return mo.md(
            "<div style='font-size:0.74rem;opacity:0.62;line-height:1.5;"
            f"border-top:1px solid #e3e2dd;padding-top:6px;margin-top:4px'>{body}</div>"
        )

    def stat(label, value, sub=""):
        return mo.md(
            "<div style='padding:8px 14px;border:1px solid #e3e2dd;border-radius:8px'>"
            f"<div style='font-size:0.72rem;opacity:0.65'>{label}</div>"
            f"<div style='font-size:1.55rem;font-weight:700;line-height:1.2'>{value}</div>"
            f"<div style='font-size:0.72rem;opacity:0.65'>{sub}</div></div>"
        )

    def notes(*items):
        """Secondary notes, collapsed so a diagram stays slide sized."""
        body = "".join(f"<li style='margin-bottom:6px'>{t}</li>" for t in items)
        return mo.accordion(
            {"Notes and caveats": mo.md(
                f"<ul style='font-size:0.84rem;margin:0;padding-left:18px'>{body}</ul>"
            )}
        )

    def flag(text, kind="note"):
        color = {"note": C["muted"], "warn": C["warn"], "ok": C["ok"]}[kind]
        return mo.md(
            f"<div style='font-size:0.8rem;border-left:3px solid {color};"
            f"padding:4px 10px;opacity:0.85'>{text}</div>"
        )

    _ = alt.data_transformers.disable_max_rows()
    return C, flag, head, load, notes, src, stat, style


@app.cell
def _(mo):
    mo.md(
        """
        # How the thesis works

        Ten interactive diagrams covering what was built, what was measured, and
        where it falls short. Every number traces to a file in the study's
        repository, named in small print under each diagram.
        """
    )
    return


# ==========================================================================
# 1. Watch the model read a game
# ==========================================================================


@app.cell
def _(head):
    head(
        1,
        "Watch the model read a game",
        "Drag the ply slider: the board, the two rating estimates, the attention "
        "weight, the per-move deviation and the running suspicion score all move "
        "with it.",
    )
    return


@app.cell
def _(load):
    d1_data = load("d1_games.json")
    d1_games = {g["title"]: g for g in d1_data["games"]}
    return d1_data, d1_games


@app.cell
def _(d1_games, mo):
    d1_pick = mo.ui.dropdown(
        options=list(d1_games), value=list(d1_games)[0], label="Game"
    )
    d1_pick
    return (d1_pick,)


@app.cell
def _(d1_games, d1_pick, mo):
    d1_game = d1_games[d1_pick.value]
    d1_ply = mo.ui.slider(
        1, d1_game["n_plies"], value=d1_game["n_plies"], full_width=True,
        label=f"Ply (1 to {d1_game['n_plies']})",
    )
    d1_ply
    return d1_game, d1_ply


@app.cell
def _(C, chess, d1_game, d1_ply, mo):
    def _board():
        board = chess.Board()
        last = None
        for record in d1_game["ply"][: d1_ply.value]:
            last = chess.Move.from_uci(record["uci"])
            board.push(last)
        fill = {}
        if last is not None and d1_ply.value in d1_game["substituted_plies"]:
            fill[last.to_square] = C["warn"] + "66"
        return str(
            chess.svg.board(board, size=330, lastmove=last, fill=fill, coordinates=True)
        )

    _row = d1_game["ply"][d1_ply.value - 1]
    _sub = d1_ply.value in d1_game["substituted_plies"]
    _run = d1_game["running"][d1_ply.value - 1]
    _mover = "White" if d1_ply.value % 2 == 1 else "Black"
    _live_w = _run.get("white_score")
    _live_b = _run.get("black_score")

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.Html(_board()),
                    mo.md(
                        f"**Ply {d1_ply.value}: {_mover} plays {_row['san']}**"
                        + ("  &nbsp; <span style='color:" + C["warn"] + "'>engine move</span>"
                           if _sub else "")
                    ),
                ]
            ),
            mo.vstack(
                [
                    mo.md(
                        f"|  | White | Black |\n|---|---|---|\n"
                        f"| Stated rating | {d1_game['white_baseline']:.0f} "
                        f"| {d1_game['black_baseline']:.0f} |\n"
                        f"| Estimate at this ply | {_row['white_rating']:.0f} "
                        f"| {_row['black_rating']:.0f} |\n"
                        f"| Estimate at the end | {d1_game['final']['white_rating']:.0f} "
                        f"| {d1_game['final']['black_rating']:.0f} |\n"
                        f"| Live suspicion score here | "
                        f"{'n/a' if _live_w is None else format(_live_w, '.3f')} | "
                        f"{'n/a' if _live_b is None else format(_live_b, '.3f')} |\n"
                        f"| Final suspicion score | "
                        f"{d1_game['final']['white_score']:.3f} | "
                        f"{d1_game['final']['black_score']:.3f} |\n"
                        f"| Final label | {d1_game['final']['white_label'].replace('_', ' ')} "
                        f"| {d1_game['final']['black_label'].replace('_', ' ')} |"
                    ),
                    mo.md(
                        f"Clock left: **{_row['clock_seconds']} s**"
                        f" &nbsp; Attention weight: **{_row['attention']:.4f}**"
                        f" &nbsp; White deviation: **{_row['white_deviation']:.0f}** rating points"
                    ),
                ]
            ),
        ],
        widths=[1, 1],
        gap=2,
    )
    return


@app.cell
def _(C, alt, d1_game, d1_ply, pd, style):
    def _charts():
        rows = pd.DataFrame(d1_game["ply"])
        marker = pd.DataFrame({"ply": [d1_ply.value]})
        subs = pd.DataFrame({"ply": d1_game["substituted_plies"]})
        width = 700

        rule = (
            alt.Chart(marker).mark_rule(color=C["muted"], strokeWidth=2)
            .encode(x="ply:Q")
        )
        sub_ticks = (
            alt.Chart(subs).mark_rule(color=C["warn"], strokeWidth=1, opacity=0.25)
            .encode(x="ply:Q")
        )
        span = alt.Scale(domain=[1, d1_game["n_plies"]], nice=False)

        ratings = rows.melt(
            id_vars=["ply", "san"],
            value_vars=["white_rating", "black_rating"],
            var_name="side", value_name="rating",
        )
        ratings["side"] = ratings["side"].map(
            {"white_rating": "White", "black_rating": "Black"}
        )
        rating_chart = (
            alt.Chart(ratings).mark_line(strokeWidth=2)
            .encode(
                x=alt.X("ply:Q", title=None, scale=span),
                strokeDash=alt.StrokeDash(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[[1, 0], [5, 3]]),
                ),
                y=alt.Y("rating:Q", title="Rating estimate",
                        scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[C["a"], C["b"]]),
                ),
                tooltip=["ply", "san", "side", alt.Tooltip("rating:Q", format=".0f")],
            )
            .properties(width=width, height=130,
                        title="Rating estimate per ply, whole-game pass "
                              "(the two lines are not independent)")
        )

        attn = (
            alt.Chart(rows).mark_bar(width=max(2, int(width / len(rows)) - 1))
            .encode(
                x=alt.X("ply:Q", title=None, scale=span),
                y=alt.Y("attention:Q", title="Attention"),
                color=alt.Color("attention:Q", scale=alt.Scale(scheme="blues"),
                                legend=None),
                tooltip=["ply", "san", alt.Tooltip("attention:Q", format=".4f")],
            )
            .properties(width=width, height=80,
                        title="How much weight each ply receives")
        )

        dev = rows.melt(
            id_vars=["ply", "san"],
            value_vars=["white_deviation", "black_deviation"],
            var_name="side", value_name="deviation",
        )
        dev["side"] = dev["side"].map(
            {"white_deviation": "White", "black_deviation": "Black"}
        )
        dev_chart = (
            alt.Chart(dev).mark_line(strokeWidth=2)
            .encode(
                x=alt.X("ply:Q", title=None, scale=span),
                strokeDash=alt.StrokeDash(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[[1, 0], [5, 3]]),
                ),
                y=alt.Y("deviation:Q", title="Deviation"),
                color=alt.Color(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[C["a"], C["b"]]),
                ),
                tooltip=["ply", "san", "side", alt.Tooltip("deviation:Q", format=".0f")],
            )
            .properties(width=width, height=90,
                        title="Distance from the stated rating, in rating points")
        )

        run = pd.DataFrame(
            [r for r in d1_game["running"] if r.get("white_score") is not None]
        )
        run_long = run.melt(
            id_vars="ply", value_vars=["white_score", "black_score"],
            var_name="side", value_name="score",
        )
        run_long["side"] = run_long["side"].map(
            {"white_score": "White", "black_score": "Black"}
        )
        cuts = pd.DataFrame(
            [
                {"y": d1_game["cutoffs"]["p75"], "band": "Typical, 75th percentile"},
                {"y": d1_game["cutoffs"]["p95"], "band": "Highly unusual, 95th percentile"},
            ]
        )
        run_chart = (
            alt.Chart(run_long).mark_line(strokeWidth=2)
            .encode(
                x=alt.X("ply:Q", title="Ply", scale=span),
                strokeDash=alt.StrokeDash(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[[1, 0], [5, 3]]),
                ),
                y=alt.Y("score:Q", title="Suspicion score",
                        scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "side:N", title=None,
                    scale=alt.Scale(domain=["White", "Black"], range=[C["a"], C["b"]]),
                ),
                tooltip=["ply", "side", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(width=width, height=130,
                        title="Running suspicion score, rescored from ply 1 each time")
        )
        cut_rules = (
            alt.Chart(cuts).mark_rule(color=C["muted"], strokeDash=[4, 3])
            .encode(y="y:Q", tooltip=["band", alt.Tooltip("y:Q", format=".3f")])
        )

        return style(
            alt.vconcat(
                rating_chart + sub_ticks + rule,
                attn + sub_ticks + rule,
                dev_chart + sub_ticks + rule,
                run_chart + cut_rules + sub_ticks + rule,
                spacing=26,
            ).resolve_scale(color="independent", strokeDash="independent")
        )

    _charts()
    return


@app.cell
def _(d1_game, d1_data, flag, mo, notes, src):
    mo.vstack(
        [
            flag(
                "Red ticks mark the plies where an engine move was planted. "
                f"This game has {len(d1_game['substituted_plies'])} of them.",
                "warn" if d1_game["substituted_plies"] else "note",
            ),
            notes(
                "The two rating lines sit on top of each other because the network "
                "produces one shared estimate for the game. Diagram 8 measures that.",
                "The whole-game pass lets every ply see the entire game, since the "
                "network reads the sequence in both directions. The running score "
                "rescores the game from ply 1 up to ply t, which is what a live "
                "viewer would have seen at that moment. The two are different "
                "computations and only agree at the last ply.",
                "The label compares this score with ordinary held-out test games at "
                "the same time control. It is a percentile, not a verdict on the "
                "player.",
                "The attention strip falls away toward the end of every game by "
                "construction, not because the model stops reading. Attention is "
                "masked so ply t may look only at plies 1 to t, and the weight drawn "
                "here is the average each ply receives across all the plies that can "
                "look at it, so a late ply has few of them.",
            ),
            src(
                "Games: " + d1_game["pgn_path"],
                "Scores: prototype/src/api.py through "
                + d1_data["checkpoint"].split("/")[-1]
                + " (the frozen tuned-attention checkpoint), method "
                + d1_data["score_method"],
                "Ground truth for the planted moves: "
                "prototype/src/static/samples/manifest.json",
                "Precomputed by presentation/build_data.py",
            ),
        ]
    )
    return


# ==========================================================================
# 2. Anomaly detection, from zero
# ==========================================================================


@app.cell
def _(head):
    head(
        2,
        "Anomaly detection, from zero",
        "What a suspicion score is, what an AUC of 0.75 means, why checking more "
        "of a player's games helps, and how much cheating the detector can "
        "actually see.",
    )
    return


@app.cell
def _(load):
    d2 = load("d2_anomaly.json")
    return (d2,)


@app.cell
def _(mo):
    mo.md(
        """
        ### 2a. One game in, one number out

        Every game gets a score between 0 and 1. Score thousands of games and the
        scores pile into two hills: honest games and games with engine moves. Where
        the hills overlap, the detector cannot tell them apart, so less overlap is a
        better detector. The **AUC** is exactly this: take one random engine game
        and one random honest game, and the AUC is how often the engine game gets
        the higher score.
        """
    )
    return


@app.cell
def _(mo):
    d2_auc_demo = mo.ui.slider(
        0.50, 0.99, step=0.01, value=0.75, full_width=True,
        label="Drag an AUC and watch the two hills pull apart",
    )
    d2_auc_demo
    return (d2_auc_demo,)


@app.cell
def _(C, alt, d2_auc_demo, np, pd, style):
    def _hills():
        # Two unit-variance bells whose separation d gives exactly this AUC.
        # AUC = Phi(d / sqrt(2)), so d = sqrt(2) * Phi_inv(AUC).
        from math import erf, sqrt

        def phi_inv(p):
            lo, hi = -8.0, 8.0
            for _ in range(80):
                mid = (lo + hi) / 2
                if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2

        d = sqrt(2) * phi_inv(d2_auc_demo.value)
        xs = np.linspace(-4, 4 + d, 400)
        frame = pd.concat(
            [
                pd.DataFrame({"x": xs, "density": np.exp(-0.5 * xs ** 2),
                              "group": "Honest games"}),
                pd.DataFrame({"x": xs, "density": np.exp(-0.5 * (xs - d) ** 2),
                              "group": "Games with engine moves"}),
            ]
        )
        return style(
            alt.Chart(frame).mark_area(opacity=0.45, line={"strokeWidth": 2})
            .encode(
                x=alt.X("x:Q", title="Less suspicious                    more suspicious",
                        axis=alt.Axis(labels=False, ticks=False, grid=False)),
                y=alt.Y("density:Q", title=None, stack=False,
                        axis=alt.Axis(labels=False, ticks=False, grid=False)),
                color=alt.Color(
                    "group:N", title=None,
                    scale=alt.Scale(
                        domain=["Honest games", "Games with engine moves"],
                        range=[C["a"], C["warn"]],
                    ),
                ),
            )
            .properties(width=620, height=180,
                        title=f"AUC {d2_auc_demo.value:.2f}: the engine game scores "
                              f"higher {d2_auc_demo.value * 100:.0f} times in 100")
        )

    _hills()
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### 2b. Checking more of a player's games, and part-time cheating

        One game is noisy: a cheater can have an ordinary game and an honest player
        a brilliant one. Averaging a player's games cancels much of that luck, and
        cheating in only some games waters the average back down.
        """
    )
    return


@app.cell
def _(mo):
    d2_k = mo.ui.slider(
        steps=[1, 5, 10, 20, 50, 100], value=20, label="Games checked per player"
    )
    d2_frac = mo.ui.slider(
        steps=[10, 25, 50, 75, 100], value=100,
        label="Share of their games the player cheats in (%)",
    )
    d2_det = mo.ui.dropdown(
        options={"Sequence detector": "sequence",
                 "Summary features (LightGBM)": "lightgbm",
                 "No training (S_att)": "s_att"},
        value="Sequence detector", label="Detector",
    )
    mo.hstack([d2_k, d2_frac, d2_det], justify="start", gap=2)
    return d2_det, d2_frac, d2_k


@app.cell
def _(C, alt, d2, d2_det, d2_frac, d2_k, mo, pd, stat, style):
    def _measured_mix():
        rows = [
            r for r in d2["mixed_cheater"]["by_detector"][d2_det.value]
        ]
        frame = pd.DataFrame(rows)
        frame["share"] = (frame["f"] * 100).round().astype(int).astype(str) + "%"
        frame["k_label"] = frame["k"].astype(str)
        k_order = [str(k) for k in d2["mixed_cheater"]["ks"]]
        share_order = [f"{int(round(f * 100))}%" for f in d2["mixed_cheater"]["fracs"]]
        picked = f"{d2_frac.value}%"
        y_scale = alt.Scale(domain=[0.45, 1.0], nice=False)
        x_enc = alt.X("k_label:N", title="Games checked per player", sort=k_order,
                      axis=alt.Axis(labelAngle=0))

        faint = (
            alt.Chart(frame[frame["share"] != picked])
            .mark_line(strokeWidth=1.5, opacity=0.3, color=C["muted"])
            .encode(x=x_enc, y=alt.Y("auc:Q", scale=y_scale), detail="share:N")
        )
        chosen = frame[frame["share"] == picked]
        band = (
            alt.Chart(chosen).mark_area(opacity=0.15, color=C["a"])
            .encode(x=x_enc, y=alt.Y("lo:Q", scale=y_scale), y2="hi:Q")
        )
        line = (
            alt.Chart(chosen).mark_line(strokeWidth=2.5, color=C["a"],
                                        point={"size": 70, "color": C["a"]})
            .encode(
                x=x_enc,
                y=alt.Y("auc:Q", title="Player-level AUC", scale=y_scale),
                tooltip=["share", "k", alt.Tooltip("auc:Q", format=".3f"),
                         alt.Tooltip("lo:Q", format=".3f"),
                         alt.Tooltip("hi:Q", format=".3f"),
                         alt.Tooltip("n_cheat_games:Q",
                                     title="Cheating games in the hand")],
            )
        )
        here = chosen[chosen["k"] == d2_k.value].iloc[0]
        marker = (
            alt.Chart(pd.DataFrame([here])).mark_point(size=180, filled=True,
                                                       color=C["warn"])
            .encode(x=x_enc, y=alt.Y("auc:Q", scale=y_scale))
        )
        chance = (
            alt.Chart(frame).mark_rule(color=C["muted"], strokeDash=[4, 3])
            .encode(y=alt.Y("half:Q", scale=y_scale))
            .transform_calculate(half="0.5")
        )
        # The bell-curve model on the same axes, so the gap is visible.
        from math import erf, sqrt

        D = d2["toy_model"]["D"]
        rho = d2["toy_model"]["rho"]
        toy = []
        for k in d2["mixed_cheater"]["ks"]:
            var = (1 - rho) / k + rho
            shift = (d2_frac.value / 100.0) * D / sqrt(var)
            toy.append({"k_label": str(k), "auc": 0.5 * (1 + erf(shift / 2.0))})
        toy_line = (
            alt.Chart(pd.DataFrame(toy))
            .mark_line(strokeWidth=2, strokeDash=[6, 4], color=C["b"])
            .encode(x=x_enc, y=alt.Y("auc:Q", scale=y_scale),
                    tooltip=[alt.Tooltip("auc:Q", format=".3f",
                                         title="bell-curve picture")])
        )
        chart = style(
            (faint + band + line + toy_line + marker + chance).properties(
                width=560, height=280,
                title="Measured player-level AUC, withheld Maia bands",
            )
        )
        return mo.vstack(
            [
                chart,
                mo.hstack(
                    [
                        stat("Measured AUC", f"{here['auc']:.3f}",
                             f"95 percent interval [{here['lo']:.3f}, {here['hi']:.3f}]"),
                        stat("Cheating games in the hand",
                             f"{int(here['n_cheat_games'])} of {d2_k.value}",
                             f"realized share {here['f_eff'] * 100:.0f}%"),
                        stat("Bell-curve picture", f"{toy[k_order.index(str(d2_k.value))]['auc']:.3f}",
                             "the dashed line, for intuition only"),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _measured_mix()
    return


@app.cell
def _(d2, flag, mo, notes, src):
    mo.vstack(
        [
            flag(
                "Post hoc: a check on simulated players, run after the pre-registered "
                "player-level results were already reported.",
                "warn",
            ),
            notes(
                d2["mixed_cheater_note"],
                d2["mixed_cheater_rounding"],
                "The dashed line is a bell-curve picture, not a measurement. "
                + d2["toy_model"]["known_shortfall"],
                "Players are simulated by pooling the detector's saved per-game scores: "
                f"{d2['mixed_cheater']['players_per_cheater_cell']} cheating players per "
                f"cell and {d2['mixed_cheater']['players_per_clean_band']} clean players "
                f"per band, seed {d2['mixed_cheater']['seed']}, "
                f"{d2['mixed_cheater']['bootstrap_reps']}-replicate hierarchical bootstrap.",
            ),
            src(
                "analysis/mixed-cheater-pooling-results.md",
                "analysis/mixed_cheater_pooling/results.json",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("### 2c. The measured curves")
    return


@app.cell
def _(mo):
    d2_group = mo.ui.dropdown(
        options={
            "Maia bands withheld from training": "withheld_band",
            "Bands and rates seen in training": "seen",
            "Substitution rate withheld from training": "withheld_rate",
            "Both withheld": "withheld_both",
        },
        value="Maia bands withheld from training",
        label="Evaluation group",
    )
    d2_show_max = mo.ui.switch(label="Also show the worst-game rule")
    d2_show_hard = mo.ui.switch(label="Also show the strong honest players check")
    mo.hstack([d2_group, d2_show_max, d2_show_hard], justify="start", gap=2)
    return d2_group, d2_show_hard, d2_show_max


@app.cell
def _(C, alt, d2, d2_group, d2_show_hard, d2_show_max, pd, style):
    def _measured():
        key = d2_group.value
        ks = d2["k_values"]
        rows = [
            {"k": k, "auc": v[0], "lo": v[1], "hi": v[2], "series": "Average of the games"}
            for k, v in zip(ks, d2["player_level_mean"][key])
        ]
        if d2_show_max.value:
            rows += [
                {"k": k, "auc": v, "lo": None, "hi": None,
                 "series": "Worst single game"}
                for k, v in zip(ks, d2["player_level_max"][key])
            ]
        if d2_show_hard.value and key in d2["hard_negative_mean"]:
            rows += [
                {"k": k, "auc": v, "lo": None, "hi": None,
                 "series": "Strong honest players read as cheaters"}
                for k, v in zip(ks, d2["hard_negative_mean"][key])
            ]
        frame = pd.DataFrame(rows)
        order = ["Average of the games", "Worst single game",
                 "Strong honest players read as cheaters"]
        domain = [s for s in order if s in set(frame["series"])]
        colors = {order[0]: C["a"], order[1]: C["b"], order[2]: C["c"]}

        frame["k_label"] = frame["k"].astype(str)
        k_order = [str(k) for k in ks]
        y_scale = alt.Scale(domain=[0.45, 1.0], nice=False)
        x_enc = alt.X("k_label:N", title="Games checked per player", sort=k_order,
                      axis=alt.Axis(labelAngle=0))
        base = alt.Chart(frame).encode(
            x=x_enc,
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=domain,
                                            range=[colors[s] for s in domain])),
        )
        line = base.mark_line(strokeWidth=2, point={"size": 70}).encode(
            y=alt.Y("auc:Q", title="Player-level AUC", scale=y_scale),
            strokeDash=alt.StrokeDash("series:N", title=None,
                                      scale=alt.Scale(domain=domain,
                                                      range=[[1, 0], [5, 3], [2, 2]][:len(domain)])),
            tooltip=["series", "k", alt.Tooltip("auc:Q", format=".3f"),
                     alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
        )
        band = (
            alt.Chart(frame[frame["lo"].notna()])
            .mark_area(opacity=0.15, color=C["a"])
            .encode(x=x_enc, y=alt.Y("lo:Q", scale=y_scale), y2="hi:Q")
        )
        chance = (
            alt.Chart(frame).mark_rule(color=C["muted"], strokeDash=[4, 3])
            .encode(y=alt.Y("datum_half:Q", scale=y_scale))
            .transform_calculate(datum_half="0.5")
        )
        return style(
            (band + line + chance).properties(
                width=620, height=280,
                title="Sequence detector, pooling one player's games",
            )
        )

    _measured()
    return


@app.cell
def _(d2, flag, mo, notes, src):
    mo.vstack(
        [
            flag("Post hoc: " + d2["post_hoc_note"], "warn"),
            flag(d2["synthetic_note"], "warn"),
            notes(d2["mean_vs_max"]),
            src(
                "CCS Thesis - Integrated/chapters/chapter4.tex, player-level "
                "aggregation of the sequence detector",
                "analysis/a3-player-level-results.md, for the worst-game rule and "
                "the strength check",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### 2d. How much cheating can it see?

        The substitution rate is the share of the cheater's moves that were engine
        moves. The trained detector climbs as cheating gets heavier; the score that
        needed no training stays on the coin-flip line at every level.
        """
    )
    return


@app.cell
def _(mo):
    d2_rate_group = mo.ui.dropdown(
        options={"Maia bands withheld from training": "withheld_band",
                 "Bands seen in training": "seen"},
        value="Maia bands withheld from training",
        label="Evaluation group",
    )
    d2_rate_group
    return (d2_rate_group,)


@app.cell
def _(C, alt, d2, d2_rate_group, pd, style):
    def _by_rate():
        key = d2_rate_group.value
        rows = []
        for method, label in [("detector", "Summary features (LightGBM)"),
                              ("s_att", "No training (S_att)")]:
            for rate, v in zip(d2["substitution_rates_pct"],
                               d2["by_substitution_rate"][method][key]):
                rows.append({"rate": f"{rate}%", "rate_n": rate, "auc": v[0],
                             "lo": v[1], "hi": v[2], "method": label})
        frame = pd.DataFrame(rows)
        domain = ["Summary features (LightGBM)", "No training (S_att)"]

        base = alt.Chart().encode(
            x=alt.X("method:N", title=None, axis=None),
            color=alt.Color("method:N", title=None,
                            scale=alt.Scale(domain=domain, range=[C["a"], C["b"]])),
        )
        y_scale = alt.Scale(domain=[0.45, 0.88], nice=False)
        # Bars anchored at chance rather than at zero: for an AUC, 0.5 is the
        # meaningful baseline, and a bar from zero would hide every difference.
        stems = base.mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
            stroke="#fcfcfb", strokeWidth=2, width=22,
        ).encode(
            y=alt.Y("auc:Q", title="ROC-AUC", scale=y_scale),
            y2=alt.datum(0.5),
        )
        bars = base.mark_point(filled=True, size=0).encode(
            y=alt.Y("auc:Q", title="ROC-AUC", scale=y_scale),
            tooltip=["method", "rate", alt.Tooltip("auc:Q", format=".3f"),
                     alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")],
        )
        whisk = base.mark_rule(strokeWidth=1.5, color=C["ink"]).encode(
            y=alt.Y("lo:Q", scale=y_scale), y2="hi:Q")
        labels = base.mark_text(dy=-9, baseline="bottom", fontSize=10,
                                color=C["ink"]).encode(
            y=alt.Y("hi:Q", scale=y_scale), text=alt.Text("auc:Q", format=".3f")
        )
        chance = (
            alt.Chart().mark_rule(color=C["muted"], strokeDash=[4, 3])
            .encode(y=alt.datum(0.5))
        )
        return style(
            alt.layer(stems, bars, whisk, labels, chance, data=frame)
            .properties(width=95, height=230)
            .facet(column=alt.Column("rate:N", title="Share of the cheater's moves replaced",
                                     sort=["2%", "5%", "20%", "40%", "60%"],
                                     header=alt.Header(labelColor=C["ink"],
                                                       titleColor=C["ink"])))
            .properties(title="Detection by how heavily the game was substituted")
        )

    _by_rate()
    return


@app.cell
def _(d2, flag, mo, notes, src):
    mo.vstack(
        [
            flag(
                "At the 2 and 5 percent rates, the rates that describe occasional "
                "cheating, the trained detector is close to a coin flip. That is the "
                "weakest part of the result."
            ),
            notes(
                "The dashed line is chance, and the bars are drawn from it rather than "
                "from zero, since an AUC below 0.5 is worse than guessing. Whiskers are "
                "95 percent confidence intervals.",
                "This is the summary-feature detector, which is not the sequence "
                "detector shown above it. On withheld bands at the 2 percent rate the "
                "sequence detector reaches 0.576 where this one reaches 0.539.",
            ),
            src(
                "CCS Thesis - Integrated/chapters/chapter4.tex, supervised detector "
                "and computed score by substitution rate",
                "Every value comes from synthetic games: "
                + d2["synthetic_note"].split(": ", 1)[1],
            ),
        ]
    )
    return


# ==========================================================================
# 3. From Lichess to training data
# ==========================================================================


@app.cell
def _(head):
    head(
        3,
        "From Lichess to training data",
        "How 2,550,000 public games were sampled, filtered, stored and split.",
    )
    return


@app.cell
def _(load):
    d3 = load("d3_corpus.json")
    return (d3,)


@app.cell
def _(d3, mo, stat):
    mo.hstack(
        [
            stat("Games", f"{d3['n_games']:,}", f"{d3['per_month']:,} a month"),
            stat("Months", str(d3["n_months"]), "January 2019 to July 2026"),
            stat("Plies kept", str(d3["max_plies"]),
                 f"{d3['overall_pct_truncated']:.1f}% of games hit the cap"),
            stat("Stored as", f"{d3['storage']['compact_store_gb']} GB",
                 f"against {d3['storage']['loose_pickles_gb']} GB loose"),
        ],
        justify="start", gap=1,
    )
    return


@app.cell
def _(mo):
    d3_metric = mo.ui.dropdown(
        options={"Share of the corpus": "pct_of_corpus",
                 "Share cut off at 100 plies": "pct_truncated"},
        value="Share of the corpus", label="Show",
    )
    d3_metric
    return (d3_metric,)


@app.cell
def _(C, alt, d3, d3_metric, pd, style):
    def _composition():
        proper = {"ultrabullet": "UltraBullet", "bullet": "Bullet", "blitz": "Blitz",
                  "rapid": "Rapid", "classical": "Classical"}
        frame = pd.DataFrame(d3["by_time_control"])
        frame["label"] = frame["name"].map(proper)
        value = d3_metric.value
        bars = (
            alt.Chart(frame).mark_bar(cornerRadiusEnd=4, color=C["a"])
            .encode(
                y=alt.Y("label:N", title=None, sort="-x"),
                x=alt.X(f"{value}:Q", title="Percent"),
                tooltip=["label", alt.Tooltip("count:Q", format=","),
                         alt.Tooltip("pct_of_corpus:Q", format=".2f"),
                         alt.Tooltip("pct_truncated:Q", format=".2f")],
            )
        )
        text = (
            alt.Chart(frame).mark_text(align="left", dx=4, fontSize=11, color=C["ink"])
            .encode(y=alt.Y("label:N", sort="-x"), x=f"{value}:Q",
                    text=alt.Text(f"{value}:Q", format=".2f"))
        )
        return style(
            (bars + text).properties(width=520, height=190,
                                     title="Realized corpus composition by time control")
        )

    _composition()
    return


@app.cell
def _(mo):
    d3_seen = mo.ui.slider(
        1000, 200000, step=1000, value=60000, full_width=True,
        label="Eligible games streamed from the month so far",
    )
    d3_seen
    return (d3_seen,)


@app.cell
def _(C, alt, d3, d3_seen, mo, np, pd, stat, style):
    def _reservoir():
        k = d3["reservoir"]["k"]
        seen = max(d3_seen.value, k)
        chance = k / seen
        # A 30-slot picture of the 30,000-slot reservoir, drawn the way the real
        # rule behaves: give every game a random key once, then keep the 30
        # smallest keys among the games seen so far. Dragging the slider replaces
        # some occupants and leaves others, exactly as the real sampler does, and
        # the arrival positions stay spread evenly across the month.
        keys = np.random.default_rng(7).random(200_000)
        kept = np.argpartition(keys[:seen], 30)[:30]
        kept.sort()
        slots = pd.DataFrame(
            {
                "slot": range(30),
                "row": [i // 10 for i in range(30)],
                "col": [i % 10 for i in range(30)],
                "arrived": kept.astype(float),
            }
        )
        grid = (
            alt.Chart(slots).mark_rect(cornerRadius=3, stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x=alt.X("col:O", title=None, axis=None),
                y=alt.Y("row:O", title=None, axis=None),
                color=alt.Color("arrived:Q", scale=alt.Scale(scheme="blues"),
                                title="Arrived at game number"),
                tooltip=[alt.Tooltip("arrived:Q", format=",.0f", title="Arrived at game")],
            )
            .properties(width=300, height=100,
                        title="The 30 kept games, shaded by when in the month they arrived")
        )
        return mo.hstack(
            [
                style(grid),
                stat("Chance any one game of the month is kept",
                     f"{chance * 100:.1f}%",
                     f"{k:,} kept out of {seen:,} eligible so far"),
            ],
            justify="start", gap=2, align="center",
        )

    _reservoir()
    return


@app.cell
def _(C, alt, d3, mo, pd, style):
    def _split():
        frame = pd.DataFrame(
            [
                {"part": "Training", "pct": d3["split"]["train_pct"], "order": 0},
                {"part": "Validation", "pct": d3["split"]["val_pct"], "order": 1},
                {"part": "Test", "pct": d3["split"]["test_pct"], "order": 2},
            ]
        )
        frame["games"] = (frame["pct"] / 100 * d3["n_games"]).astype(int)
        chart = (
            alt.Chart(frame).mark_bar(cornerRadius=3, stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x=alt.X("pct:Q", title=None, stack="zero",
                        axis=alt.Axis(labels=False, ticks=False, grid=False)),
                color=alt.Color(
                    "part:N", title=None, sort=["Training", "Validation", "Test"],
                    scale=alt.Scale(domain=["Training", "Validation", "Test"],
                                    range=[C["a"], C["b"], C["c"]]),
                ),
                order="order:Q",
                tooltip=["part", "pct", alt.Tooltip("games:Q", format=",")],
            )
            .properties(width=620, height=46, title="The 72 / 18 / 10 split")
        )
        steps = "".join(
            f"<li style='margin-bottom:3px'>{s}</li>" for s in d3["eligibility"]
        )
        return mo.vstack(
            [
                style(chart),
                mo.md(
                    "<div style='font-size:0.86rem'><b>A game has to clear every "
                    f"filter to reach the corpus</b><ol>{steps}</ol></div>"
                ),
            ]
        )

    _split()
    return


@app.cell
def _(d3, mo, notes, src):
    mo.vstack(
        [
            notes(d3["reservoir"]["note"], d3["storage"]["note"], d3["split"]["note"],
                  d3["after_the_filters"]),
            src(
                "analysis/heldout_test_eval/corpus_composition.json",
                "CCS Thesis - Integrated/chapters/chapter3.tex, sampling and split",
                "analysis/corpus-storage-solution.md, compact-store sizing",
            ),
        ]
    )
    return


# ==========================================================================
# 4. How a move becomes numbers
# ==========================================================================


@app.cell
def _(head):
    head(
        4,
        "How a move becomes numbers",
        "The network never sees a chessboard. It sees twelve 8 by 8 grids of ones "
        "and zeros, one per piece type and color, plus one standardized clock "
        "reading.",
    )
    return


@app.cell
def _(d1_games, mo):
    d4_pick = mo.ui.dropdown(options=list(d1_games), value=list(d1_games)[0], label="Game")
    d4_pick
    return (d4_pick,)


@app.cell
def _(d1_games, d4_pick, mo):
    d4_game = d1_games[d4_pick.value]
    d4_ply = mo.ui.slider(1, d4_game["n_plies"], value=min(12, d4_game["n_plies"]),
                          full_width=True, label="Ply")
    d4_ply
    return d4_game, d4_ply


@app.cell
def _(C, alt, chess, d4_game, d4_ply, load, mo, pd, stat, style):
    def _planes():
        arch = load("d5_architecture.json")
        board = chess.Board(d4_game["fens"][d4_ply.value - 1])
        names = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]
        order = [("White " if i < 6 else "Black ") + names[i % 6] for i in range(12)]
        occupied = set()
        for square, piece in board.piece_map().items():
            plane = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            occupied.add((plane, chess.square_file(square), 7 - chess.square_rank(square)))
        # Every square of every plane, so each plane reads as an 8 by 8 grid of
        # ones and zeros rather than as loose dots.
        rows = [
            {"name": order[plane], "file": f, "rank": r,
             "value": 1 if (plane, f, r) in occupied else 0}
            for plane in range(12) for f in range(8) for r in range(8)
        ]
        frame = pd.DataFrame(rows)
        grid = (
            alt.Chart(frame).mark_rect(stroke="#fcfcfb", strokeWidth=0.5)
            .encode(
                x=alt.X("file:O", title=None, axis=None,
                        scale=alt.Scale(domain=list(range(8)))),
                y=alt.Y("rank:O", title=None, axis=None,
                        scale=alt.Scale(domain=list(range(8)))),
                color=alt.Color("value:Q", legend=None,
                                scale=alt.Scale(domain=[0, 1],
                                                range=["#eeede9", C["a"]])),
                tooltip=["name", alt.Tooltip("value:Q", title="Plane value")],
            )
            .properties(width=78, height=78)
            .facet(facet=alt.Facet("name:N", title=None, sort=order,
                                   header=alt.Header(labelFontSize=10,
                                                     labelColor=C["ink"])),
                   columns=6)
            .properties(title="The twelve piece planes at this ply")
        )

        clock = d4_game["ply"][d4_ply.value - 1]["clock_seconds"]
        mean = arch["normalization"]["clocks_mean"]
        std = arch["normalization"]["clocks_std"]
        z = (clock - mean) / std
        return mo.vstack(
            [
                style(grid),
                mo.hstack(
                    [
                        stat("Clock reading", f"{clock} s", "from the game's own notation"),
                        stat("Standardized", f"{z:+.3f}",
                             f"({clock} minus {mean}) divided by {std}"),
                        stat("What the network reads", "12 x 8 x 8 + 1",
                             "769 numbers for this one ply"),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _planes()
    return


@app.cell
def _(src):
    src(
        "prototype/src/format_data.py, board_to_array",
        "prototype/src/chess_rating_net.py, the dataset's clock mean 273 and "
        "standard deviation 380",
        "Positions come from the same bundled game as diagram 1",
    )
    return


# ==========================================================================
# 5. The architecture
# ==========================================================================


@app.cell
def _(head):
    head(
        5,
        "The architecture, stage by stage",
        "Pick a stage to see what it does and the shapes going in and out, taken "
        "from a real forward pass over a 40-ply game.",
    )
    return


@app.cell
def _(load):
    d5 = load("d5_architecture.json")
    return (d5,)


@app.cell
def _(d5, mo):
    d5_pick = mo.ui.radio(
        options={s["name"]: s["id"] for s in d5["stages"]},
        value=d5["stages"][0]["name"], inline=True, label="Stage",
    )
    d5_pick
    return (d5_pick,)


@app.cell
def _(C, d5, d5_pick, mo, stat):
    def _diagram():
        boxes = []
        for stage in d5["stages"]:
            on = stage["id"] == d5_pick.value
            boxes.append(
                "<div style='flex:1;min-width:120px;padding:10px 12px;border-radius:8px;"
                f"border:2px solid {C['a'] if on else '#e3e2dd'};"
                f"background:{C['a'] + '14' if on else 'transparent'}'>"
                f"<div style='font-size:0.78rem;opacity:0.65'>{stage['out']}</div>"
                f"<div style='font-weight:700;font-size:0.95rem'>{stage['name']}</div>"
                f"<div style='font-size:0.78rem;opacity:0.75'>{stage['role']}</div></div>"
            )
        arrow = ("<div style='align-self:center;color:" + C["muted"]
                 + ";font-size:1.2rem'>&#8594;</div>")
        chain = arrow.join(boxes)
        current = next(s for s in d5["stages"] if s["id"] == d5_pick.value)
        params = d5["n_parameters"]
        key = {"cnn": "cnn", "lstm": "lstm", "attention": "attention",
               "head": "rating_head"}[current["id"]]
        return mo.vstack(
            [
                mo.md(f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{chain}</div>"),
                mo.md(f"**{current['name']}.** {current['detail']}"),
                mo.hstack(
                    [
                        stat("Input", current["in"], "batch of 1, 40 plies"),
                        stat("Output", current["out"], ""),
                        stat("Weights in this stage", f"{params[key]:,}",
                             f"of {params['total']:,} in the whole network"),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _diagram()
    return


@app.cell
def _(d5, flag, mo, src):
    mo.vstack(
        [
            flag(
                "Rating estimates leave the network standardized. They are turned "
                f"back into rating points with mean {d5['normalization']['ratings_mean']} "
                f"and standard deviation {d5['normalization']['ratings_std']}."
            ),
            src(
                "prototype/src/chess_rating_net.py",
                "Shapes and weight counts recorded from a real forward pass by "
                "presentation/build_data.py",
                "Settings: analysis/heldout_test_eval/attn_tuned__best.json",
            ),
        ]
    )
    return


# ==========================================================================
# 6. Did attention help?
# ==========================================================================


@app.cell
def _(head):
    head(
        6,
        "Did attention help? The paired bootstrap",
        "Resample all 255,000 test games with replacement, take the mean "
        "difference, and repeat. Drag the slider to build the distribution.",
    )
    return


@app.cell
def _(load):
    d6 = load("d6_bootstrap.json")
    return (d6,)


@app.cell
def _(d6, mo):
    d6_n = mo.ui.slider(
        steps=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
        value=100, label="Resamples drawn", full_width=True,
    )
    mo.vstack([d6_n, mo.md(f"*{d6['method']}*")])
    return (d6_n,)


@app.cell
def _(C, alt, d6, d6_n, mo, pd, stat, style):
    def _bootstrap():
        draws = d6["resampled_means"][: d6_n.value]
        frame = pd.DataFrame({"delta": draws})
        lo = sorted(draws)[max(0, int(0.025 * len(draws)) - 1)]
        hi = sorted(draws)[min(len(draws) - 1, int(0.975 * len(draws)))]

        hist = (
            alt.Chart(frame).mark_bar(color=C["a"], cornerRadiusTopLeft=2,
                                      cornerRadiusTopRight=2)
            .encode(
                x=alt.X("delta:Q", bin=alt.Bin(maxbins=45),
                        title="Mean difference in rating points, full model minus baseline",
                        scale=alt.Scale(domain=[-3.65, -2.5])),
                y=alt.Y("count()", title="Resamples"),
            )
            .properties(width=620, height=250)
        )
        recorded = pd.DataFrame(
            [{"x": d6["recorded"]["ci95_lo"], "what": "recorded 95% interval"},
             {"x": d6["recorded"]["ci95_hi"], "what": "recorded 95% interval"}]
        )
        rec_rules = (
            alt.Chart(recorded).mark_rule(color=C["warn"], strokeDash=[5, 3],
                                          strokeWidth=2)
            .encode(x="x:Q", tooltip=["what", alt.Tooltip("x:Q", format=".3f")])
        )
        point = (
            alt.Chart(pd.DataFrame({"x": [d6["recorded"]["point_delta_mae"]]}))
            .mark_rule(color=C["b"], strokeWidth=2).encode(x="x:Q")
        )
        return mo.vstack(
            [
                style(hist + rec_rules + point),
                mo.hstack(
                    [
                        stat("Point difference",
                             f"{d6['recorded']['point_delta_mae']:.2f}",
                             "rating points, full model minus baseline"),
                        stat("Recorded 95% interval",
                             f"[{d6['recorded']['ci95_lo']:.2f}, "
                             f"{d6['recorded']['ci95_hi']:.2f}]",
                             "clear of zero, so the difference is real"),
                        stat(f"Interval from these {len(draws):,} draws",
                             f"[{lo:.2f}, {hi:.2f}]",
                             "narrows toward the recorded one"),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _bootstrap()
    return


@app.cell
def _(C, alt, d6, pd, style):
    def _arms():
        rows = []
        for key, label in d6["arm_labels"].items():
            if key in d6["arm_mae"]:
                rows.append({"arm": label, "mae": d6["arm_mae"][key],
                             "kind": "Trained in this study"})
        frame = pd.DataFrame(rows).sort_values("mae")
        order = list(frame["arm"])
        x_scale = alt.Scale(domain=[169, 177], nice=False)
        base_mae = d6["arm_mae"]["baseline"]
        stems = (
            alt.Chart(pd.DataFrame([{"x": base_mae, "what": "reproduced baseline"}]))
            .mark_rule(color=C["muted"], strokeDash=[4, 3])
            .encode(x=alt.X("x:Q", scale=x_scale), tooltip=["what"])
        )
        dots = (
            alt.Chart(frame).mark_point(filled=True, size=140, color=C["a"])
            .encode(
                y=alt.Y("arm:N", title=None, sort=order),
                x=alt.X("mae:Q", title="Held-out test MAE in rating points",
                        scale=x_scale),
                tooltip=["arm", alt.Tooltip("mae:Q", format=".2f")],
            )
        )
        labels = (
            alt.Chart(frame).mark_text(align="left", dx=10, fontSize=11, color=C["ink"])
            .encode(y=alt.Y("arm:N", sort=order), x=alt.X("mae:Q", scale=x_scale),
                    text=alt.Text("mae:Q", format=".2f"))
        )
        return style(
            (stems + dots + labels).properties(
                width=440, height=150,
                title="Every full-corpus arm, measured against the reproduced baseline",
            )
        )

    _arms()
    return


@app.cell
def _(d6, flag, mo, notes, src):
    mo.vstack(
        [
            flag(
                "The confound control matters. Retraining the plain baseline at the "
                "higher learning rate alone accounts for "
                f"{abs(d6['other_pairs']['baseline_lr3e4_vs_baseline']['point_delta_mae']):.2f} "
                "of the improvement; attention adds "
                f"{abs(d6['other_pairs']['tuned_attention_vs_baseline_lr3e4']['point_delta_mae']):.2f} "
                "on top of that, and both intervals clear zero."
            ),
            notes(
                "Every arm is a single training run at training seed 0, so none of "
                "these points carries a seed-to-seed spread.",
                d6["recomputed"]["note"],
                "The dashed red lines are the recorded 95 percent interval and the "
                "solid line is the recorded point difference. Zero is off the chart to "
                "the right, which is what makes the difference significant.",
            ),
            src(
                "analysis/heldout_test_eval/bootstrap.json",
                "analysis/heldout_test_eval/attn_tuned__best.csv and "
                "baseline__best.csv, 255,000 per-game errors each",
                "CCS Thesis - Integrated/chapters/chapter4.tex, controlled ablation "
                "of the attention mechanism",
            ),
        ]
    )
    return


# ==========================================================================
# 7. Where the model is weaker
# ==========================================================================


@app.cell
def _(head):
    head(
        7,
        "Where the model is weaker",
        "The study set itself a threshold: no subgroup may exceed twice the "
        "overall error. One does.",
    )
    return


@app.cell
def _(load):
    d7 = load("d7_subgroups.json")
    return (d7,)


@app.cell
def _(mo):
    d7_by = mo.ui.dropdown(
        options={"Time control": "by_time_control", "Rating bracket": "by_rating_bracket"},
        value="Rating bracket", label="Break out by",
    )
    d7_limit = mo.ui.switch(label="Show the factor-of-two line", value=True)
    mo.hstack([d7_by, d7_limit], justify="start", gap=2)
    return d7_by, d7_limit


@app.cell
def _(C, alt, d7, d7_by, d7_limit, pd, style):
    def _subgroups():
        key = d7_by.value
        rows = []
        for r in d7[key]:
            label = r["name"].capitalize() if key == "by_time_control" else r["name"]
            rows.append({"group": label, "mae": r["baseline"],
                         "arm": "Reproduced baseline", "n": r["n"]})
            rows.append({"group": label, "mae": r["full_model"],
                         "arm": "Full model", "n": r["n"]})
        frame = pd.DataFrame(rows)
        order = [(r["name"].capitalize() if key == "by_time_control" else r["name"])
                 for r in d7[key]]
        domain = ["Reproduced baseline", "Full model"]

        bars = (
            alt.Chart(frame).mark_bar(cornerRadiusEnd=3, stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x=alt.X("group:N", title=None, sort=order,
                        axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("arm:N", sort=domain),
                y=alt.Y("mae:Q", title="Test MAE in rating points"),
                color=alt.Color("arm:N", title=None,
                                scale=alt.Scale(domain=domain, range=[C["a"], C["b"]])),
                tooltip=["group", "arm", alt.Tooltip("mae:Q", format=".2f"),
                         alt.Tooltip("n:Q", format=",")],
            )
            .properties(width=560, height=280,
                        title="Held-out test error by subgroup")
        )
        layers = [bars]
        if d7_limit.value:
            limit = 2 * d7["overall"]["full_model"]
            rule = (
                alt.Chart(pd.DataFrame({"y": [limit], "what": ["twice the overall MAE"]}))
                .mark_rule(color=C["warn"], strokeDash=[5, 3], strokeWidth=2)
                .encode(y="y:Q", tooltip=["what", alt.Tooltip("y:Q", format=".1f")])
            )
            overall = (
                alt.Chart(pd.DataFrame({"y": [d7["overall"]["full_model"]],
                                        "what": ["overall MAE, full model"]}))
                .mark_rule(color=C["muted"], strokeDash=[3, 3])
                .encode(y="y:Q", tooltip=["what", alt.Tooltip("y:Q", format=".1f")])
            )
            layers += [rule, overall]
        return style(alt.layer(*layers))

    _subgroups()
    return


@app.cell
def _(d7, flag, mo, notes, src):
    mo.vstack(
        [
            flag(d7["breaches"], "warn"),
            notes(
                "The improvement is not uniform. Attention lowers the error in three "
                "brackets, leaves one unchanged and raises it by 11.3 rating points "
                "in the lowest bracket.",
                d7["counting_note"],
                d7["subset_note"],
            ),
            src(
                "analysis/heldout_test_eval/{baseline,attn_tuned}__best.json and "
                "__best__rating_buckets.json",
                "CCS Thesis - Integrated/chapters/chapter4.tex, subgroup performance "
                "by time control and by rating bracket",
            ),
        ]
    )
    return


# ==========================================================================
# 8. White and Black are one number
# ==========================================================================


@app.cell
def _(head):
    head(
        8,
        "White and Black are one number",
        "The network reports two rating estimates per game. Across all 255,000 "
        "test games they are almost the same number.",
    )
    return


@app.cell
def _(load):
    d8 = load("d8_side_gap.json")
    return (d8,)


@app.cell
def _(mo):
    d8_arm = mo.ui.dropdown(
        options={"Full model": "full_model", "Reproduced baseline": "baseline"},
        value="Full model", label="Arm",
    )
    d8_same = mo.ui.switch(
        label="Put both axes on the same scale, with the line independent estimates would follow"
    )
    d8_gap = mo.ui.slider(0, 1000, step=25, value=300, full_width=True,
                          label="Keep only games where the two players really differ by at least")
    mo.vstack([mo.hstack([d8_arm, d8_same], justify="start", gap=2), d8_gap])
    return d8_arm, d8_gap, d8_same


@app.cell
def _(C, alt, d8, d8_arm, d8_gap, d8_same, mo, pd, stat, style):
    def _gap():
        arm = d8["arms"][d8_arm.value]
        scatter = pd.DataFrame(arm["scatter"])
        scatter["kept"] = scatter["true"].abs() >= d8_gap.value
        limit = 1300 if d8_same.value else 6
        points = (
            alt.Chart(scatter).mark_circle(size=18, opacity=0.5)
            .encode(
                x=alt.X("true:Q", title="Real rating gap, White minus Black",
                        scale=alt.Scale(domain=[-limit, limit], nice=False)),
                y=alt.Y("pred:Q", title="Predicted gap",
                        scale=alt.Scale(domain=[-limit, limit], nice=False)),
                color=alt.Color(
                    "kept:N", title=None,
                    scale=alt.Scale(domain=[True, False], range=[C["warn"], C["muted"]]),
                    legend=alt.Legend(labelExpr="datum.label == 'true' ? 'Kept' : 'Filtered out'"),
                ),
                tooltip=[alt.Tooltip("true:Q", format=".0f"),
                         alt.Tooltip("pred:Q", format=".2f")],
            )
            .properties(width=430 if d8_same.value else 560,
                        height=430 if d8_same.value else 260,
                        title="If the two estimates were independent, these dots would "
                              "follow the diagonal")
        )
        zero = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color=C["muted"], strokeDash=[3, 3]).encode(y="y:Q")
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"x": [-limit, limit], "y": [-limit, limit],
                                    "what": ["independent estimates"] * 2}))
            .mark_line(color=C["b"], strokeDash=[6, 4], strokeWidth=2)
            .encode(x="x:Q", y="y:Q", tooltip=["what"])
        )
        layers = points + zero + (diagonal if d8_same.value else zero)

        curve = pd.DataFrame([c for c in arm["mae_above_threshold"] if c["mae"] is not None])
        mae_chart = (
            alt.Chart(curve).mark_line(strokeWidth=2, color=C["a"])
            .encode(
                x=alt.X("threshold:Q", title="Minimum real rating gap"),
                y=alt.Y("mae:Q", title="Test MAE on the games kept",
                        scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("threshold:Q"), alt.Tooltip("mae:Q", format=".1f"),
                         alt.Tooltip("n_games:Q", format=",")],
            )
            .properties(width=560, height=180,
                        title="What the shared estimate costs as the players get further apart")
        )
        here = next(c for c in curve.to_dict("records")
                    if c["threshold"] == d8_gap.value)
        marker = (
            alt.Chart(pd.DataFrame([here])).mark_point(size=110, filled=True,
                                                       color=C["warn"])
            .encode(x="threshold:Q", y="mae:Q")
        )
        return mo.vstack(
            [
                style(layers),
                style(mae_chart + marker),
                mo.hstack(
                    [
                        stat("Games kept", f"{here['n_games']:,}",
                             f"of {arm['n_games']:,}"),
                        stat("MAE on those games", f"{here['mae']:.1f}",
                             f"against {arm['overall_mae']:.1f} overall"),
                        stat("Largest predicted gap in any game",
                             f"{arm['predicted_gap']['max']:.2f}",
                             f"median {arm['predicted_gap']['median']:.2f} rating points"),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _gap()
    return


@app.cell
def _(d8, mo, notes, src):
    mo.vstack(
        [
            notes(
                "The two rows of the network's final layer have a cosine similarity "
                f"of {d8['cosine_similarity_fc2_rows']['full_model']:.4f}, so that "
                "layer applies very nearly the same function twice. The released "
                "reference checkpoint behaves the same way, at "
                f"{d8['cosine_similarity_fc2_rows']['omori_model_55']:.4f}, so this "
                "belongs to the architecture and not to the attention work.",
                "Replacing both estimates with their average leaves the overall test "
                "error unchanged, so the two numbers carry no separable per-side "
                "information.",
            ),
            src(
                "analysis/side-mirroring-results.md",
                "analysis/heldout_test_eval/{attn_tuned,baseline}__best.csv",
                "CCS Thesis - Integrated/chapters/chapter4.tex, agreement between "
                "the White and Black estimates",
                f"Summary values use all {d8['arms']['full_model']['n_games']:,} test "
                f"games; only the {d8['n_scatter']:,} dots are a random subsample.",
            ),
        ]
    )
    return


# ==========================================================================
# 9. The tuning sweep
# ==========================================================================


@app.cell
def _(head):
    head(
        9,
        "The tuning sweep",
        "Seventeen runs, each moving one setting around a fixed control. Six of "
        "them repeat the control exactly and give the sweep its own noise band.",
    )
    return


@app.cell
def _(load):
    d9 = load("d9_sweep.json")
    return (d9,)


@app.cell
def _(d9, mo):
    d9_factor = mo.ui.dropdown(
        options=["All factors"] + sorted({r["factor"] for r in d9["rows"]}),
        value="All factors", label="Factor",
    )
    d9_factor
    return (d9_factor,)


@app.cell
def _(C, alt, d9, d9_factor, pd, style):
    def _sweep():
        rows = d9["rows"]
        if d9_factor.value != "All factors":
            rows = [r for r in rows if r["factor"] == d9_factor.value]
        frame = pd.DataFrame(rows)
        frame["label"] = frame["factor"] + " " + frame["value"]
        frame["role"] = frame["is_control"].map(
            {True: "Control cell", False: "Changed setting"}
        )
        band = d9["control_band"]

        frame = frame.sort_values("best_val")
        order = list(frame["label"])
        x_scale = alt.Scale(domain=[208, 235], nice=False)
        colors = alt.Color("role:N", title=None,
                           scale=alt.Scale(domain=["Changed setting", "Control cell"],
                                           range=[C["a"], C["muted"]]))
        noise = (
            alt.Chart(pd.DataFrame([{"lo": band["lo"], "hi": band["hi"],
                                     "what": "the six control cells"}]))
            .mark_rect(color=C["warn"], opacity=0.12)
            .encode(x=alt.X("lo:Q", scale=x_scale), x2="hi:Q", tooltip=["what"])
        )
        dots = (
            alt.Chart(frame).mark_point(filled=True, size=130)
            .encode(
                y=alt.Y("label:N", title=None, sort=order),
                x=alt.X("best_val:Q", title="Best-validation loss in rating points",
                        scale=x_scale),
                color=colors,
                shape=alt.Shape("role:N", title=None,
                                scale=alt.Scale(domain=["Changed setting", "Control cell"],
                                                range=["circle", "square"])),
                tooltip=["label", alt.Tooltip("best_val:Q", format=".2f"),
                         alt.Tooltip("terminal:Q", format=".2f")],
            )
            .properties(width=520, height=max(120, 26 * len(frame)),
                        title="Each run against the sweep's own noise band")
        )
        text = (
            alt.Chart(frame).mark_text(align="left", dx=10, fontSize=10, color=C["ink"])
            .encode(y=alt.Y("label:N", sort=order),
                    x=alt.X("best_val:Q", scale=x_scale),
                    text=alt.Text("best_val:Q", format=".2f"))
        )
        return style(noise + dots + text)

    _sweep()
    return


@app.cell
def _(d9, flag, mo, notes, src):
    mo.vstack(
        [
            flag(
                "The shaded band is the six control cells, "
                f"{d9['control_band']['lo']} to {d9['control_band']['hi']} around a "
                f"mean of {d9['control_band']['mean']}. A run inside it moved nothing."
            ),
            notes(
                "Only the learning rate cleared that band by a clear margin, at "
                f"{d9['adopted']['best_val']}. " + d9["adopted"]["note"],
                d9["metric"],
            ),
            src("CCS Thesis - Integrated/chapters/appendix.tex:460, "
                "one-factor-at-a-time sweep results"),
        ]
    )
    return


# ==========================================================================
# 10. Real-time speed
# ==========================================================================


@app.cell
def _(head):
    head(
        10,
        "Real-time speed",
        "How long one move takes to score end to end, measured on 100 matched "
        "test games at three game lengths.",
    )
    return


@app.cell
def _(load):
    d10 = load("d10_latency.json")
    return (d10,)


@app.cell
def _(d10, mo):
    d10_budget = mo.ui.dropdown(
        options={b["name"]: i for i, b in enumerate(d10["budget_reference"])},
        value=d10["budget_reference"][0]["name"],
        label="Compare against the time per move in",
    )
    d10_budget
    return (d10_budget,)


@app.cell
def _(C, alt, d10, d10_budget, mo, pd, stat, style):
    def _latency():
        rows = []
        for dev in d10["devices"]:
            for point in dev["at_ply"]:
                rows.append({"device": dev["label"], "ply": str(point["ply"]),
                             "mean_ms": point["mean_ms"], "p95_ms": point["p95_ms"],
                             "n": point["n"]})
        frame = pd.DataFrame(rows)
        domain = [d["label"] for d in d10["devices"]]

        bars = (
            alt.Chart(frame).mark_bar(cornerRadiusEnd=3, stroke="#fcfcfb", strokeWidth=2)
            .encode(
                x=alt.X("ply:N", title="Game length in plies", sort=["10", "50", "100"],
                        axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("device:N", sort=domain),
                y=alt.Y("mean_ms:Q", title="Milliseconds for one move"),
                color=alt.Color("device:N", title=None,
                                scale=alt.Scale(domain=domain, range=[C["a"], C["b"]])),
                tooltip=["device", "ply", alt.Tooltip("mean_ms:Q", format=".2f"),
                         alt.Tooltip("p95_ms:Q", format=".2f"),
                         alt.Tooltip("n:Q", format=",")],
            )
            .properties(width=480, height=250, title="Per-move inference latency")
        )
        whisk = (
            alt.Chart(frame).mark_rule(strokeWidth=1.5, color=C["ink"])
            .encode(x=alt.X("ply:N", sort=["10", "50", "100"]),
                    xOffset=alt.XOffset("device:N", sort=domain),
                    y="mean_ms:Q", y2="p95_ms:Q")
        )
        budget = d10["budget_reference"][d10_budget.value]
        gpu = d10["devices"][0]["overall"]
        cpu = d10["devices"][1]["overall"]
        return mo.vstack(
            [
                style(bars + whisk),
                mo.hstack(
                    [
                        stat("GPU, typical move", f"{gpu['p50_ms']:.1f} ms",
                             f"99th percentile {gpu['p99_ms']:.1f} ms"),
                        stat("CPU only, typical move", f"{cpu['p50_ms']:.1f} ms",
                             f"99th percentile {cpu['p99_ms']:.1f} ms"),
                        stat(f"Budget in {budget['name']}",
                             f"{budget['seconds_per_move'] * 1000:,.0f} ms",
                             budget["note"]),
                    ],
                    justify="start", gap=1,
                ),
            ]
        )

    _latency()
    return


@app.cell
def _(d10, mo, notes, src):
    mo.vstack(
        [
            notes(d10["measured_plies_note"], d10["cpu_caveat"], d10["agreement_note"]),
            src(
                "analysis/deployment_measurements/latency_agreement_gpu_a5000.json",
                "analysis/deployment_measurements/latency_agreement_cpu_xeon4309y.json",
                "analysis/deployment-measurements.md",
                "Hardware: " + "; ".join(d["hardware"] for d in d10["devices"]),
            ),
        ]
    )
    return


@app.cell
def _(load, mo):
    mo.md(
        "<div style='font-size:0.76rem;opacity:0.6;border-top:1px solid #e3e2dd;"
        "padding-top:8px'>Data rebuilt "
        + load("meta.json")["built"]
        + " by presentation/build_data.py. "
        + load("meta.json")["rule"]
        + "</div>"
    )
    return


if __name__ == "__main__":
    app.run()

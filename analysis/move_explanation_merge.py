"""Merge curves_3000.jsonl + engine_3000.jsonl into one per-ply pandas
DataFrame and save it locally (not committed -- same convention the project
uses for its other per-game CSVs, see mirror-retrain-results.md). Run once;
the regression and case-study scripts both load the saved parquet.
"""
import json
import math
import pandas as pd

# --- parse_time_control / compute_time_spent, copied in pure-python form
# from prototype/src/format_data.py (which imports torch at module level,
# so it is not imported directly here to keep this analysis script
# dependency-light). Logic must stay identical to that module.
def parse_time_control(tc):
    if not tc:
        return None, None
    parts = tc.split("+")
    if len(parts) != 2 or not (parts[0].isdigit() and parts[1].isdigit()):
        return None, None
    return int(parts[0]), int(parts[1])


def compute_time_spent(clock_seconds, base_seconds, increment_seconds):
    inc = increment_seconds or 0
    spent = []
    for i, clock in enumerate(clock_seconds):
        previous = clock_seconds[i - 2] if i >= 2 else base_seconds
        spent.append(None if previous is None else previous - clock + inc)
    return spent


def categorize_time_control(estimated_duration):
    if estimated_duration < 29:
        return "ultrabullet"
    elif estimated_duration < 179:
        return "bullet"
    elif estimated_duration < 479:
        return "blitz"
    elif estimated_duration < 1499:
        return "rapid"
    else:
        return "classical"


def tc_bucket(tc):
    base, inc = parse_time_control(tc)
    if base is None:
        return None
    return categorize_time_control(base + 40 * inc)


def ply_band(ply):
    if ply <= 10:
        return "1-10"
    elif ply <= 20:
        return "11-20"
    elif ply <= 40:
        return "21-40"
    elif ply <= 60:
        return "41-60"
    elif ply <= 80:
        return "61-80"
    else:
        return "81-100"


def main():
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

    rows = []
    n_games_used = 0
    for gid, c in curves.items():
        e = engine.get(gid)
        if e is None or "error" in e:
            continue
        n = min(c["n_plies"], e["n_plies"])
        if n < 3:
            continue
        n_games_used += 1

        tc = c.get("time_control")
        base, inc = parse_time_control(tc)
        bucket = tc_bucket(tc)
        time_spent = compute_time_spent(c["clocks_seconds"][:n], base, inc)

        sw, sb = c["served"]["live_white"][:n], c["served"]["live_black"][:n]
        mw, mb = c["mirror_arm_a_gapweight"]["live_white"][:n], c["mirror_arm_a_gapweight"]["live_black"][:n]
        attn = (c["served"].get("offline_attention") or [None] * n)[:n]

        movers = e["mover"][:n]
        cp_loss = e["cp_loss"][:n]
        eb = e["eval_before_white_pov"][:n]
        ea = e["eval_after_white_pov"][:n]
        phase = e["phase"][:n]
        san = e["san"][:n]

        for i in range(n):
            ply = i + 1
            mover = movers[i]
            if mover is None:
                continue
            # First move of each side falls back to the TimeControl base
            # allotment (no earlier in-game clock to diff against); berserk
            # halves that base, so this ply's "time spent" is not comparable
            # to a normal move and is dropped rather than kept as a guess.
            ts = time_spent[i] if ply > 2 else None
            if ts is not None and ts < 0:
                ts = 0.0

            if i == 0:
                d_white_s = d_black_s = d_white_m = d_black_m = None
            else:
                d_white_s = sw[i] - sw[i - 1]
                d_black_s = sb[i] - sb[i - 1]
                d_white_m = mw[i] - mw[i - 1]
                d_black_m = mb[i] - mb[i - 1]

            if mover == "white":
                d_own_s, d_opp_s = d_white_s, d_black_s
                d_own_m, d_opp_m = d_white_m, d_black_m
            else:
                d_own_s, d_opp_s = d_black_s, d_white_s
                d_own_m, d_opp_m = d_black_m, d_white_m

            d_mean_s = None if d_white_s is None else (d_white_s + d_black_s) / 2.0
            d_mean_m = None if d_white_m is None else (d_white_m + d_black_m) / 2.0
            d_diff_s = None if d_own_s is None else d_own_s - d_opp_s
            d_diff_m = None if d_own_m is None else d_own_m - d_opp_m

            c_loss = cp_loss[i]
            e_before, e_after = eb[i], ea[i]
            signed_eval_change = None if (e_before is None or e_after is None) else (e_after - e_before)

            rows.append(
                dict(
                    game_id=gid,
                    ply=ply,
                    ply_band=ply_band(ply),
                    mover=mover,
                    phase=phase[i],
                    san=san[i],
                    tc_bucket=bucket,
                    time_spent=ts,
                    cp_loss=c_loss,
                    eval_before=e_before,
                    eval_after=e_after,
                    signed_eval_change=signed_eval_change,
                    attention=attn[i],
                    white_elo=c.get("white_elo"),
                    black_elo=c.get("black_elo"),
                    served_white=sw[i],
                    served_black=sb[i],
                    mirror_white=mw[i],
                    mirror_black=mb[i],
                    delta_white_served=d_white_s,
                    delta_black_served=d_black_s,
                    delta_own_served=d_own_s,
                    delta_opp_served=d_opp_s,
                    delta_mean_served=d_mean_s,
                    delta_own_minus_opp_served=d_diff_s,
                    delta_white_mirror=d_white_m,
                    delta_black_mirror=d_black_m,
                    delta_own_mirror=d_own_m,
                    delta_opp_mirror=d_opp_m,
                    delta_mean_mirror=d_mean_m,
                    delta_own_minus_opp_mirror=d_diff_m,
                )
            )

    df = pd.DataFrame(rows)
    print("n_games_used:", n_games_used)
    print("n_rows:", len(df))
    df.to_csv("merged_per_ply.csv", index=False)
    print("saved merged_per_ply.csv")

    # Quick descriptive summary for the write-up.
    final_rows = []
    for gid, c in curves.items():
        e = engine.get(gid)
        if e is None or "error" in e:
            continue
        n = min(c["n_plies"], e["n_plies"])
        if n < 1:
            continue
        we, be = c.get("white_elo"), c.get("black_elo")
        if we is None or be is None:
            continue
        sw_f, sb_f = c["served"]["live_white"][n - 1], c["served"]["live_black"][n - 1]
        mw, mb = c["mirror_arm_a_gapweight"]["live_white"], c["mirror_arm_a_gapweight"]["live_black"]
        mirror_final_wb = mw[n - 1] - mb[n - 1]
        # Parity-free: mean of W-B over the last two plies, which (since the
        # last-mover artifact below flips sign by mover) cancels one
        # White-just-moved upswing against one Black-just-moved downswing
        # when n >= 2.
        mirror_parity_free = (
            (mw[n - 1] - mb[n - 1] + mw[n - 2] - mb[n - 2]) / 2.0 if n >= 2 else mirror_final_wb
        )
        final_rows.append(
            dict(
                game_id=gid,
                n_plies=n,
                last_mover="white" if n % 2 == 1 else "black",
                tc_bucket=tc_bucket(c.get("time_control")),
                white_elo=float(we),
                black_elo=float(be),
                served_white_final=sw_f,
                served_black_final=sb_f,
                served_gap=abs(sw_f - sb_f),
                mirror_white_final=mw[n - 1],
                mirror_black_final=mb[n - 1],
                mirror_final_wb=mirror_final_wb,
                mirror_gap=abs(mirror_final_wb),
                mirror_parity_free_wb=mirror_parity_free,
                mirror_parity_free_gap=abs(mirror_parity_free),
            )
        )
    fdf = pd.DataFrame(final_rows)
    fdf.to_csv("final_per_game.csv", index=False)
    print("median served_gap:", fdf["served_gap"].median())
    print("median mirror_gap:", fdf["mirror_gap"].median())
    print("n games with elo:", len(fdf))

    # Last-mover parity diagnostic for mirror_arm_a's final-ply gap: does the
    # final W-B's sign track who moved last (the same artifact the per-ply
    # regression finds), and how much of the raw gap survives a parity
    # correction? Written to a small committed JSON since the full
    # final_per_game.csv is not committed.
    def median_abs(series):
        return float(series.abs().median()) if len(series) else None

    parity = {
        "n_games": int(len(fdf)),
        "median_abs_final_wb_overall": median_abs(fdf["mirror_final_wb"]),
        "median_abs_parity_free_wb_overall": median_abs(fdf["mirror_parity_free_wb"]),
        "by_bucket": {},
    }
    for label, sub in [
        ("non_ultrabullet", fdf[fdf["tc_bucket"] != "ultrabullet"]),
        ("ultrabullet", fdf[fdf["tc_bucket"] == "ultrabullet"]),
    ]:
        entry = {
            "n_games": int(len(sub)),
            "median_abs_final_wb": median_abs(sub["mirror_final_wb"]),
            "median_abs_parity_free_wb": median_abs(sub["mirror_parity_free_wb"]),
            "by_last_mover": {},
        }
        for mover in ["white", "black"]:
            s2 = sub[sub["last_mover"] == mover]
            if len(s2):
                entry["by_last_mover"][mover] = {
                    "n_games": int(len(s2)),
                    "median_final_wb_signed": float(s2["mirror_final_wb"].median()),
                    "mean_final_wb_signed": float(s2["mirror_final_wb"].mean()),
                }
        parity["by_bucket"][label] = entry
    with open("final_gap_parity_diagnostics.json", "w") as f:
        json.dump(parity, f, indent=2)
    print("Wrote final_gap_parity_diagnostics.json")


if __name__ == "__main__":
    main()

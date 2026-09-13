"""E8: data-side and evaluation-side facts, measured on the real 6k subset.

Q1. Player leakage across the 72/18/10 split (audit line 45 flags "no
    player-leakage check"). Can't test directly -- the pickles drop usernames
    (format_data.parse_game:75-83 keeps no player ID). Documented as a
    structural blind spot instead.
Q2. How much label noise is there? Two players, one game, one shared board
    sequence -- the model must emit two different numbers from one input.
Q3. What does the 100-ply cap discard?
Q4. Naive baselines: predict-mean, and predict-mean-per-time-control.
Q5. Target distribution vs the hardcoded normalisation constants.
"""
import glob, pickle, sys
import numpy as np
sys.path.insert(0, "/tmp/opencode/lab/src")
from format_data import categorize_time_control, time_to_seconds
from sklearn.model_selection import train_test_split

files = sorted(glob.glob("/tmp/opencode/lab/data/2024-07/*.pkl"))
print(f"corpus: {len(files)} games (Lichess 2024-07, paper's window)\n")

W, Bl, L, TC, gap = [], [], [], [], []
for f in files:
    d = pickle.load(open(f, "rb"))
    w, b = int(d["WhiteElo"]), int(d["BlackElo"])
    W.append(w); Bl.append(b); L.append(len(d["Positions"]))
    i, inc = map(int, d["Time"].split("+"))
    TC.append(categorize_time_control(i + 40 * inc))
    gap.append(abs(w - b))
W, Bl, L, gap = map(np.array, (W, Bl, L, gap))
TC = np.array(TC)

print("=== Q5: target distribution vs hardcoded constants ===")
allr = np.concatenate([W, Bl])
print(f"  hardcoded (chess_rating_net.py:53-54): mean=1514 std=366")
print(f"  measured on this subset:               mean={allr.mean():.0f} std={allr.std():.0f}")
print(f"  -> standardised targets would have mean "
      f"{(allr.mean()-1514)/366:+.3f}, std {allr.std()/366:.3f} instead of 0.0/1.0")
print("  (one month only; the 1.2M corpus spans 2021-2024 so the paper's")
print("   constants may well be right for the full run. Worth a recompute check,")
print("   NOT a claimed improvement.)")

print("\n=== Q2: irreducible label noise (the two-players-one-game problem) ===")
print(f"  |WhiteElo - BlackElo|: mean={gap.mean():.0f} median={np.median(gap):.0f} "
      f"p90={np.percentile(gap,90):.0f}")
print(f"  frac of games with a rating gap > 200: {(gap>200).mean():.3f}")
print("  The board sequence is IDENTICAL for both targets; only the clock")
print("  channel and move parity distinguish them. Half the rating gap is a")
print("  floor on what any symmetric model can do on the pair-averaged MAE:")
print(f"  -> a model that predicts the same number for both players eats")
print(f"     >= {gap.mean()/2:.0f} MAE on average from this alone.")

print("\n=== Q3: what the 100-ply cap discards ===")
print(f"  games at the cap: {(L==100).mean():.3f}  (mean len {L.mean():.1f})")
print("  For capped games the model never sees the endgame, where blunder")
print("  rate is most rating-discriminative. Raising the cap is a DATA change")
print("  (chapter3.tex:176-181 fixes it at 100) -- out of scope, note only.")

print("\n=== Q4: naive baselines on this subset (last-ply-equivalent MAE) ===")
tv, te = train_test_split(np.arange(len(files)), test_size=0.1, random_state=42)
tr, va = train_test_split(tv, test_size=0.2, random_state=42)
tr_r = np.concatenate([W[tr], Bl[tr]])
gmean, gmed = tr_r.mean(), np.median(tr_r)


def mae(pred_w, pred_b, idx):
    return (np.abs(pred_w - W[idx]) + np.abs(pred_b - Bl[idx])).mean() / 2


print(f"  predict train mean ({gmean:.0f}):   test MAE = "
      f"{mae(gmean, gmean, te):.1f}")
print(f"  predict train median ({gmed:.0f}): test MAE = "
      f"{mae(gmed, gmed, te):.1f}")
pw = np.zeros(len(te)); pb = np.zeros(len(te))
for i, j in enumerate(te):
    sel = TC[tr] == TC[j]
    pw[i] = pb[i] = np.median(np.concatenate([W[tr][sel], Bl[tr][sel]])) if sel.any() else gmed
print(f"  predict per-time-control median: test MAE = "
      f"{(np.abs(pw-W[te])+np.abs(pb-Bl[te])).mean()/2:.1f}")
print("  (paper reports 346 for predict-mean on the full 1.2M corpus,")
print("   Table 1 'Mean' column / chapter3.tex:236)")

print("\n=== Q1: player-leakage blind spot ===")
d0 = pickle.load(open(files[0], "rb"))
print(f"  keys stored per game: {sorted(d0.keys())}")
print("  -> no username/player-ID field, so a player-disjoint split CANNOT be")
print("     built from the current pickles. Same player appearing in train and")
print("     test is undetectable. This is a preprocessing change, and it would")
print("     break exact comparability with the baseline's random split.")

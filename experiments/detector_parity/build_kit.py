"""Build a tiny fake v2-style corpus for the detector parity check.

Samples: each bundled PGN becomes two pickles (white as suspect, black as
suspect) parsed with the thesis src/format_data.parse_game, maia_band = that
side's PGN header Elo. Corpus: a deterministic handful of real anomaly_corpus_v2
games from the grouped split's test/heldout_band partitions, copied unchanged,
and exported as PGN for the app side.
"""
import glob, json, os, pickle, shutil, sys
import numpy as np
import chess, chess.pgn

T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "src"))
from format_data import parse_game  # thesis copy

OUT = os.path.join(T, "scratch/detector-default/parity")
CORP = os.path.join(OUT, "corpus")
shutil.rmtree(CORP, ignore_errors=True)
os.makedirs(CORP)
os.makedirs(os.path.join(OUT, "corpus_pgn"), exist_ok=True)
index = []
k = 0

def put(cell, raw, meta):
    global k
    d = os.path.join(CORP, cell); os.makedirs(d, exist_ok=True)
    fn = f"game_{k:05d}.pkl"; k += 1
    pickle.dump(raw, open(os.path.join(d, fn), "wb"))
    meta["gid"] = f"{cell}/{fn}"
    index.append(meta)

for p in sorted(glob.glob(os.path.join(OUT, "samples/*.pgn"))):
    g = chess.pgn.read_game(open(p))
    info = parse_game(g, max_plies=100)
    for color in ("white", "black"):
        elo = int(round(float(g.headers["WhiteElo" if color == "white" else "BlackElo"])))
        raw = {"Positions": info["Positions"], "Moves": info["Moves"], "Clocks": info["Clocks"],
               "suspect_color": color, "maia_band": elo,
               "WhiteElo": int(float(g.headers["WhiteElo"])), "BlackElo": int(float(g.headers["BlackElo"]))}
        put(f"{elo}_none_r00", raw, {"kind": "sample", "source": os.path.basename(p), "side": color})

z = np.load(os.path.join(T, "analysis/extra_arms/A3g_seed0/scores.npz"), allow_pickle=True)
gid = z["gid"].astype(str); split = z["split"].astype(str); score = z["score"]
rng = np.random.default_rng(20260917)
picked = []
for sp, cellpat, nsel in [("test", "_r00", 2), ("test", "_r60", 1), ("heldout_band", "_r02", 1),
                          ("heldout_band", "_r05", 1), ("heldout_band", "_r40", 1), ("test", "_hardneg", 1),
                          ("heldout_band", "_r20", 1)]:
    cand = [i for i in range(len(gid)) if split[i] == sp and gid[i].split("/")[0].endswith(cellpat)]
    picked += list(rng.choice(cand, nsel, replace=False))
for i in picked:
    cell, fn = gid[i].split("/")
    raw = pickle.load(open(os.path.join(T, "data/anomaly_corpus_v2", cell, fn), "rb"))
    band = int(cell.split("_")[0])
    board = chess.Board(); game = chess.pgn.Game()
    game.headers["Event"] = f"anomaly_corpus_v2 {gid[i]}"
    game.headers["White"] = "Maia"; game.headers["Black"] = "Maia"
    game.headers["WhiteElo"] = str(raw.get("maia_band", band)); game.headers["BlackElo"] = str(raw.get("maia_band", band))
    game.headers["TimeControl"] = raw.get("Time", "-"); game.headers["Result"] = raw.get("Result", "*")
    node = game
    for uci, clk in zip(raw["Moves"][:100], raw["Clocks"][:100]):
        node = node.add_variation(chess.Move.from_uci(uci)); node.comment = f"[%clk {clk}]"
    name = gid[i].replace("/", "__").replace(".pkl", ".pgn")
    open(os.path.join(OUT, "corpus_pgn", name), "w").write(str(game) + "\n")
    put(cell, raw, {"kind": "corpus", "source": name, "side": raw["suspect_color"], "orig_gid": gid[i],
                    "split": sp, "thesis_score": float(score[i])})
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(len(index), "entries")

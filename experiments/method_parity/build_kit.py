"""Build a tiny v2-style corpus for the LightGBM (A0g) and CNN-BiLSTM (A4)
parity checks (HPC). Same construction as experiments/detector_parity/build_kit.py.

Samples: every bundled PGN (current and archived) becomes two pickles, white and
black in turn as the suspect, parsed with the thesis src/format_data.parse_game,
maia_band = that side's header Elo. Corpus: the 8 anomaly_corpus_v2 games of the
detector parity check plus 4 more (2 from A4's v2 test partition, 2 from A0g's
grouped heldout_both partition), copied unchanged and exported as PGN.
"""
import glob, json, os, pickle, shutil, sys
import numpy as np
import chess, chess.pgn

T = os.path.expanduser("~/Bacsain/thesis2")
sys.path.insert(0, os.path.join(T, "src"))
from format_data import parse_game  # thesis copy

OUT = os.path.join(T, "scratch/arm-selector/parity")
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

prev = json.load(open(os.path.join(T, "scratch/detector-default/parity/index.json")))
picked = [e["orig_gid"] for e in prev if e["kind"] == "corpus"]
rng = np.random.default_rng(20260917)
for arm, sp, n in (("A4", "test", 2), ("A0g", "heldout_both", 2)):
    z = np.load(os.path.join(T, f"analysis/extra_arms/{arm}/scores.npz"), allow_pickle=True)
    gid = z["gid"].astype(str); split = z["split"].astype(str)
    cand = sorted(set(gid[split == sp].tolist()) - set(picked))
    picked += [str(x) for x in rng.choice(cand, n, replace=False)]

for g_id in picked:
    cell, fn = g_id.split("/")
    raw = pickle.load(open(os.path.join(T, "data/anomaly_corpus_v2", cell, fn), "rb"))
    band = int(cell.split("_")[0])
    game = chess.pgn.Game()
    game.headers["Event"] = f"anomaly_corpus_v2 {g_id}"
    game.headers["White"] = "Maia"; game.headers["Black"] = "Maia"
    game.headers["WhiteElo"] = str(raw.get("maia_band", band)); game.headers["BlackElo"] = str(raw.get("maia_band", band))
    game.headers["TimeControl"] = raw.get("Time", "-"); game.headers["Result"] = raw.get("Result", "*")
    node = game
    for uci, clk in zip(raw["Moves"][:100], raw["Clocks"][:100]):
        node = node.add_variation(chess.Move.from_uci(uci)); node.comment = f"[%clk {clk}]"
    name = g_id.replace("/", "__").replace(".pkl", ".pgn")
    open(os.path.join(OUT, "corpus_pgn", name), "w").write(str(game) + "\n")
    put(cell, raw, {"kind": "corpus", "source": name, "side": raw["suspect_color"], "orig_gid": g_id})
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(len(index), "entries")

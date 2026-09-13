"""Stream a small real Lichess subset (paper's window) into the repo's pickle format."""
import argparse, io, os, pickle, subprocess, sys
sys.path.insert(0, "/tmp/opencode/lab/src")
import chess.pgn, zstandard
from format_data import parse_game

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-games", type=int, default=4000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

proc = subprocess.Popen(["curl", "-sL", a.url], stdout=subprocess.PIPE)
reader = zstandard.ZstdDecompressor().stream_reader(proc.stdout)
text = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

kept = scanned = 0
while kept < a.max_games:
    try:
        g = chess.pgn.read_game(text)
    except Exception:
        continue
    if g is None:
        break
    scanned += 1
    h = g.headers
    if h.get("Variant", "Standard") != "Standard":
        continue
    we, be, tc = h.get("WhiteElo", "?"), h.get("BlackElo", "?"), h.get("TimeControl", "")
    if not (we.isdigit() and be.isdigit()):
        continue
    if "+" not in tc:
        continue
    try:
        int(tc.split("+")[0]); int(tc.split("+")[1])
    except Exception:
        continue
    d = parse_game(g)
    if d is None:
        continue
    # The model requires one clock reading per position.
    if len(d["Clocks"]) != len(d["Positions"]) or len(d["Positions"]) < 6:
        continue
    with open(os.path.join(a.out, f"game_{kept:06d}.pkl"), "wb") as f:
        pickle.dump(d, f)
    kept += 1
    if kept % 500 == 0:
        print(f"kept={kept} scanned={scanned}", flush=True)

print(f"DONE scanned={scanned} kept={kept}", flush=True)
proc.kill()

"""E5: last-ply-only supervision vs dense (all-ply) supervision.

The baseline trains on ONE timestep per game (chess_rating_net.py:305-306
discards per_move_preds). The model emits a prediction after every ply and
the label is constant across the game, so every ply is a valid training
target. Dense supervision multiplies the gradient signal per game by ~65x
at zero extra forward cost.

Both arms are IDENTICAL except for the loss reduction. Same seed, same data,
same split, same optimizer, same architecture. Eval is always last-ply MAE,
i.e. the baseline's own metric -- dense supervision gets no metric advantage.
"""
import argparse, glob, pickle, sys, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor
from format_data import time_to_seconds, categorize_time_control

R_MEAN, R_STD = 1514.0, 366.0
C_MEAN, C_STD = 273.0, 380.0

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["last", "dense"], required=True)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--games", type=int, default=6000)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-ply", type=int, default=100)
a = ap.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)


class DS(Dataset):
    def __init__(self, files):
        self.items = []
        for f in files:
            d = pickle.load(open(f, "rb"))
            p = torch.stack(d["Positions"])[: a.max_ply].to(torch.uint8)
            c = torch.tensor([time_to_seconds(x) for x in d["Clocks"]],
                             dtype=torch.float)[: a.max_ply]
            c = (c - C_MEAN) / C_STD
            t = torch.tensor([float(d["WhiteElo"]), float(d["BlackElo"])])
            t = (t - R_MEAN) / R_STD
            i, inc = map(int, d["Time"].split("+"))
            self.items.append((p, c, t, len(p), categorize_time_control(i + 40 * inc)))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def coll(b):
    return (pad_sequence([x[0] for x in b], batch_first=True),
            pad_sequence([x[1] for x in b], batch_first=True),
            torch.stack([x[2] for x in b]),
            torch.tensor([x[3] for x in b], dtype=torch.int),
            [x[4] for x in b])


files = sorted(glob.glob("/tmp/opencode/lab/data/2024-07/*.pkl"))[: a.games]
tv, te = train_test_split(files, test_size=0.1, random_state=42)
tr, va = train_test_split(tv, test_size=0.2, random_state=42)
print(f"[{a.mode}] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

t0 = time.time()
dtr, dva, dte = DS(tr), DS(va), DS(te)
print(f"[{a.mode}] loaded in {time.time()-t0:.0f}s", flush=True)

ltr = DataLoader(dtr, batch_size=32, shuffle=True, collate_fn=coll)
lva = DataLoader(dva, batch_size=64, shuffle=False, collate_fn=coll)
lte = DataLoader(dte, batch_size=64, shuffle=False, collate_fn=coll)

model = ChessEloPredictor()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=5, factor=0.5)


def evaluate(loader):
    """Always last-ply MAE in rating points -- the baseline's own metric."""
    model.eval()
    tot = n = 0.0
    by = {}
    with torch.no_grad():
        for p, c, t, l, tcs in loader:
            _, out = model(p.float(), c, l)
            e = (out * R_STD - t * R_STD).abs().mean(dim=1)
            tot += e.sum().item(); n += len(e)
            for i, tc in enumerate(tcs):
                by.setdefault(tc, []).append(e[i].item())
    return tot / n, {k: float(np.mean(v)) for k, v in by.items()}


best = float("inf"); best_state = None
for ep in range(a.epochs):
    model.train(); s = time.time(); tl = 0.0; nb = 0
    for p, c, t, l, _ in ltr:
        opt.zero_grad()
        per_move, last = model(p.float(), c, l)
        if a.mode == "last":
            loss = (last * R_STD - t * R_STD).abs().mean()
        else:
            # Mask to real plies; label is constant across the game.
            T = per_move.shape[1]
            m = (torch.arange(T).unsqueeze(0) < l.unsqueeze(1)).unsqueeze(-1)
            err = (per_move * R_STD - t.unsqueeze(1) * R_STD).abs() * m
            loss = err.sum() / (m.sum() * 2)
        loss.backward(); opt.step()
        tl += loss.item(); nb += 1
    vm, _ = evaluate(lva)
    sched.step(vm)
    if vm < best:
        best = vm
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"[{a.mode}] epoch {ep+1:2d} train_loss={tl/nb:7.2f} val_MAE={vm:7.2f} "
          f"({time.time()-s:.0f}s)", flush=True)

model.load_state_dict(best_state)
tm, by = evaluate(lte)
print(f"\n[{a.mode}] RESULT best_val_MAE={best:.2f} test_MAE={tm:.2f}")
print(f"[{a.mode}] by time control: "
      + " ".join(f"{k}={v:.0f}" for k, v in sorted(by.items())), flush=True)

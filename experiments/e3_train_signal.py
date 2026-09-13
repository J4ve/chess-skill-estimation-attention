"""E3: Where does the training signal actually come from, and what does
padding do to it?

Hypotheses under test:
  H3a  The loss supervises ONLY the final ply, discarding T-1 per-move outputs.
  H3b  BatchNorm2d sees the zero-padded boards, so its batch statistics are
       contaminated by however much of the batch is padding.
  H3c  SelfAttention's step_importance averages over padded QUERY rows.
  H3d  Positions and clocks can have different per-game lengths.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor, collate_fn
from attention import SelfAttention

torch.manual_seed(0)

print("=== H3a: how many timesteps of the output are supervised? ===")
m = ChessEloPredictor()
B, T = 4, 30
pos = torch.randn(B, T, 12, 8, 8)
clk = torch.randn(B, T)
lens = torch.tensor([30, 22, 17, 9], dtype=torch.int)
per_move, last = m(pos, clk, lens)
print(f"  per_move_preds shape {tuple(per_move.shape)}  -> {per_move.shape[1]} predictions/game")
print(f"  last_step_preds shape {tuple(last.shape)}  -> 1 prediction/game")
print("  train_one_epoch (chess_rating_net.py:305-306) does:")
print("      _, outputs = model(...)      # discards per_move_preds")
print("      loss = criterion(outputs...) # loss on the LAST ply only")
sup = 1
tot = int(lens.float().mean().item())
print(f"  -> supervised timesteps per game: {sup} of ~{tot} real plies "
      f"({100*sup/tot:.1f}% of the emitted curve)")

print("\n=== H3b: fraction of the CNN's batch that is zero padding ===")
# Realistic ply-length distribution: Lichess games capped at 100 plies.
# Use a plausible spread; the point is the mechanism, not the exact number.
for desc, lengths in [
    ("all games hit the 100-ply cap", torch.full((32,), 100)),
    ("mixed short/long (30..100)", torch.randint(30, 101, (32,))),
    ("bullet-heavy, many short games (10..100)", torch.randint(10, 101, (32,))),
]:
    Tmax = int(lengths.max())
    real = int(lengths.sum())
    padded = 32 * Tmax
    print(f"  {desc:42s} T_max={Tmax:3d}  padding={100*(1-real/padded):5.1f}% of CNN batch")

print("\n  Proof the padded boards actually enter BatchNorm:")
m.train()
pos2 = torch.randn(2, 10, 12, 8, 8)
pos2[1, 5:] = 0.0                       # game 1 is only 5 plies long -> zeros
lens2 = torch.tensor([10, 5], dtype=torch.int)
before = m.bn1.running_mean.clone()
_ = m(pos2, torch.randn(2, 10), lens2)
after = m.bn1.running_mean.clone()
# Now the same batch with the padding replaced by real data.
m.bn1.running_mean.copy_(before)
pos3 = pos2.clone(); pos3[1, 5:] = torch.randn(5, 12, 8, 8)
_ = m(pos3, torch.randn(2, 10), lens2)
after_nopad = m.bn1.running_mean.clone()
print(f"    ||running_mean(with padding) - running_mean(padding replaced)|| = "
      f"{(after - after_nopad).norm().item():.6e}")
print(f"    -> padded boards change BN statistics? "
      f"{(after - after_nopad).norm().item() > 1e-9}")
print("    chess_rating_net.py:237 flattens (B,T,12,8,8)->(B*T,12,8,8) BEFORE")
print("    the conv stack; pack_padded_sequence at :251 only protects the LSTM.")

print("\n=== H3c: SelfAttention step_importance averages over padded queries ===")
sa = SelfAttention(hidden_dim=128)
h = torch.randn(2, 10, 128)
mask = torch.zeros(2, 10, dtype=torch.bool); mask[0, :10] = True; mask[1, :4] = True
_, imp = sa(h, mask=mask)
print(f"  step_importance for game 1 (only 4 real plies of 10):")
print(f"    {imp[1].detach().numpy().round(4)}")
print(f"    sum over the 6 PADDED steps = {imp[1][4:].sum().item():.4f}")
print(f"    (attention.py:99 does weights.mean(dim=(1,2)) -- dim 2 is the QUERY")
print(f"     axis, which still contains the padded rows)")

print("\n=== H3d: can positions and clocks have different lengths? ===")
print("  format_data.parse_game:61-64 appends a clock ONLY when the move comment")
print("  carries a [%clk ...] tag, but appends a position for EVERY move")
print("  (:58). A game with partial clock annotation yields len(clocks) < len(positions).")
item = {
    "positions": torch.randn(20, 12, 8, 8),
    "clocks": torch.randn(14),          # 6 plies missing a [%clk] tag
    "targets": torch.randn(2), "length": 20, "time_control": "blitz",
    "white": False, "last_rating": None, "result": "1-0",
}
batch = collate_fn([item, item])
print(f"  collate_fn -> positions {tuple(batch['positions'].shape)}, "
      f"clocks {tuple(batch['clocks'].shape)}, lengths={batch['lengths'].tolist()}")
try:
    m2 = ChessEloPredictor(); m2.eval()
    with torch.no_grad():
        m2(batch["positions"], batch["clocks"], batch["lengths"])
    print("  forward pass: OK (no error raised)")
except Exception as e:
    print(f"  forward pass RAISES: {type(e).__name__}: {e}")
    print("  -> such games would crash training, i.e. they must be absent")
    print("     from the corpus or be silently dropped upstream.")

"""E1: Does the attention module receive gradient on the training path?

train_one_epoch() calls model(positions, clocks, lengths) with the default
return_attention=False. Check whether attention params get .grad populated,
and whether the rating head consumes the attention context vector at all.
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor, train_one_epoch

torch.manual_seed(0)

B, T = 4, 20
positions = torch.randint(0, 2, (B, T, 12, 8, 8)).float()
clocks = torch.randn(B, T)
lengths = torch.tensor([T, T - 3, T - 5, T - 1], dtype=torch.int)
targets = torch.randn(B, 2)

model = ChessEloPredictor(use_attention=True, attention_type="bahdanau", use_anomaly=True)
crit = nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# Exactly what train_one_epoch does internally.
opt.zero_grad()
_, outputs = model(positions, clocks, lengths)
loss = crit(outputs * 366 + 1514, targets * 366 + 1514)
loss.backward()

print("=== E1a: gradient reaching attention params on the TRAINING path ===")
any_attn_grad = False
for name, p in model.named_parameters():
    if name.startswith("attention"):
        g = p.grad
        has = g is not None and torch.count_nonzero(g).item() > 0
        any_attn_grad |= has
        print(f"  {name:36s} shape={tuple(p.shape)!s:14s} grad={'None' if g is None else f'{g.abs().sum().item():.3e}'}")
print(f"  -> any attention gradient? {any_attn_grad}")

print("\n=== E1b: control - CNN/LSTM/head params DO get gradient ===")
for name in ["conv1.weight", "conv4.weight", "lstm.weight_ih_l0", "fc1.weight", "fc2.weight"]:
    p = dict(model.named_parameters())[name]
    print(f"  {name:36s} grad_sum={p.grad.abs().sum().item():.3e}")

print("\n=== E1c: does the prediction change at all when attention weights change? ===")
model.eval()
with torch.no_grad():
    _, base = model(positions, clocks, lengths)
    # Perturb every attention parameter massively.
    for n, p in model.named_parameters():
        if n.startswith("attention"):
            p.add_(torch.randn_like(p) * 100.0)
    _, pert = model(positions, clocks, lengths)
delta = (base - pert).abs().max().item()
print(f"  max |pred_before - pred_after| after perturbing attention by N(0,100): {delta:.6e}")
print(f"  -> attention affects the rating prediction? {delta > 1e-9}")

print("\n=== E1d: same check for SelfAttention variant ===")
torch.manual_seed(0)
m2 = ChessEloPredictor(use_attention=True, attention_type="self")
m2.eval()
with torch.no_grad():
    _, b2 = m2(positions, clocks, lengths)
    for n, p in m2.named_parameters():
        if n.startswith("attention"):
            p.add_(torch.randn_like(p) * 100.0)
    _, p2 = m2(positions, clocks, lengths)
print(f"  max delta: {(b2 - p2).abs().max().item():.6e}")

print("\n=== E1e: attention param count vs total ===")
tot = sum(p.numel() for p in model.parameters())
att = sum(p.numel() for n, p in model.named_parameters() if n.startswith("attention"))
print(f"  total params={tot:,}  attention params={att:,} ({100*att/tot:.2f}%)")

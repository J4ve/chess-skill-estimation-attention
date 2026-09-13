"""E2: Trace the actual spatial shapes through the baseline CNN trunk.

Claim under test: the 4-block CNN pools 8x8 -> 1x1 before conv4, so conv4
operates on a single spatial cell and the model's "positional features" are
a global average over all 64 squares.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor

m = ChessEloPredictor()
x = torch.randn(8, 12, 8, 8)  # 8 board states

print("=== E2a: spatial shape trace through the CNN trunk ===")
print(f"  input                       {tuple(x.shape)}")
h = F.leaky_relu(m.bn1(m.conv1(x)));  print(f"  conv1+bn+lrelu              {tuple(h.shape)}")
h = m.pool(h);                        print(f"  avgpool                     {tuple(h.shape)}")
h = F.leaky_relu(m.bn2(m.conv2(h)));  print(f"  conv2+bn+lrelu              {tuple(h.shape)}")
h = m.pool(h);                        print(f"  avgpool                     {tuple(h.shape)}")
h = F.leaky_relu(m.bn3(m.conv3(h)));  print(f"  conv3+bn+lrelu              {tuple(h.shape)}")
h = m.pool(h);                        print(f"  avgpool                     {tuple(h.shape)}")
h = F.leaky_relu(m.bn4(m.conv4(h)));  print(f"  conv4+bn+lrelu              {tuple(h.shape)}   <-- LAST CONV")
print(f"  flattened per board         {h[0].numel()} features")

print("\n=== E2b: how much of conv4's 3x3 kernel touches real data? ===")
print("  conv4 input spatial size is 1x1 with padding=1 -> 3x3 window is")
print("  8/9 zero-padding, 1/9 real signal. Effective params used:")
w = m.conv4.weight  # (256,128,3,3)
print(f"  conv4.weight shape {tuple(w.shape)}, total {w.numel():,} params")
print(f"  params on the center tap actually seeing data: {w[:, :, 1, 1].numel():,}"
      f" ({100*w[:,:,1,1].numel()/w.numel():.1f}%)")

print("\n=== E2c: is per-square spatial information recoverable after the trunk? ===")
# Two boards that differ only by WHERE a piece is (same material) --
# does the 256-d CNN output distinguish them?
m.eval()
b1 = torch.zeros(1, 12, 8, 8); b1[0, 0, 6, 0] = 1  # white pawn a2
b2 = torch.zeros(1, 12, 8, 8); b2[0, 0, 6, 7] = 1  # white pawn h2  (mirror file)
b3 = torch.zeros(1, 12, 8, 8); b3[0, 0, 1, 0] = 1  # white pawn a7  (nearly promoting!)


def trunk(t):
    h = F.leaky_relu(m.bn1(m.conv1(t))); h = m.pool(h)
    h = F.leaky_relu(m.bn2(m.conv2(h))); h = m.pool(h)
    h = F.leaky_relu(m.bn3(m.conv3(h))); h = m.pool(h)
    h = F.leaky_relu(m.bn4(m.conv4(h)))
    return h.flatten()


with torch.no_grad():
    f1, f2, f3 = trunk(b1), trunk(b2), trunk(b3)
print(f"  ||f(pawn a2) - f(pawn h2)||  = {(f1-f2).norm().item():.6f}")
print(f"  ||f(pawn a2) - f(pawn a7)||  = {(f1-f3).norm().item():.6f}")
print(f"  ||f(pawn a2)||               = {f1.norm().item():.6f}")
print("  (untrained weights, so this measures representational capacity,")
print("   not learned behaviour: how strongly can the trunk encode WHERE?)")

print("\n=== E2d: parameter budget by block ===")
tot = sum(p.numel() for p in m.parameters())
for blk in ["conv1", "conv2", "conv3", "conv4", "lstm", "fc1", "fc2"]:
    n = sum(p.numel() for nm, p in m.named_parameters() if nm.startswith(blk))
    print(f"  {blk:8s} {n:9,}  ({100*n/tot:5.2f}%)")
print(f"  {'TOTAL':8s} {tot:9,}")

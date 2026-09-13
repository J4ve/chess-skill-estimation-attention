"""E6: cost model for the candidate CNN-side changes.

Measures parameter count and CPU forward+backward time for:
  A. baseline (conv_filters=32, 4 blocks, pool after every block)
  B. wider baseline (conv_filters=48/64) -- "increased filter capacity"
  C. deeper trunk that KEEPS 8x8 resolution (pool only twice, 6 conv layers)
  D. C + wider

The thesis promises a "deeper CNN with increased filter capacity"
(chapter3.tex:37) but never specifies it. This gives concrete options and
their compute cost relative to the ~12h/run HPC budget.
"""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor


class DeepTrunk(nn.Module):
    """Deeper CNN that preserves board geometry: pool only twice, not 3x.

    Baseline pools after every block, so conv4 runs on a 1x1 map (see E2).
    This variant keeps 8x8 for the first 3 convs, then pools down, so the
    later convs still see real spatial structure.
    """

    def __init__(self, f=32, out=256):
        super().__init__()
        c = [12, f, f, f * 2, f * 2, f * 4, out]
        self.convs = nn.ModuleList(
            [nn.Conv2d(c[i], c[i + 1], 3, padding=1) for i in range(6)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c[i + 1]) for i in range(6)])
        self.pool = nn.AvgPool2d(2, 2)
        self.out = out

    def forward(self, x):
        for i, (cv, bn) in enumerate(zip(self.convs, self.bns)):
            x = F.leaky_relu(bn(cv(x)))
            if i in (2, 4):            # pool after conv3 and conv5 only
                x = self.pool(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def baseline_trunk(f=32):
    m = ChessEloPredictor(conv_filters=f)
    def fwd(x):
        x = m.pool(F.leaky_relu(m.bn1(m.conv1(x))))
        x = m.pool(F.leaky_relu(m.bn2(m.conv2(x))))
        x = m.pool(F.leaky_relu(m.bn3(m.conv3(x))))
        return F.leaky_relu(m.bn4(m.conv4(x))).flatten(1)
    params = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("conv", "bn")))
    return fwd, params, m


torch.manual_seed(0)
B = 32 * 65           # a realistic batch: 32 games x ~65 plies
x = torch.randn(B, 12, 8, 8)

print(f"batch = {B} board states (32 games x 65 plies)\n")
print(f"{'variant':44s} {'cnn params':>12s} {'out dim':>8s} {'fwd+bwd':>10s} {'vs base':>8s}")
print("-" * 88)

results = {}
for name, builder in [
    ("A. baseline f=32 (pool x3, conv4 on 1x1)", lambda: baseline_trunk(32)),
    ("B1. wider f=48 (same topology)", lambda: baseline_trunk(48)),
    ("B2. wider f=64 (same topology)", lambda: baseline_trunk(64)),
]:
    fwd, params, m = builder()
    for _ in range(2):
        y = fwd(x); y.sum().backward()
    t = time.time()
    for _ in range(3):
        m.zero_grad(); y = fwd(x); y.sum().backward()
    dt = (time.time() - t) / 3
    results[name] = dt
    print(f"{name:44s} {params:>12,} {y.shape[1]:>8d} {dt:>9.2f}s "
          f"{dt/results['A. baseline f=32 (pool x3, conv4 on 1x1)']:>7.2f}x")

for name, f in [("C. deep 6-conv, pool x2 (keeps 8x8 early), f=32", 32),
                ("D. deep 6-conv, pool x2, f=48", 48)]:
    t_ = DeepTrunk(f)
    params = sum(p.numel() for p in t_.parameters())
    for _ in range(2):
        y = t_(x); y.sum().backward()
    tt = time.time()
    for _ in range(3):
        t_.zero_grad(); y = t_(x); y.sum().backward()
    dt = (time.time() - tt) / 3
    print(f"{name:44s} {params:>12,} {y.shape[1]:>8d} {dt:>9.2f}s "
          f"{dt/results['A. baseline f=32 (pool x3, conv4 on 1x1)']:>7.2f}x")

print("\nNOTE: CPU timings. GPU ratios differ (conv on 8x8 parallelises well),")
print("but the ORDERING and the parameter counts carry over. Use the ratio as")
print("an upper bound on the wall-clock cost against the ~12h/run budget.")

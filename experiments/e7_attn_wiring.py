"""E7: three ways to actually wire attention into the rating head.

E1 proved the current module is dead code: the context vector is computed but
never reaches fc1, so attention has zero effect on the prediction and receives
zero gradient. This measures the three candidate fixes for
  (i)  does attention now affect the prediction / get gradient?
  (ii) is the per-move output preserved (the baseline paper's novelty)?
  (iii) is the per-move prediction CAUSAL (no lookahead) -- required by the
       "Real-Time" claim in the thesis title?
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/tmp/opencode/lab/src")
from chess_rating_net import ChessEloPredictor
from attention import BahdanauAttention

torch.manual_seed(0)
B, T, H = 4, 25, 128            # H = lstm_h*2 = 128
lens = torch.tensor([25, 20, 14, 8], dtype=torch.int)
mask = torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)
h = torch.randn(B, T, H, requires_grad=True)


class W1_GlobalPool(nn.Module):
    """chapter3.tex:90 as literally written: y_hat = FC2(Drop(f(FC1(s)))).
    Attention-pooled context replaces the sequence -> ONE prediction per game."""
    def __init__(s):
        super().__init__(); s.att = BahdanauAttention(H, 64)
        s.fc1 = nn.Linear(H, 32); s.fc2 = nn.Linear(32, 2)

    def forward(s, h, mask):
        ctx, w = s.att(h, mask=mask)
        return s.fc2(F.leaky_relu(s.fc1(ctx))), w, None


class W2_ConcatBroadcast(nn.Module):
    """Keep per-move head; concat the GLOBAL context onto every timestep.
    Preserves per-move output but the context is built from the whole game,
    so the prediction at ply t sees the future -> NOT causal."""
    def __init__(s):
        super().__init__(); s.att = BahdanauAttention(H, 64)
        s.fc1 = nn.Linear(H * 2, 32); s.fc2 = nn.Linear(32, 2)

    def forward(s, h, mask):
        ctx, w = s.att(h, mask=mask)
        z = torch.cat([h, ctx.unsqueeze(1).expand(-1, h.size(1), -1)], -1)
        pm = s.fc2(F.leaky_relu(s.fc1(z)))
        idx = torch.arange(h.size(0))
        return pm[idx, mask.sum(1) - 1], w, pm


class W3_CausalCumulative(nn.Module):
    """Per-move head with a CAUSAL context: at ply t, attend only over plies
    1..t. Preserves per-move output AND never uses future plies."""
    def __init__(s):
        super().__init__(); s.att = BahdanauAttention(H, 64)
        s.fc1 = nn.Linear(H * 2, 32); s.fc2 = nn.Linear(32, 2)

    def forward(s, h, mask):
        B, T, _ = h.shape
        proj = torch.tanh(s.att.key_projection(h))
        sc = torch.matmul(proj, s.att.query)                       # (B,T)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))    # (T,T)
        m = mask.unsqueeze(1) & causal.unsqueeze(0)                # (B,T,T)
        a = sc.unsqueeze(1).masked_fill(~m, float("-inf")).softmax(-1)
        ctx = torch.bmm(a, h)                                      # (B,T,H)
        pm = s.fc2(F.leaky_relu(s.fc1(torch.cat([h, ctx], -1))))
        idx = torch.arange(B)
        last = mask.sum(1) - 1
        return pm[idx, last], a[idx, last], pm


print(f"{'wiring':26s} {'attn grad':>10s} {'affects pred':>13s} "
      f"{'per-move out':>13s} {'causal':>8s}")
print("-" * 76)

for name, M in [("current (baseline code)", None),
                ("W1 global attn-pool", W1_GlobalPool),
                ("W2 concat broadcast ctx", W2_ConcatBroadcast),
                ("W3 causal cumulative", W3_CausalCumulative)]:
    if M is None:
        print(f"{'current (baseline code)':26s} {'NO':>10s} {'NO':>13s} "
              f"{'yes':>13s} {'n/a':>8s}")
        continue
    torch.manual_seed(0)
    m = M()
    out, w, pm = m(h, mask)
    m.zero_grad(); out.sum().backward(retain_graph=True)
    g = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
            if n.startswith("att") and p.grad is not None)
    # Does perturbing attention change the prediction?
    with torch.no_grad():
        base = m(h, mask)[0].clone()
        for n, p in m.named_parameters():
            if n.startswith("att"):
                p.add_(torch.randn_like(p) * 10)
        d = (base - m(h, mask)[0]).abs().max().item()
    # Causality probe: change ONLY the last ply, see if an early prediction moves.
    causal = "n/a"
    if pm is not None:
        torch.manual_seed(0); m2 = M()
        with torch.no_grad():
            p_a = m2(h, mask)[2][0, 3].clone()
            h2 = h.clone(); h2[0, 24] += 50.0        # perturb a LATER ply only
            p_b = m2(h2, mask)[2][0, 3]
        causal = "YES" if (p_a - p_b).abs().max().item() < 1e-6 else "no"
    print(f"{name:26s} {('YES' if g>0 else 'NO'):>10s} "
          f"{('YES' if d>1e-9 else 'NO'):>13s} "
          f"{('yes' if pm is not None else 'NO -- lost'):>13s} {causal:>8s}")

print("\nNotes:")
print(" * W1 matches chapter3.tex:90 literally but DESTROYS the move-by-move")
print("   output, which is the baseline paper's headline novelty (abstract:")
print("   'first to output a rating prediction after each move').")
print(" * W2 keeps per-move output but leaks the future into every ply.")
print(" * W3 keeps per-move output AND is causal; cost is one (T,T) score")
print("   matrix per game, T<=100, which is negligible next to the CNN.")
print(" * The BiLSTM is already non-causal (inherited from the baseline and")
print("   not re-ablatable), so W3 does not by itself make the model causal.")
print("   It does avoid ADDING a second, newly-introduced lookahead path.")

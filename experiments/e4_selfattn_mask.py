"""E4: quantify the SelfAttention step_importance mask bug.

attention.py:99  step_importance = weights.mean(dim=(1, 2))
weights is (batch, heads, QUERY, KEY). Masking at :91 zeroes the KEY axis only
(unsqueeze(1).unsqueeze(1) broadcasts the mask over queries). Averaging over
dim=2 therefore divides by seq_len (padded), not by the real length -- so the
reported per-move importances are scaled DOWN by real_len/padded_len and no
longer sum to 1. The anomaly detector then renormalizes (anomaly.py:66-67),
which rescues the aggregate but NOT any downstream use of the raw weights.
"""
import sys, torch
sys.path.insert(0, "/tmp/opencode/lab/src")
from attention import SelfAttention, BahdanauAttention

torch.manual_seed(0)
sa = SelfAttention(hidden_dim=128)
ba = BahdanauAttention(hidden_dim=128)

print(f"{'padded_T':>9} {'real_T':>7} | {'Bahdanau sum':>13} | {'SelfAttn sum':>13} {'expected':>9}")
print("-" * 62)
for padded_T, real_T in [(100, 100), (100, 80), (100, 50), (100, 20), (100, 10)]:
    h = torch.randn(1, padded_T, 128)
    mask = torch.zeros(1, padded_T, dtype=torch.bool)
    mask[0, :real_T] = True
    _, wb = ba(h, mask=mask)
    _, ws = sa(h, mask=mask)
    print(f"{padded_T:>9} {real_T:>7} | {wb.sum().item():>13.4f} | "
          f"{ws.sum().item():>13.4f} {real_T/padded_T:>9.4f}")

print("\n-> BahdanauAttention weights always sum to 1.0 (correct).")
print("-> SelfAttention step_importance sums to real_T/padded_T, i.e. it is")
print("   silently scaled by the batch's padding ratio. Two identical games")
print("   batched with different neighbours get different importance scales.")

print("\n=== E4b: same game, two different batch companions ===")
torch.manual_seed(1)
game = torch.randn(30, 128)
for companion_len in [30, 100]:
    T = max(30, companion_len)
    h = torch.zeros(1, T, 128); h[0, :30] = game
    mask = torch.zeros(1, T, dtype=torch.bool); mask[0, :30] = True
    _, w = sa(h, mask=mask)
    print(f"  batched with a {companion_len}-ply game -> importance of ply 0 = "
          f"{w[0, 0].item():.6f}, sum={w.sum().item():.4f}")
print("  -> the SAME game gets different per-move importances depending on")
print("     which games happen to share its batch. Not batch-invariant.")

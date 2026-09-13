"""E4b (CORRECTION): my E4 hypothesis was WRONG. Recording the refutation.

Predicted: SelfAttention.step_importance sums to real_T/padded_T.
Measured : it sums to 1.0 exactly, same as Bahdanau.

Why: masking at attention.py:91 puts -inf on the KEY axis, so EVERY query row
(padded ones included) softmaxes to a distribution summing to 1 over keys.
Averaging those rows (dim=2) therefore preserves the sum. No scaling bug.

Residual real effect: padded QUERY rows are still averaged in. Those rows come
from zeroed hidden states, so they contribute a spurious, roughly uniform
distribution that dilutes the real signal. Quantified below.
"""
import sys, torch
sys.path.insert(0, "/tmp/opencode/lab/src")
from attention import SelfAttention

torch.manual_seed(0)
sa = SelfAttention(hidden_dim=128).eval()

print("Effect of including padded query rows in the dim=2 average")
print(f"{'padded_T':>9} {'real_T':>7} | {'L1 drift':>10} {'max drift':>10} {'rel %':>8}")
print("-" * 52)
for padded_T, real_T in [(100, 90), (100, 60), (100, 30), (100, 15)]:
    h = torch.randn(1, padded_T, 128)
    h[0, real_T:] = 0.0                       # what pad_packed_sequence produces
    mask = torch.zeros(1, padded_T, dtype=torch.bool); mask[0, :real_T] = True
    with torch.no_grad():
        _, w_buggy = sa(h, mask=mask)         # averages over ALL query rows
        # Correct version: average over real query rows only.
        q = sa.q_proj(h).view(1, padded_T, 1, 128).transpose(1, 2)
        k = sa.k_proj(h).view(1, padded_T, 1, 128).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)
        s = s.masked_fill(~mask.bool().unsqueeze(1).unsqueeze(1), float("-inf"))
        ww = torch.softmax(s, dim=-1)
        w_fixed = ww[:, :, :real_T, :].mean(dim=(1, 2))
    l1 = (w_buggy - w_fixed).abs().sum().item()
    mx = (w_buggy - w_fixed).abs().max().item()
    print(f"{padded_T:>9} {real_T:>7} | {l1:>10.5f} {mx:>10.6f} {100*l1/2:>7.2f}%")

print("\nVerdict: real but small (a few percent of total attention mass) on")
print("untrained weights. It is a correctness wart worth a one-line fix, NOT a")
print("credible source of MAE. Do not sell it as an accuracy improvement.")

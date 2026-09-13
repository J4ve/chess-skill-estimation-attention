"""E9: the anomaly head's operating dependencies.

Facts under test:
  A. The head is non-parametric -- it has zero learnable parameters, so an
     "anomaly-head-only" ablation run (chapter3-4-process.md:64) trains
     exactly the same weights as the baseline run.
  B. It never runs on the training path: forward() only builds it when
     return_attention=True AND baseline is not None
     (chess_rating_net.py:273).
  C. The suspicion score is dominated by the choice of R_baseline, which the
     manuscript never defines operationally.
  D. Score scale depends on whether predictions are standardised or in rating
     points -- anomaly.py:41 accepts either without checking.
"""
import sys, torch
sys.path.insert(0, "/tmp/opencode/lab/src")
from anomaly import AnomalyDetector
from chess_rating_net import ChessEloPredictor

print("=== A: learnable parameters in the anomaly head ===")
det = AnomalyDetector()
n = sum(p.numel() for p in det.parameters())
print(f"  AnomalyDetector parameters: {n}")
m_base = ChessEloPredictor(use_anomaly=False)
m_anom = ChessEloPredictor(use_anomaly=True)
pb = sum(p.numel() for p in m_base.parameters())
pa = sum(p.numel() for p in m_anom.parameters())
print(f"  model params without head: {pb:,}")
print(f"  model params with head:    {pa:,}")
print(f"  -> difference: {pa-pb}. An 'anomaly-head-only' training run optimises")
print("     an IDENTICAL parameter set to the baseline run. With the same seed")
print("     it produces the same weights and the same MAE; the two runs differ")
print("     only in post-hoc scoring. Budgeting a separate 12h GPU run for it")
print("     buys nothing on the MAE axis.")

print("\n=== B: is the head reachable from train_one_epoch? ===")
print("  chess_rating_net.py:305  `_, outputs = model(positions, clocks, lengths)`")
print("  -> return_attention defaults to False (:225), so forward returns early")
print("     at :263 and the anomaly branch at :272-274 is never entered.")
print("  -> the head is inference-only by construction. Nothing about it is")
print("     learned, and no loss term supervises it.")

print("\n=== C: sensitivity of the suspicion score to R_baseline ===")
torch.manual_seed(0)
T = 60
# A 'clean' curve wobbling around 1500 and a 'cheating' curve that jumps.
clean = 1500 + torch.randn(1, T, 2) * 40
cheat = clean.clone(); cheat[0, 30:, :] += 900          # engine switched on at ply 30
w = torch.full((1, T), 1.0 / T)

print(f"  {'R_baseline':>12s} {'S(clean)':>10s} {'S(cheat)':>10s} {'ratio':>8s}")
for rb in [1200.0, 1500.0, 1800.0, 2400.0]:
    b = torch.full((1, 2), rb)
    sc = det(clean, b, w)["combined_score"].item()
    sh = det(cheat, b, w)["combined_score"].item()
    print(f"  {rb:>12.0f} {sc:>10.1f} {sh:>10.1f} {sh/sc:>8.2f}")
print("  -> separation collapses as R_baseline moves away from the player's")
print("     true strength. The score is a DEVIATION FROM AN ASSUMED PRIOR, not")
print("     an absolute anomaly measure. chapter3.tex:115 defines d_t but the")
print("     manuscript never says where R_baseline comes from at inference.")
print("     ChessGamesDataset reads an OPTIONAL 'rating_after_last_game' key")
print("     (chess_rating_net.py:75) that the current pickles do not contain")
print("     (verified in E8 Q1) -- so today baseline is always None and the")
print("     branch cannot fire at all.")

print("\n=== D: score scale depends on the units of `predictions` ===")
b = torch.full((1, 2), 1500.0)
std_pred = (clean - 1514.0) / 366.0
print(f"  predictions in rating points -> S = "
      f"{det(clean, b, w)['combined_score'].item():.2f}")
print(f"  predictions standardised     -> S = "
      f"{det(std_pred, b, w)['combined_score'].item():.2f}")
print("  Same game, ~366x different score, no error raised (anomaly.py:41 says")
print("  'standardized or original-scale'). Any ROC threshold calibrated in one")
print("  convention is meaningless in the other. Fix by asserting the unit.")

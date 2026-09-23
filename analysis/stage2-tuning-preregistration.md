# Stage-2 Hyperparameter Tuning — Pre-registration

**Date recorded:** 2026-08-16 (committed before the seed-rerun test MAE was computed).
**Status of the experiment at this timestamp:** the seed-controlled 170k rerun is COMPLETE
(10/10, all 60 epochs). Best-validation **loss** is known: baseline 224.24 ± 1.64, attention
223.38 ± 0.87 (sample std, n−1) across 5 seeds — a statistical tie. The headline **test MAE** (rating points)
has NOT yet been computed; the eval pass is running separately. No stage-2 tuning run has
started.

## Purpose

This document pre-registers the hyperparameter tuning procedure for Stage 2 of the
control-and-tune strategy (Chapter 3). Writing it before the headline test MAE exists — and
before any tuning run starts — is what makes the tuning defensible as planned exploration
rather than post-hoc search.

## Trigger condition (pre-registered)

Stage 2 fires only if the stage-1 attention mean does **not** clearly beat the baseline
mean. With stage-1 val loss at 224.24 ± 1.64 (baseline) vs 223.38 ± 0.87 (attention), the
gap is smaller than the seed-to-seed spread — the trigger is met. Final confirmation waits
on the test MAE, which must be computed before any tuning run begins.

## Budget (declared in advance, not extended)

17 runs at 170k, one seed per cell, ≈ 18 h wall on 4 GPUs. Tune in this fixed order, one
factor at a time:

| # | Hyperparameter | Grid | Runs |
|---|---|---|---|
| 1 | early-stopping patience | {5, 10, none} | 3 |
| 2 | weight decay | {1e-5, 1e-3, 1e-2} | 3 |
| 3 | learning rate | {5e-5, 1e-4, 3e-4} | 3 |
| 4 | batch size | {32, 64, 128} | 3 |
| 5 | dropout | {0.3, 0.5} | 2 |
| 6 | attention_dim | {32, 64, 128} | 3 |

The winner of the search is re-run at 5 seeds before any claim is made.

## Scientific guardrails (non-negotiable)

- Tune on the **validation** split only. Never peek at test MAE to choose the next run.
  Report the test MAE once, at the end, on the frozen split.
- Report **all runs**, not the winner. The full cell table is the evidence against
  cherry-picking.
- Report mean ± std across seeds for the adopted configuration, and state whether its
  confidence interval crosses the stage-1 attention mean.
- Any tuned result is reported as a **separate row** from the control run, per
  `chapter3.tex` control-and-tune requirements.

## Success criterion (pre-registered)

A configuration is adopted only if it beats the stage-1 attention mean by more than
**2 × σ_seed** (where σ_seed is the seed-to-seed standard deviation from the stage-1
5-seed rerun). Anything smaller is indistinguishable from restarting training.

## Checkpoint-selection rule (pre-registered)

Best-validation checkpoint is the primary reported number; the terminal epoch-60 number is
reported alongside, since it is the one that checkpoint selection cannot inflate.

## Honest interpretation (agreed in advance)

If no configuration clears 2 × σ_seed, we say so plainly. The defensible finding in that
case is: "Attention did not measurably improve rating MAE at this scale; its contribution is
the interpretable per-move weighting that the anomaly module requires."

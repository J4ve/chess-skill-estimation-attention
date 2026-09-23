# Anomaly decision-point eval results (computed suspicion score, Objective 3)

The results file named in the 2026-08-30 plan (`analysis/anomaly-eval-plan-2026-08-30.md`).
Runs a-c finished on HPC 2026-09-11 (tmux run via `run_anomaly_decision_eval.sh`). Run d, the
constant-clock control, had been interrupted and never wrote its output; it was re-run on its own
2026-09-13 (tmux `anomd`, GPU 2, log `logs/anomaly_decision_eval_d.log`).

## Setup

- Host and path: `<hpc-host>`, `~/<workdir>/`, conda env `ratingnet2`.
- Script: `eval_anomaly_v3.py`.
- Checkpoint: `models/preflight_check_2m/best_model.pth` (the frozen tuned-attention arm).
- Corpus: `data/anomaly_corpus/` (90,000 games, 1,000 per cell, all 90 cells complete).
- Subset: bands 1100, 1300, 1500, 1700, 1900; engines lc0 and stockfish16; all 5 rates;
  `--games_per_dir 80`. Clean cells split 50/50 (seed 42): the calibration half sets the FPR
  thresholds and the empirical baseline, the other half is scored.
- Scored games: 3,600 (400 clean, 3,200 substituted, 88.9 percent substituted).

## Headline: game-level ROC-AUC

95 percent CIs by Hanley-McNeil; every entry has SE about 0.015.

| Run | Clocks | Baseline | S_att | S_max | S_mean |
|---|---|---|---|---|---|
| a | original engine time | empirical | 0.509 [0.479, 0.539] | 0.538 | 0.543 |
| b | synthesized, label-blind | nominal | 0.555 [0.527, 0.584] | 0.531 | 0.511 |
| c | synthesized, label-blind | empirical | 0.554 [0.525, 0.583] | 0.544 | 0.533 |
| d | constant (control) | nominal | 0.509 [0.479, 0.539] | 0.506 | 0.487 |

Precision at calibrated FPR, S_att: b 0.920 (1 percent) / 0.942 (5 percent); c 0.963 / 0.946;
a 0.895 / 0.892; d 0.877 / 0.904. Base rate is 0.889, so precision here is uninformative.

## Breakdowns (S_att, run b)

| Cut | Values |
|---|---|
| Substitution rate (vs held-out clean) | 5%: 0.527 [0.492, 0.561]; 15%: 0.532 [0.498, 0.567]; 30%: 0.570 [0.536, 0.603]; 60%: 0.593 [0.560, 0.627] |
| Engine | lc0 0.565 [0.525, 0.606]; stockfish16 0.547 [0.506, 0.588] |
| Maia band | 1100 0.624; 1300 0.526; 1500 0.622; 1700 0.474; 1900 0.544 (80 clean per band, about +/- 0.06) |
| Ply-count strata | 0-60: 0.545; 60-80: 0.567; 80-99: 0.577; 99-101: 0.543 |
| Mean plies | clean 79.5; 5%: 74.4; 15%: 77.2; 30%: 76.5; 60%: 72.0 |

Move level: substituted/clean mean deviation ratio a 1.02, b 1.04, c 1.04, d 1.00. Top-5 by
alpha*d hit rate on substituted moves: a 0.148, b 0.144, c 0.147, d 0.150.

Empirical clean baselines are far above nominal (printed per band/engine in the log), for example
1700_lc0 2107.1 and 1700_stockfish16 2076.3.

## Reading

- Runs b and c are statistically above chance, but only slightly. No pre-set AUC threshold applies
  (results are reported descriptively).
- The effect is weak and absent at 5 and 15 percent substitution.
- a and d at chance, b and c above: the gain comes from putting the clock input back in the
  training range. The synthesized clocks are seeded by band and game index only, so they cannot
  carry the label.
- S_max and S_mean are no stronger than S_att, and top-k localization is no better than the
  constant control, so this is not attention downweighting substituted moves.
- Most likely cause: domain gap. The human-trained rating model reads Maia as far stronger than its
  nominal band, so a substituted engine move is a small relative change.

## Loose end, not chased

Run a was meant as a regression anchor reproducing the 2026-08-26 figure of 0.3438. It gave 0.509.
The 0.3438 run used a different 2,700-game subset and `eval_anomaly_v2.py`. The manuscript reports
only the v3 numbers.

## Not done

- Kaggle Chess Cheating Dataset cross-evaluation: not run.
- Full 90,000-game run: not run; the subset result did not justify it.

## Files

- Local copies: `analysis/anomaly_decision_eval/anomaly_v3_{a_anchor,b_treat_nom,c_treat_emp,d_control}.json`
- HPC originals: `~/<workdir>/anomaly_v3_*.json`, logs `logs/anomaly_decision_eval.log`,
  `logs/anomaly_decision_eval_d.log`
- Clock synthesis evidence: `analysis/clock_params.json`, `analysis/clock_ks_report.json` (on HPC)

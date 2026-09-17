# Suspicion cutoffs for all four methods (2026-09-17)

Typical / Unusual / Highly unusual cutoffs for every selectable suspicion-score
method, written together to `src/static/suspicion_cutoffs.json` under
`"methods"`, keyed by method id. Each entry names its own score in `"score"`,
and `api._load_method_cutoffs` drops an entry whose score does not match its
key, so one method's cutoffs can never label another method's score.

## Method

1. **Same games for every method.** `score_method_cutoffs.py` reads the 2,822
   games of the per-move detector's cutoffs run
   (`experiments/detector_cutoffs/`, HPC
   `scratch/detector-default/cutoffs/per_side_scores.csv`, copied as
   `reference_scores.csv`) instead of drawing a new sample: stratified
   held-out TEST-partition games, seed 20260915, at least 20 plies.
2. **Scoring** (HPC CPU, tmux). Each pickle goes through the frozen rating
   checkpoint (md5 `fcb38b8a6a3ee04a6b1859dd5bc69b75`) once, as
   `api._run_inference` does; then S_att (`anomaly.py`), the LightGBM detector
   (`src/lgbm_detector.py`), the per-move detector (`src/detector.py`) and the
   CNN-BiLSTM detector (`src/cnn_bilstm_detector.py`), each side against its
   PGN-header rating, using this repo's own code copied next to the script.
   All 2,822 games scored with no errors (`progress.txt`). The per-move
   detector's scores match the reference run to 5.0e-07, and its cutoffs below
   are identical to the ones it shipped with.
3. **Cutoffs** (`cutoffs_from_scores.py`, local, from `per_side_scores.csv`).
   75th and 95th percentiles (linear interpolation) overall and per time
   control, 95% bootstrap CIs from 2,000 resamples, the same seed per method;
   a time control with fewer than 300 sides would fall back to the overall
   cutoff (none needed it).

This replaces the separate S_att cutoffs file (`suspicion_cutoffs_s_att.json`,
from `experiments/satt_cutoffs/`: 2,536 games from a draw of up to 600 per
fast time control rather than 700), so all four methods are now labelled
against the same games. The two S_att results agree closely (overall p75
382.17 and p95 634.11 there, 379.27 and 629.16 here, each inside the other's
95% CI).

## Result

### Computed score S_att (rating points)

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,644 | 379.27 | [372.19, 386.67] | 629.16 | [615.59, 642.68] |
| Blitz | 1,330 | 364.37 | [351.09, 374.45] | 590.80 | [568.93, 617.99] |
| Bullet | 1,346 | 306.36 | [294.90, 328.17] | 526.81 | [487.43, 558.12] |
| Classical | 1,230 | 419.51 | [404.76, 439.48] | 678.49 | [646.56, 704.02] |
| Rapid | 1,306 | 410.57 | [395.49, 425.85] | 661.36 | [636.21, 693.67] |
| Ultrabullet | 432 | 408.87 | [386.69, 441.69] | 633.99 | [601.15, 680.66] |

### LightGBM detector (A0g)

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,644 | 0.5807 | [0.5741, 0.5884] | 0.7656 | [0.7587, 0.7723] |
| Blitz | 1,330 | 0.5555 | [0.5438, 0.5747] | 0.7393 | [0.7245, 0.7708] |
| Bullet | 1,346 | 0.5994 | [0.5877, 0.6125] | 0.7810 | [0.7703, 0.7997] |
| Classical | 1,230 | 0.5712 | [0.5489, 0.5839] | 0.7480 | [0.7308, 0.7626] |
| Rapid | 1,306 | 0.5521 | [0.5399, 0.5747] | 0.7499 | [0.7333, 0.7623] |
| Ultrabullet | 432 | 0.6789 | [0.6611, 0.7017] | 0.8145 | [0.7888, 0.8255] |

### Per-move detector (A3g, default)

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,644 | 0.4805 | [0.4678, 0.4929] | 0.8227 | [0.8084, 0.8336] |
| Blitz | 1,330 | 0.4841 | [0.4680, 0.5040] | 0.7862 | [0.7567, 0.8200] |
| Bullet | 1,346 | 0.4949 | [0.4669, 0.5184] | 0.8317 | [0.7941, 0.8522] |
| Classical | 1,230 | 0.4683 | [0.4455, 0.4986] | 0.8378 | [0.8172, 0.8609] |
| Rapid | 1,306 | 0.4717 | [0.4433, 0.5028] | 0.8105 | [0.7811, 0.8351] |
| Ultrabullet | 432 | 0.4722 | [0.4398, 0.4956] | 0.8309 | [0.7873, 0.8675] |

### CNN-BiLSTM detector (A4)

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,644 | 0.6275 | [0.6186, 0.6371] | 0.8111 | [0.8033, 0.8192] |
| Blitz | 1,330 | 0.6115 | [0.5935, 0.6260] | 0.8074 | [0.7867, 0.8234] |
| Bullet | 1,346 | 0.5988 | [0.5790, 0.6115] | 0.7826 | [0.7622, 0.8012] |
| Classical | 1,230 | 0.6796 | [0.6545, 0.6978] | 0.8345 | [0.8246, 0.8488] |
| Rapid | 1,306 | 0.6235 | [0.6099, 0.6407] | 0.7904 | [0.7734, 0.8111] |
| Ultrabullet | 432 | 0.6660 | [0.6275, 0.7053] | 0.8317 | [0.8050, 0.8576] |

## Bundled samples under each method

Through `api._run_inference` with these cutoffs (scores for the trained methods
are the parity-checked values in `tests/regression/expected_sample_scores.json`):

| Sample | Time control | S_att W / B | LightGBM W / B | Per-move W / B | CNN-BiLSTM W / B |
| --- | --- | --- | --- | --- | --- |
| bullet-best-predicted | bullet | 67 Typical / 67 Typical | 0.705 Unusual / 0.293 Typical | 0.423 Typical / 0.178 Typical | 0.404 Typical / 0.344 Typical |
| bullet-typical | bullet | 264 Typical / 264 Typical | 0.571 Typical / 0.574 Typical | 0.462 Typical / 0.500 Unusual | 0.510 Typical / 0.581 Typical |
| bullet-worst-predicted | bullet | 366 Unusual / 447 Unusual | 0.780 Unusual / 0.397 Typical | 0.448 Typical / 0.150 Typical | 0.887 Highly unusual / 0.888 Highly unusual |
| blitz-best-predicted | blitz | 99 Typical / 99 Typical | 0.550 Typical / 0.441 Typical | 0.338 Typical / 0.425 Typical | 0.571 Typical / 0.561 Typical |
| blitz-typical | blitz | 376 Unusual / 348 Typical | 0.696 Unusual / 0.395 Typical | 0.893 Highly unusual / 0.256 Typical | 0.682 Unusual / 0.112 Typical |
| blitz-worst-predicted | blitz | 1758 Highly unusual / 1681 Highly unusual | 0.416 Typical / 0.694 Unusual | 0.665 Unusual / 0.340 Typical | 0.592 Typical / 0.352 Typical |
| rapid-best-predicted | rapid | 371 Typical / 372 Typical | 0.433 Typical / 0.628 Unusual | 0.368 Typical / 0.546 Unusual | 0.485 Typical / 0.703 Unusual |
| rapid-typical | rapid | 235 Typical / 219 Typical | 0.366 Typical / 0.595 Unusual | 0.322 Typical / 0.293 Typical | 0.671 Unusual / 0.764 Unusual |
| rapid-worst-predicted | rapid | 1007 Highly unusual / 1293 Highly unusual | 0.453 Typical / 0.537 Typical | 0.386 Typical / 0.800 Unusual | 0.542 Typical / 0.387 Typical |
| synthetic-caught | bullet | 1198 Highly unusual / 1200 Highly unusual | 0.743 Unusual / 0.393 Typical | 0.949 Highly unusual / 0.277 Typical | 0.813 Highly unusual / 0.550 Typical |
| synthetic-false-alarm | bullet | 132 Typical / 132 Typical | 0.369 Typical / 0.539 Typical | 0.236 Typical / 0.892 Highly unusual | 0.252 Typical / 0.345 Typical |
| synthetic-missed | blitz | 282 Typical / 284 Typical | 0.415 Typical / 0.388 Typical | 0.386 Typical / 0.250 Typical | 0.495 Typical / 0.396 Typical |

The four methods often disagree on the same side, which is expected from
detectors that each separate synthetic engine games only moderately (withheld
band ROC-AUC 0.51 to 0.75).

## Caveats

- These cutoffs describe how each score is distributed on ordinary human games,
  not a cheat-detection threshold. About 5 percent of ordinary sides land in
  Highly unusual by construction, under every method.
- The trained detectors trained only on synthetic Maia games; their measured
  results come from that corpus, not from real cheating cases.
- A reviewer-entered baseline changes S_att and the LightGBM and per-move
  detectors' features, and so their scores. The CNN-BiLSTM detector reads only
  positions and clocks, so the baseline does not affect its score.

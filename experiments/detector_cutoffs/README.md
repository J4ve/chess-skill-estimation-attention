# Detector score cutoffs (2026-09-17)

> Superseded on 2026-09-17 by `experiments/method_cutoffs/`: the combined
> `src/static/suspicion_cutoffs.json` now holds these per-move detector cutoffs
> (unchanged) alongside the other three methods, and `suspicion_cutoffs_s_att.json`
> is gone.

Cutoffs for the trained detector's Typical / Unusual / Highly unusual labels,
written to `src/static/suspicion_cutoffs.json` (`"score": "detector_a3g_seed0"`).
The computed score S_att keeps its own, still provisional, cutoffs in
`src/static/suspicion_cutoffs_s_att.json`; the API refuses a cutoffs file whose
`score` field names a different score.

## Method

Same approach as the S_att cutoffs, with one change to reach about 3,000 games.

1. **Corpus.** Held-out TEST-partition games listed in
   `analysis/heldout_test_eval/attn_tuned__best.csv` on the HPC checkout,
   restricted to the 13 months whose `data/processed_games/<month>/` directories
   are non-empty.
2. **Stratified sample (seed 20260915).** Up to 700 games each for bullet, blitz
   and rapid (the S_att run used 600), every available classical (690) and
   ultrabullet (228) game: 3,018 sampled.
3. **Scoring** (`score_detector_cutoffs.py`, HPC CPU, tmux). Each pickle's
   positions, clocks and moves go through the frozen rating checkpoint
   (md5 `fcb38b8a6a3ee04a6b1859dd5bc69b75`) exactly as `api._run_inference` does,
   then through this repo's own `src/detector.py` with each side as the suspect
   against its PGN-header rating. Games under 20 plies are skipped: 2,822 games
   scored (5,644 sides), 196 skipped as too short, no other skips.
   The full sample completed, so the file is not provisional.
4. **Cutoffs** (`cutoffs_from_scores.py`). 75th and 95th percentiles (linear
   interpolation) overall and per time control, 95% bootstrap CIs from 2,000
   resamples; a time control with fewer than 300 sides falls back to the
   overall cutoff (none needed it).

## Result

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,644 | 0.4805 | [0.4678, 0.4929] | 0.8227 | [0.8084, 0.8336] |
| Bullet | 1,346 | 0.4949 | [0.4669, 0.5184] | 0.8317 | [0.7941, 0.8522] |
| Blitz | 1,330 | 0.4841 | [0.4680, 0.5040] | 0.7862 | [0.7567, 0.8200] |
| Rapid | 1,306 | 0.4717 | [0.4433, 0.5028] | 0.8105 | [0.7811, 0.8351] |
| Classical | 1,230 | 0.4683 | [0.4455, 0.4986] | 0.8378 | [0.8172, 0.8609] |
| Ultrabullet | 432 | 0.4722 | [0.4398, 0.4956] | 0.8309 | [0.7873, 0.8675] |

Per-time-control cutoffs sit close together (p75 about 0.47 to 0.49, p95
about 0.79 to 0.84), so the detector score is roughly comparable across time
controls on ordinary games.

## Sanity check: bundled samples

Through `api._run_inference` with these cutoffs:

| Sample | Time control (cutoffs) | White | White label | Black | Black label |
| --- | --- | --- | --- | --- | --- |
| bullet-best-predicted | bullet (own) | 0.423 | Typical | 0.178 | Typical |
| bullet-typical | bullet (own) | 0.462 | Typical | 0.500 | Unusual |
| bullet-worst-predicted | bullet (own) | 0.448 | Typical | 0.150 | Typical |
| blitz-best-predicted | blitz (own) | 0.338 | Typical | 0.425 | Typical |
| blitz-typical | blitz (own) | 0.893 | Highly unusual | 0.256 | Typical |
| blitz-worst-predicted | blitz (own) | 0.665 | Unusual | 0.340 | Typical |
| rapid-best-predicted | rapid (own) | 0.368 | Typical | 0.546 | Unusual |
| rapid-typical | rapid (own) | 0.322 | Typical | 0.293 | Typical |
| rapid-worst-predicted | rapid (own) | 0.386 | Typical | 0.800 | Unusual |
| synthetic-caught | bullet (own) | 0.949 | Highly unusual | 0.277 | Typical |
| synthetic-false-alarm | bullet (own) | 0.236 | Typical | 0.892 | Highly unusual |
| synthetic-missed | blitz (own) | 0.386 | Typical | 0.250 | Typical |
| synthetic-false-alarm-s-att (archived) | bullet (own) | 0.355 | Typical | 0.297 | Typical |
| synthetic-missed-s-att (archived) | rapid (own) | 0.172 | Typical | 0.877 | Highly unusual |


## Caveats

- These cutoffs describe the detector score on ordinary human games, not a
  cheat-detection threshold. About 5 percent of ordinary sides land in Highly
  unusual by construction.
- The detector trained only on synthetic Maia games. Its measured results
  (ROC-AUC about 0.75 on withheld rating bands, about 0.58 and 0.61 at 2 and
  5 percent substitution) come from that synthetic corpus, not from real
  cheating cases.
- Each side is scored against its PGN-header rating; a reviewer-entered baseline
  changes the features (`dev_signed`, `d_t`) and so the score.


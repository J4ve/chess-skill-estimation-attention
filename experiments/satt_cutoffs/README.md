# S_att cutoffs, full run (2026-09-17)

Cutoffs for the computed score S_att's Typical / Unusual / Highly unusual
labels, written to `src/static/suspicion_cutoffs_s_att.json` (`"score":
"s_att"`). This completes the run first reported at
`src/static/suspicion_cutoffs_s_att.json` on 2026-09-15, which was stopped
early after 1,200 of the planned ~2,700 games and shipped as provisional.
The trained detector's own cutoffs live separately in
`src/static/suspicion_cutoffs.json`; the API refuses a cutoffs file whose
`score` field names a different score.

## Method

Same approach and same seed as the 2026-09-15 provisional run, carried
through to completion.

1. **Corpus.** Held-out TEST-partition games listed in
   `analysis/heldout_test_eval/attn_tuned__best.csv` on the HPC checkout,
   restricted to the 13 months whose `data/processed_games/<month>/`
   directories are non-empty.
2. **Stratified sample (seed 20260915).** Up to 600 games each for bullet,
   blitz and rapid, every available classical (690) and ultrabullet (228)
   game: 2,718 sampled. Because the sampling is seeded and deterministic,
   this full run reselected the identical 2,718 game IDs as the provisional
   run, including its first 1,200; only games 1,201 to 2,718 needed scoring.
3. **Scoring** (`score_cutoffs.py`, HPC CPU, tmux). Each pickle's positions
   and clocks go through the frozen rating checkpoint (md5
   `fcb38b8a6a3ee04a6b1859dd5bc69b75`) exactly as `api._run_inference` does,
   then through the attention-weighted `AnomalyDetector` in `anomaly.py`.
   Games under 20 plies are skipped: 2,536 games scored (5,072 sides), 182
   skipped as too short, no other skips.
4. **Cutoffs** (`cutoffs_from_scores.py`). 75th and 95th percentiles (linear
   interpolation) overall and per time control, 95% bootstrap CIs from 2,000
   resamples; a time control with fewer than 300 sides falls back to the
   overall cutoff. Every time control cleared 300 sides this run (rapid,
   the only bucket that fell back in the provisional file at 110 sides, has
   1,120 sides now), so no bucket needs the fallback.

## Result

| Group | n (sides) | p75 | p75 95% CI | p95 | p95 95% CI |
| --- | --- | --- | --- | --- | --- |
| Overall | 5,072 | 382.17 | [373.12, 390.70] | 634.11 | [617.89, 647.34] |
| Bullet | 1,148 | 297.91 | [289.34, 318.20] | 524.38 | [474.89, 556.27] |
| Blitz | 1,142 | 367.88 | [354.07, 381.21] | 591.64 | [572.45, 624.84] |
| Rapid | 1,120 | 413.92 | [397.18, 428.61] | 662.12 | [634.65, 705.90] |
| Classical | 1,230 | 419.51 | [404.02, 439.31] | 678.49 | [644.53, 704.04] |
| Ultrabullet | 432 | 408.87 | [386.69, 439.33] | 633.99 | [600.27, 680.66] |

Bullet and blitz cutoffs are unchanged from the provisional file (their full
target was already scored by game 1,200). Faster time controls score lower:
bullet's p75/p95 sit well below rapid, classical and ultrabullet, which
cluster close together.

## Sanity check: bundled samples

Through `api._run_inference` with these cutoffs, on the HPC (same checkpoint,
same code path the app uses):

| Sample | Time control (cutoffs) | White S_att | White label | Black S_att | Black label |
| --- | --- | --- | --- | --- | --- |
| bullet-best-predicted | bullet (own) | 67.32 | Typical | 67.27 | Typical |
| bullet-typical | bullet (own) | 263.86 | Typical | 264.33 | Typical |
| bullet-worst-predicted | bullet (own) | 365.67 | Unusual | 447.04 | Unusual |
| blitz-best-predicted | blitz (own) | 98.64 | Typical | 98.92 | Typical |
| blitz-typical | blitz (own) | 376.31 | Unusual | 347.60 | Typical |
| blitz-worst-predicted | blitz (own) | 1757.87 | Highly unusual | 1681.36 | Highly unusual |
| rapid-best-predicted | rapid (own) | 371.02 | Typical | 372.48 | Typical |
| rapid-typical | rapid (own) | 235.28 | Typical | 219.21 | Typical |
| rapid-worst-predicted | rapid (own) | 1006.57 | Highly unusual | 1292.60 | Highly unusual |
| synthetic-caught | bullet (own) | 1198.14 | Highly unusual | 1200.34 | Highly unusual |
| synthetic-false-alarm | bullet (own) | 131.64 | Typical | 132.06 | Typical |
| synthetic-missed | blitz (own) | 281.94 | Typical | 283.72 | Typical |

The bundled `synthetic: false alarm` and `synthetic: missed` samples land
Typical on both sides under S_att, unlike under the trained detector (where
false-alarm's Black is Highly unusual and missed's Black is Highly unusual
too): this is the same weak separation the note field already documents
(ROC-AUC 0.555 in thesis evaluation), not a new finding.

## Caveats

- These cutoffs describe S_att on ordinary human games, not a cheat-detection
  threshold. About 5 percent of ordinary sides land in Highly unusual by
  construction.
- S_att separated engine-substituted synthetic games only weakly in thesis
  evaluation (ROC-AUC 0.555); a clean synthetic game scored 1097 on this
  scale. Prefer the trained detector's cutoffs
  (`experiments/detector_cutoffs/README.md`) for anything beyond illustrating
  how S_att behaves.
- Each side is scored against its PGN-header rating; a reviewer-entered
  baseline changes the attention-weighted deviation and so the score.

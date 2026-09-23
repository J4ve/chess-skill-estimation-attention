# BOT accounts among the frozen model's largest rating errors (2026-09-17)

Post-hoc check, not pre-registered. Prompted by the prototype sample-game selection
(firstmate `data/corpus-sample-games/report.md`), where the worst-predicted rapid and
blitz test games were BOT-vs-BOT games.

## Method

1. Per-game errors: `analysis/heldout_test_eval/attn_tuned__best.csv` on the HPC (frozen
   `models/preflight_check_2m/best_model.pth`, 255,000 test games; copied, not edited).
   Per-game error = mean(`white_err`, `black_err`).
2. `analysis/scripts/bot_check_groups.py` (run on the HPC from `~/<workdir>`,
   env `ratingnet2`, output under `scratch/bot-check/`): candidates in rank order for
   three groups (highest error, lowest error, `numpy.random.default_rng(20260917)`
   permutation), player names and ply counts read from the per-game pickles in
   `data/corpus_archive_transfer/<month>.tar.gz`, first 200 games with >= 40 plies kept
   per group. Rejected as too short: 162 (highest), 36 (lowest), 45 (random).
3. `analysis/scripts/bot_check_query.py` (local): `POST https://lichess.org/api/users`,
   300 names per call, 2 s between calls, descriptive User-Agent. All 1,194 distinct
   names were returned. BOT = `title == "BOT"`.

The corpus preprocessing (`prototype/src/preprocess_lichess.py::is_eligible`) has no
account-title filter, so BOT games are in train and test.

## Result (`analysis/bot_account_check/bot_result.json`)

| Group | Games with >= 1 BOT player | BOT-vs-BOT | BOT accounts / accounts | Error range (rating points) |
|---|---|---|---|---|
| 200 highest error | 22 (11.0%) | 18 | 36 / 395 | 779.7 to 1296.8 |
| 200 lowest error | 0 (0.0%) | 0 | 0 / 400 | 0.09 to 2.27 |
| 200 random | 1 (0.5%) | 1 | 2 / 400 | 5.5 to 676.9 |

One-sided Fisher exact test, highest vs random games with a BOT player: p = 1.6e-6
(computed in-session, not reported in the manuscript). Reading: strongly over-represented,
but a minority (about 1 in 9) of the largest errors. Per-game flags without usernames:
`analysis/bot_account_check/groups_bot_flags.json`. Usernames are kept only on the HPC
(`scratch/bot-check/groups.json`) and are not named in the manuscript.

Also seen but not written into the manuscript: the API's `tosViolation` flag was set on
24 accounts in the highest group vs 10 random and 8 lowest. That flag is not specific to
fair-play and no account may be described as cheating, so it is recorded here only.

## Prototype sample games (`analysis/bot_account_check/sample_scores.json`)

`analysis/scripts/bot_check_score_samples.py` scored every bundled sample PGN in
`prototype/src/static/samples/` through the app's own `/predict/pgn` (FastAPI TestClient,
prototype at submodule `4c7d7c5`, `RATINGNET_CHECKPOINT` = frozen checkpoint, CPU,
baselines from the PGN headers). The four players of `rapid_worst_predicted.pgn` and
`blitz_worst_predicted.pgn` carry the BOT title; the two players of
`bullet_best_predicted.pgn` do not.

| Sample | Stated W/B | Final estimate W/B | S_att W/B | Label |
|---|---|---|---|---|
| rapid_worst_predicted | 2744 / 3030 | 1720.52 / 1720.41 | 1006.57 / 1292.60 | highly_unusual (p95 cutoff 573.83) |
| blitz_worst_predicted | 2827 / 2748 | 1491.15 / 1490.36 | 1757.87 / 1681.36 | highly_unusual (p95 cutoff 591.64) |
| bullet_best_predicted | 1500 / 1500 | 1503.52 / 1502.63 | 67.32 / 67.27 | typical (p75 cutoff 297.91) |

The cutoffs file is marked provisional (1,200 test games).

# Detector parity check (2026-09-17)

Confirms the app's detector path (`src/detector.py` via `api._run_inference`)
reproduces the thesis HPC code that trained and evaluated arm A3g.

1. `build_kit.py` (HPC): turns the 12 bundled sample PGNs into a tiny v2-style
   corpus (each side in turn as the suspect, `maia_band` = that side's header
   Elo, parsed with the thesis `src/format_data.py`), and copies 8 real
   `anomaly_corpus_v2` games from the grouped split's test/heldout_band
   partitions, exporting them as PGN.
2. `extract_detector_features_v2.py --stage rating_cheap` (HPC, CPU) on that corpus.
3. `hpc_score.py` (HPC, CPU): `extra_arm_a3.build_ply_matrix`, the stored
   training standardization, `make_model` + `Batcher` + `predict` with the
   A3g weights.
4. `app_parity.py` (local): the same PGNs through `api._run_inference`, run
   from this directory with `index.json`, `hpc_scores.json` and
   `corpus_pgn/` copied from the HPC scratch directory.

Result: max |HPC code path - app code path| = 5.6e-07 over 32 side-scores;
the corpus games also match the original thesis GPU run
(`A3g_seed0/scores.npz`) to at most 7.7e-05.

| kind | game | suspect side | HPC code path | app code path | abs diff | thesis run (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| sample | blitz_best_predicted | white | 0.338154 | 0.338154 | 7.8e-08 |  |
| sample | blitz_best_predicted | black | 0.425406 | 0.425406 | 2.1e-08 |  |
| sample | blitz_typical | white | 0.893026 | 0.893026 | 1.2e-07 |  |
| sample | blitz_typical | black | 0.255620 | 0.255620 | 2.7e-09 |  |
| sample | blitz_worst_predicted | white | 0.664791 | 0.664791 | 7.2e-08 |  |
| sample | blitz_worst_predicted | black | 0.339782 | 0.339782 | 4.2e-07 |  |
| sample | bullet_best_predicted | white | 0.422638 | 0.422638 | 3.0e-07 |  |
| sample | bullet_best_predicted | black | 0.178359 | 0.178359 | 4.8e-07 |  |
| sample | bullet_typical | white | 0.462272 | 0.462272 | 4.4e-07 |  |
| sample | bullet_typical | black | 0.499602 | 0.499602 | 3.1e-07 |  |
| sample | bullet_worst_predicted | white | 0.447734 | 0.447734 | 3.9e-07 |  |
| sample | bullet_worst_predicted | black | 0.149612 | 0.149611 | 5.0e-07 |  |
| sample | rapid_best_predicted | white | 0.367735 | 0.367735 | 1.8e-07 |  |
| sample | rapid_best_predicted | black | 0.546354 | 0.546354 | 4.2e-07 |  |
| sample | rapid_typical | white | 0.321525 | 0.321525 | 2.0e-07 |  |
| sample | rapid_typical | black | 0.293129 | 0.293130 | 5.6e-07 |  |
| sample | rapid_worst_predicted | white | 0.385680 | 0.385680 | 7.0e-08 |  |
| sample | rapid_worst_predicted | black | 0.799591 | 0.799591 | 1.7e-07 |  |
| sample | synthetic_caught | white | 0.948874 | 0.948874 | 2.4e-07 |  |
| sample | synthetic_caught | black | 0.276653 | 0.276653 | 4.6e-07 |  |
| sample | synthetic_false_alarm | white | 0.355413 | 0.355413 | 1.9e-07 |  |
| sample | synthetic_false_alarm | black | 0.297366 | 0.297366 | 3.0e-07 |  |
| sample | synthetic_missed | white | 0.172342 | 0.172342 | 2.7e-08 |  |
| sample | synthetic_missed | black | 0.876502 | 0.876502 | 9.7e-08 |  |
| corpus | 1600_stockfish16_r00__game_00291 | black | 0.276599 | 0.276599 | 2.2e-07 | 0.276675 |
| corpus | 1500_lc0_r00__game_01825 | white | 0.575283 | 0.575283 | 2.3e-07 | 0.575317 |
| corpus | 1200_lc0_r60__game_00606 | black | 0.421589 | 0.421589 | 3.1e-07 | 0.421610 |
| corpus | 1300_stockfish16_r02__game_00002 | white | 0.121473 | 0.121473 | 3.7e-07 | 0.121504 |
| corpus | 1300_stockfish16_r05__game_01202 | white | 0.632383 | 0.632383 | 2.5e-07 | 0.632333 |
| corpus | 1700_lc0_r40__game_01002 | black | 0.367820 | 0.367820 | 3.5e-08 | 0.367813 |
| corpus | 1500_hardneg__game_01480 | white | 0.359076 | 0.359076 | 2.3e-08 | 0.359074 |
| corpus | 1300_lc0_r20__game_01178 | white | 0.850387 | 0.850387 | 1.4e-07 | 0.850391 |

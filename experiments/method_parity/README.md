# LightGBM and CNN-BiLSTM parity checks (2026-09-17)

Confirms the app's two newly ported suspicion-score methods reproduce the
thesis HPC code that trained and evaluated them, the same way
`experiments/detector_parity/` did for the per-move detector:

- **LightGBM detector** (thesis arm A0g, `src/lgbm_detector.py`);
- **Full CNN-BiLSTM detector** (thesis arm A4, `src/cnn_bilstm_detector.py`).

The per-move detector (A3g) is re-checked on the same kit for the regression suite.

## Steps

1. `a0g_refit.py` (HPC). The thesis A0g run kept its scores but not the fitted
   model, so this re-runs `extra_arm_a0.fit_lgbm` unchanged on the cached pooled
   features and the same grouped split and saves the booster. The re-fit
   reproduces all 268,000 stored A0g scores exactly (max abs diff 0.0,
   best iteration 460), so `src/models/detector_lgbm_a0g.txt` is the thesis
   model, not a look-alike.
2. `build_kit.py` (HPC): the 14 bundled PGNs (12 current samples plus the 2
   archived ones), each side in turn as the suspect with `maia_band` = that
   side's header Elo, plus 12 real `anomaly_corpus_v2` games (the detector
   parity check's 8, 2 from A4's v2 test partition, 2 from A0g's grouped
   heldout_both partition).
3. `extract_detector_features_v2.py --stage rating_cheap` (HPC, CPU) on that kit.
4. `hpc_score.py` (HPC, CPU): A0g through `extra_arms_common.pooled_features`
   and the re-fit booster (`lightgbm.Booster.predict`); A4 through
   `extra_arm_a4.stage_cache` (frozen CNN trunk, float16 cache), `load_cache`,
   `make_model`, `Batcher` and `predict` with `A4/model_best.pt`; A3g as in
   `experiments/detector_parity/hpc_score.py`. Also records the original thesis
   GPU runs' scores for the corpus games. Output: `hpc_scores.json`.
5. `app_parity.py` (local): the same PGNs through `api._run_inference`, reading
   each method from `suspicion_methods`. Output: `parity_rows.json` and the
   tables below. Fails if any side-score differs by 1e-4 or more.
6. `make_fixtures.py` (local): the model-free test fixture for
   `tests/test_lgbm_detector.py` and `tests/test_cnn_bilstm_detector.py`.

## Result

| method | max abs diff, HPC code path vs app | side-scores | max abs diff, HPC CPU vs thesis GPU run (corpus games) |
| --- | --- | --- | --- |
| LightGBM (A0g) | 4.8e-07 | 40 | 1.4e-03 |
| CNN-BiLSTM (A4) | 2.6e-05 | 40 | 1.3e-02 |
| Per-move detector (A3g) | 5.6e-07 | 40 | 7.7e-05 |

Both new methods match the HPC code path well inside 1e-4.

**HPC CPU vs the thesis GPU runs.** That column compares the thesis code with
itself on different hardware, not the port:

- LightGBM: 10 of 12 corpus games match the thesis run exactly; two differ
  (1.4e-03 and 5.7e-04) because the rating features were re-extracted on CPU
  for the kit, and a CPU-vs-GPU difference in a pooled feature can cross a tree
  split threshold.
- CNN-BiLSTM: most games are within about 5e-04, but one
  (`1300_stockfish16_r05__game_01202`) differs by 1.3e-02. `a4_gpu_check.py`
  (HPC) shows the cause is the BiLSTM/attention head running on GPU vs CPU, not
  the embeddings or the port: feeding the thesis run's own GPU-cached
  embeddings through the unchanged thesis model gives 0.5797 on CPU and
  0.5922 on GPU (the thesis run's 0.5923). The app runs on CPU, so its A4
  score for an individual game can differ from what the thesis GPU evaluation
  would have given by about this much.

| corpus game | plies | max abs embedding diff (thesis GPU cache vs CPU) | A4 on CPU, thesis embeddings | A4 on GPU, thesis embeddings |
| --- | --- | --- | --- | --- |
| 1200_lc0_r60/game_00606 | 36 | 6.1e-04 | 0.514692 | 0.514834 |
| 1300_lc0_r10/game_01307 | 54 | 4.6e-04 | 0.210388 | 0.211988 |
| 1300_lc0_r20/game_01178 | 69 | 8.2e-04 | 0.875515 | 0.875898 |
| 1300_stockfish16_r02/game_00002 | 60 | 6.0e-04 | 0.209162 | 0.209122 |
| 1300_stockfish16_r05/game_01202 | 43 | 6.1e-04 | 0.579671 | 0.592239 |
| 1300_stockfish16_r10/game_00002 | 100 | 4.9e-04 | 0.430261 | 0.430521 |
| 1500_hardneg/game_01480 | 100 | 6.1e-04 | 0.406235 | 0.406527 |
| 1500_lc0_r00/game_01825 | 74 | 6.7e-04 | 0.291308 | 0.291588 |
| 1500_lc0_r40/game_00675 | 93 | 5.5e-04 | 0.477841 | 0.477710 |
| 1600_lc0_r60/game_01598 | 48 | 6.1e-04 | 0.706285 | 0.706349 |
| 1600_stockfish16_r00/game_00291 | 77 | 6.1e-04 | 0.291117 | 0.291064 |
| 1700_lc0_r40/game_01002 | 95 | 5.8e-04 | 0.347015 | 0.347280 |

## LightGBM (A0g)

| kind | game | suspect side | HPC code path | app code path | abs diff | thesis run (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| sample | blitz_best_predicted | white | 0.549863 | 0.549863 | 2.0e-08 |  |
| sample | blitz_best_predicted | black | 0.440815 | 0.440815 | 2.1e-08 |  |
| sample | blitz_typical | white | 0.696088 | 0.696088 | 3.3e-07 |  |
| sample | blitz_typical | black | 0.394835 | 0.394835 | 2.4e-07 |  |
| sample | blitz_worst_predicted | white | 0.416299 | 0.416299 | 2.2e-08 |  |
| sample | blitz_worst_predicted | black | 0.694185 | 0.694185 | 4.5e-08 |  |
| sample | bullet_best_predicted | white | 0.705131 | 0.705131 | 4.2e-07 |  |
| sample | bullet_best_predicted | black | 0.292711 | 0.292711 | 2.2e-07 |  |
| sample | bullet_typical | white | 0.570588 | 0.570588 | 5.9e-08 |  |
| sample | bullet_typical | black | 0.574148 | 0.574148 | 4.7e-07 |  |
| sample | bullet_worst_predicted | white | 0.779809 | 0.779809 | 4.3e-08 |  |
| sample | bullet_worst_predicted | black | 0.397389 | 0.397389 | 3.2e-07 |  |
| sample | rapid_best_predicted | white | 0.432723 | 0.432723 | 2.5e-07 |  |
| sample | rapid_best_predicted | black | 0.628106 | 0.628106 | 4.0e-07 |  |
| sample | rapid_typical | white | 0.365749 | 0.365749 | 4.1e-07 |  |
| sample | rapid_typical | black | 0.594772 | 0.594772 | 2.6e-07 |  |
| sample | rapid_worst_predicted | white | 0.453497 | 0.453497 | 4.8e-07 |  |
| sample | rapid_worst_predicted | black | 0.536784 | 0.536784 | 4.0e-07 |  |
| sample | synthetic_caught | white | 0.743125 | 0.743125 | 6.3e-08 |  |
| sample | synthetic_caught | black | 0.393197 | 0.393197 | 1.7e-07 |  |
| sample | synthetic_false_alarm | white | 0.354297 | 0.354297 | 9.8e-08 |  |
| sample | synthetic_false_alarm | black | 0.357426 | 0.357426 | 2.0e-07 |  |
| sample | synthetic_false_alarm_v2 | white | 0.368943 | 0.368943 | 1.1e-07 |  |
| sample | synthetic_false_alarm_v2 | black | 0.539478 | 0.539478 | 4.3e-07 |  |
| sample | synthetic_missed | white | 0.306656 | 0.306656 | 3.5e-08 |  |
| sample | synthetic_missed | black | 0.692025 | 0.692025 | 5.4e-08 |  |
| sample | synthetic_missed_v2 | white | 0.415146 | 0.415146 | 4.0e-07 |  |
| sample | synthetic_missed_v2 | black | 0.388459 | 0.388459 | 3.9e-07 |  |
| corpus | 1600_stockfish16_r00__game_00291 | black | 0.336865 | 0.336865 | 9.8e-08 | 0.336865 |
| corpus | 1500_lc0_r00__game_01825 | white | 0.554575 | 0.554575 | 1.3e-07 | 0.554575 |
| corpus | 1200_lc0_r60__game_00606 | black | 0.527648 | 0.527648 | 8.2e-08 | 0.527648 |
| corpus | 1300_stockfish16_r02__game_00002 | white | 0.279312 | 0.279312 | 3.6e-08 | 0.279312 |
| corpus | 1300_stockfish16_r05__game_01202 | white | 0.615935 | 0.615935 | 1.3e-07 | 0.615935 |
| corpus | 1700_lc0_r40__game_01002 | black | 0.428553 | 0.428553 | 4.3e-07 | 0.428553 |
| corpus | 1500_hardneg__game_01480 | white | 0.419901 | 0.419901 | 2.3e-08 | 0.419901 |
| corpus | 1300_lc0_r20__game_01178 | white | 0.709870 | 0.709870 | 1.7e-07 | 0.711241 |
| corpus | 1600_lc0_r60__game_01598 | black | 0.732514 | 0.732514 | 6.9e-08 | 0.732514 |
| corpus | 1500_lc0_r40__game_00675 | white | 0.521460 | 0.521460 | 4.8e-07 | 0.521460 |
| corpus | 1300_stockfish16_r10__game_00002 | white | 0.454542 | 0.454542 | 3.9e-07 | 0.454542 |
| corpus | 1300_lc0_r10__game_01307 | white | 0.328501 | 0.328501 | 2.2e-07 | 0.329075 |

## CNN-BiLSTM (A4)

| kind | game | suspect side | HPC code path | app code path | abs diff | thesis run (GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| sample | blitz_best_predicted | white | 0.571291 | 0.571292 | 5.5e-07 |  |
| sample | blitz_best_predicted | black | 0.560818 | 0.560818 | 4.6e-07 |  |
| sample | blitz_typical | white | 0.681780 | 0.681780 | 7.9e-08 |  |
| sample | blitz_typical | black | 0.112246 | 0.112245 | 5.1e-07 |  |
| sample | blitz_worst_predicted | white | 0.591876 | 0.591902 | 2.6e-05 |  |
| sample | blitz_worst_predicted | black | 0.352294 | 0.352293 | 1.1e-06 |  |
| sample | bullet_best_predicted | white | 0.403975 | 0.403975 | 2.0e-07 |  |
| sample | bullet_best_predicted | black | 0.344398 | 0.344399 | 1.0e-06 |  |
| sample | bullet_typical | white | 0.510408 | 0.510408 | 3.1e-07 |  |
| sample | bullet_typical | black | 0.580819 | 0.580819 | 4.9e-08 |  |
| sample | bullet_worst_predicted | white | 0.886955 | 0.886958 | 2.5e-06 |  |
| sample | bullet_worst_predicted | black | 0.887543 | 0.887548 | 4.9e-06 |  |
| sample | rapid_best_predicted | white | 0.485494 | 0.485493 | 5.4e-07 |  |
| sample | rapid_best_predicted | black | 0.703417 | 0.703417 | 4.7e-07 |  |
| sample | rapid_typical | white | 0.671031 | 0.671031 | 3.0e-07 |  |
| sample | rapid_typical | black | 0.763769 | 0.763769 | 2.7e-07 |  |
| sample | rapid_worst_predicted | white | 0.542446 | 0.542443 | 3.3e-06 |  |
| sample | rapid_worst_predicted | black | 0.387257 | 0.387258 | 1.3e-06 |  |
| sample | synthetic_caught | white | 0.812938 | 0.812939 | 7.9e-07 |  |
| sample | synthetic_caught | black | 0.549849 | 0.549847 | 2.3e-06 |  |
| sample | synthetic_false_alarm | white | 0.536906 | 0.536906 | 4.0e-09 |  |
| sample | synthetic_false_alarm | black | 0.557426 | 0.557425 | 1.3e-06 |  |
| sample | synthetic_false_alarm_v2 | white | 0.251938 | 0.251938 | 4.9e-07 |  |
| sample | synthetic_false_alarm_v2 | black | 0.344666 | 0.344667 | 8.5e-07 |  |
| sample | synthetic_missed | white | 0.278722 | 0.278725 | 3.5e-06 |  |
| sample | synthetic_missed | black | 0.747507 | 0.747505 | 1.8e-06 |  |
| sample | synthetic_missed_v2 | white | 0.494937 | 0.494937 | 2.7e-09 |  |
| sample | synthetic_missed_v2 | black | 0.395801 | 0.395801 | 2.8e-07 |  |
| corpus | 1600_stockfish16_r00__game_00291 | black | 0.291347 | 0.291348 | 7.6e-07 | 0.291066 |
| corpus | 1500_lc0_r00__game_01825 | white | 0.291329 | 0.291328 | 1.4e-06 | 0.291587 |
| corpus | 1200_lc0_r60__game_00606 | black | 0.515075 | 0.515075 | 3.9e-07 | 0.514839 |
| corpus | 1300_stockfish16_r02__game_00002 | white | 0.209234 | 0.209234 | 4.9e-07 | 0.209110 |
| corpus | 1300_stockfish16_r05__game_01202 | white | 0.579365 | 0.579369 | 4.2e-06 | 0.592283 |
| corpus | 1700_lc0_r40__game_01002 | black | 0.346876 | 0.346875 | 1.2e-06 | 0.347283 |
| corpus | 1500_hardneg__game_01480 | white | 0.406199 | 0.406200 | 1.4e-06 | 0.406527 |
| corpus | 1300_lc0_r20__game_01178 | white | 0.875519 | 0.875519 | 1.4e-07 | 0.875897 |
| corpus | 1600_lc0_r60__game_01598 | black | 0.706668 | 0.706668 | 1.4e-07 | 0.706349 |
| corpus | 1500_lc0_r40__game_00675 | white | 0.477969 | 0.477968 | 9.3e-07 | 0.477789 |
| corpus | 1300_stockfish16_r10__game_00002 | white | 0.430434 | 0.430434 | 5.0e-07 | 0.430523 |
| corpus | 1300_lc0_r10__game_01307 | white | 0.210309 | 0.210309 | 1.2e-07 | 0.211990 |

## Per-move detector (A3g), re-checked on this kit

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
| sample | rapid_worst_predicted | black | 0.799591 | 0.799591 | 1.1e-07 |  |
| sample | synthetic_caught | white | 0.948874 | 0.948874 | 2.4e-07 |  |
| sample | synthetic_caught | black | 0.276653 | 0.276653 | 4.6e-07 |  |
| sample | synthetic_false_alarm | white | 0.355413 | 0.355413 | 1.9e-07 |  |
| sample | synthetic_false_alarm | black | 0.297366 | 0.297366 | 3.0e-07 |  |
| sample | synthetic_false_alarm_v2 | white | 0.235945 | 0.235945 | 4.2e-07 |  |
| sample | synthetic_false_alarm_v2 | black | 0.891577 | 0.891577 | 5.4e-09 |  |
| sample | synthetic_missed | white | 0.172342 | 0.172342 | 2.7e-08 |  |
| sample | synthetic_missed | black | 0.876502 | 0.876502 | 9.7e-08 |  |
| sample | synthetic_missed_v2 | white | 0.385702 | 0.385702 | 2.8e-07 |  |
| sample | synthetic_missed_v2 | black | 0.249967 | 0.249967 | 2.0e-07 |  |
| corpus | 1600_stockfish16_r00__game_00291 | black | 0.276599 | 0.276599 | 2.2e-07 | 0.276675 |
| corpus | 1500_lc0_r00__game_01825 | white | 0.575283 | 0.575283 | 2.3e-07 | 0.575317 |
| corpus | 1200_lc0_r60__game_00606 | black | 0.421589 | 0.421589 | 3.1e-07 | 0.421610 |
| corpus | 1300_stockfish16_r02__game_00002 | white | 0.121473 | 0.121473 | 3.7e-07 | 0.121504 |
| corpus | 1300_stockfish16_r05__game_01202 | white | 0.632383 | 0.632383 | 2.5e-07 | 0.632333 |
| corpus | 1700_lc0_r40__game_01002 | black | 0.367820 | 0.367820 | 3.5e-08 | 0.367813 |
| corpus | 1500_hardneg__game_01480 | white | 0.359076 | 0.359076 | 2.3e-08 | 0.359074 |
| corpus | 1300_lc0_r20__game_01178 | white | 0.850387 | 0.850387 | 1.4e-07 | 0.850391 |
| corpus | 1600_lc0_r60__game_01598 | black | 0.922655 | 0.922655 | 1.9e-07 | 0.922663 |
| corpus | 1500_lc0_r40__game_00675 | white | 0.948520 | 0.948520 | 5.5e-08 | 0.948509 |
| corpus | 1300_stockfish16_r10__game_00002 | white | 0.277798 | 0.277798 | 2.4e-07 | 0.277842 |
| corpus | 1300_lc0_r10__game_01307 | white | 0.222610 | 0.222610 | 2.8e-07 | 0.222602 |

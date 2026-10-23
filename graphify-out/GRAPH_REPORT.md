# Graph Report - chess-skill-estimation-attention  (2026-10-23)

## Corpus Check
- Large corpus: 500 files · ~680,437 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2870 nodes · 6087 edges · 175 communities (139 shown, 32 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 477 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- vendored chart.umd.js
- vendored chart.umd.js
- vendored chess.js
- main.rs
- app.js
- synthesize_anomaly_clocks.py
- vectors()
- vendored chart.umd.js
- vendored chart.umd.js
- mixed_cheater_pooling.py
- vendored chart.umd.js
- extract_detector_features_v2.py
- extra_arms_common.py
- vendored chart.umd.js
- vendored chart.umd.js
- vendored chart.umd.js
- vendored chart.umd.js
- vendored chart.umd.js
- test_format_data.py
- vendored chart.umd.js
- 170k-verification.md
- test_side_separation.py
- generate_anomaly_corpus.py
- A0g and A4 parity
- vendored chart.umd.js
- anomaly-extra-arms-plan.md
- refreshPlayBoardLocalUi()
- vendored chart.umd.js
- test_live.py
- Stage-2 Hyperparameter Tuning Pre-Registration
- anomaly-clock-synthesis-plan.md
- api.py
- LichessError
- RatingNet Prototype index.html
- _run_inference()
- rerenderForActualRatingToggle()
- README.md (chess-skill-estimation-attention)
- limits.py
- renderBoardAtPly()
- chess_rating_net.py
- closed-account-data-readiness-plan.md
- parse_game()
- Tuned attention arm attn_tuned
- GameBlobStore
- resolve_baseline()
- vendored chart.umd.js
- test_rating_model_selector.py
- ChessEloPredictor
- AnomalyDetector
- corpus_stream_parallel.sh
- vendored chart.umd.js
- test_pipeline_recovery.sh
- detector.py
- cnn_bilstm_detector.py
- lgbm_detector.py
- resolve_cutoffs()
- clkscan Rust corpus scanner
- Live mode (follow a
- _live_stream_game_events()
- build_corpus_store.py
- live.py
- submitAnalysis()
- main()
- train_anomaly_detector.py
- ARM64 VPS deployment (Ubuntu
- A3g detector cutoffs run
- BahdanauAttention
- compute_critical_moves()
- format_data_legacy.py
- anomaly-corpus-generated.md
- reap_stale_claims()
- Schema Completeness Check: Stored
- move_explanation_engine.py
- handleLiveMessage()
- vendored chart.umd.js
- vendored chart.umd.js
- model()
- move_explanation_case_studies.py
- main()
- extra_arm_a4.py
- Cluster training runners (tmux
- LiveGamePrefix
- vendored chart.umd.js
- extra_arm_a2.py
- corpus_stream.sh
- build_opening_tables.py
- vendored chart.umd.js
- SQLite blob store corpus.sqlite
- plot_results_figures.py
- preprocess_lichess.py
- score_cutoffs.py
- .forward_causal()
- aggregate_extra_arms.py
- Shared predict/* response shape
- parse_ndjson_line()
- setSuspicionMethod()
- vendored chart.umd.js
- test_detector.py
- corpus-parallel-pipeline-review.md
- Reservoir RNG stream invariance
- e5_dense_sup.py
- hpc_sweep_queue.sh
- _pgn_to_tensor_inputs()
- check_corpus_store.py
- MILGRU
- Held-Out Test Eval Results
- make_edge_case_pgns.py
- SelfAttention
- vendored chart.umd.js
- test_cnn_bilstm_detector.py
- test_lgbm_detector.py
- Handler
- paired_bootstrap_heldout.py
- move_explanation_attention_diagnostics.py
- move_explanation_regression.py
- preprocess_fast.py
- scan_equivalence_check.py
- score_detector_cutoffs.py
- seconds_to_clock_str()
- playOnScreen()
- compare_processed_dirs.py
- lr=3e-4 attention configuration (5-seed
- DeepTrunk
- method_cutoffs/cutoffs_from_scores.py
- score_method_cutoffs.py
- vendored chart.umd.js
- test_suspicion_methods.py
- corpus_composition.py
- fm-corpus-heal.sh
- detector_cutoffs/cutoffs_from_scores.py
- [dense] pilot training run
- satt_cutoffs/cutoffs_from_scores.py
- renderPlayerBars()
- Full query-key additive (Bahdanau)
- .corpus_done month completion marker
- --scanner rust opt-in flag
- extra_arm_a1_on_a3g.py
- LiveStreamError
- run_anomaly_corpus_v2.sh
- plot_mixed_cheater_pooling.py
- rebucket_by_rating.py
- fm-corpus-autopreprocess.sh
- fm-corpus-zst-cleanup.sh
- hpc_training_heal.sh
- load_model_and_sample_games_to_txt()
- Prefetch per-worker read budget
- naive_floor.py
- run_heldout_test_eval.sh
- e2_cnn_shapes.py
- method_parity/build_kit.py
- run_baseline_repro.sh
- v2 corpus twin-game leakage
- run_move_explanation.sh
- stage.sh
- start.sh
- e1_grad_audit.py
- e3_train_signal.py
- e4_selfattn_mask.py
- e4b_selfattn_correction.py
- e9_anomaly.py
- method_parity/app_parity.py
- build_regression_expected.py
- method_parity/hpc_score.py
- make_fixtures.py
- headline_auc.py
- prep.py
- format.sh
- hpc_launch_run.sh
- src/__init__.py
- _mock_client()
- clkscan

## God Nodes (most connected - your core abstractions)
1. `an()` - 61 edges
2. `ns()` - 55 edges
3. `s()` - 42 edges
4. `o()` - 39 edges
5. `a()` - 38 edges
6. `n()` - 37 edges
7. `Chess()` - 37 edges
8. `ChessEloPredictor` - 35 edges
9. `no` - 35 edges
10. `LiveGamePrefix` - 33 edges

## Surprising Connections (you probably didn't know these)
- `A0 v2 LightGBM per-game detector` --references--> `AnomalyDetector`  [INFERRED]
  analysis/mixed-cheater-pooling-results.md → src/anomaly.py
- `Prefix-estimate semantics (frozen move-by-move live curve)` --references--> `LiveGamePrefix`  [INFERRED]
  docs/web-prototype.md → src/live.py
- `Peak raw staging ~153-168 GB at CONCURRENCY=5` --rationale_for--> `free_bytes()`  [INFERRED]
  analysis/corpus-parallel-pipeline-review.md → corpus-ops/corpus_stream_parallel.sh
- `No preprocess retry; each month fed to xargs once` --rationale_for--> `attempts_inc()`  [INFERRED]
  analysis/corpus-parallel-pipeline-review.md → corpus-ops/corpus_stream_parallel.sh
- `Dropped safety properties vs serial corpus_stream.sh (curl -f, resume, validations, cd guard)` --rationale_for--> `validate_months()`  [INFERRED]
  analysis/corpus-parallel-pipeline-review.md → corpus-ops/corpus_stream_parallel.sh

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Synthetic clock design, review, fix and recalibration cycle** — chess_skill_estimation_attention_analysis_anomaly_clock_synthesis_plan, chess_skill_estimation_attention_analysis_anomaly_clock_review, chess_skill_estimation_attention_analysis_anomaly_clock_fixes, chess_skill_estimation_attention_analysis_anomaly_clock_recalibrate, chess_skill_estimation_attention_analysis_synthesize_anomaly_clocks [EXTRACTED 1.00]
- **Claim-liveness mechanism and its reaper failure modes** — chess_skill_estimation_attention_analysis_corpus_pipeline_hardening_claim_heartbeat_liveness, chess_skill_estimation_attention_analysis_corpus_pipeline_hardening_stale_heartbeat_reaper, chess_skill_estimation_attention_analysis_corpus_resilience_review_g1_ssh_failure_read_as_no_heartbeat, chess_skill_estimation_attention_analysis_corpus_resilience_review_g2_zero_grace_ownerless_claim, chess_skill_estimation_attention_analysis_corpus_resilience_review_g5_no_ownership_recheck, chess_skill_estimation_attention_analysis_corpus_resilience_review_g6_orphan_heartbeat_subshell, chess_skill_estimation_attention_analysis_corpus_resilience_review_g10_pid_reuse [EXTRACTED 1.00]
- **Pre-registered anomaly detector arms A0-A4** — chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a0_lightgbm_anchor, chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a0g_grouped_split, chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a1_player_level_aggregation, chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a2_band_normalized_features, chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a3_bigru_sequence_detector, chess_skill_estimation_attention_analysis_anomaly_extra_arms_plan_a4_cnn_bilstm_detector [EXTRACTED 1.00]
- **Reproducible rating-corpus sampling chain** — chess_skill_estimation_attention_readme_clkscan_rust_scanner, chess_skill_estimation_attention_readme_cpython_random_port, chess_skill_estimation_attention_readme_algorithm_r_reservoir_sampling, chess_skill_estimation_attention_github_workflows_scanner_tests_cpython_equivalence_suite, chess_skill_estimation_attention_readme_preprocessing_equivalence [EXTRACTED 1.00]
- **Four selectable suspicion-score methods sharing one rating-model pass and per-method cutoffs** — chess_skill_estimation_attention_docs_web_prototype_s_att_computed_score, chess_skill_estimation_attention_docs_web_prototype_lightgbm_detector_a0g, chess_skill_estimation_attention_docs_web_prototype_per_move_detector_a3g, chess_skill_estimation_attention_docs_web_prototype_cnn_bilstm_detector_a4, chess_skill_estimation_attention_docs_api_suspicion_methods_field, chess_skill_estimation_attention_experiments_method_cutoffs_readme_suspicion_cutoffs_json [EXTRACTED 1.00]
- **Suspicion label cutoff calibration runs on held-out test games** — chess_skill_estimation_attention_experiments_satt_cutoffs_readme_s_att_cutoffs_run, chess_skill_estimation_attention_experiments_detector_cutoffs_readme_detector_cutoffs_run, chess_skill_estimation_attention_experiments_method_cutoffs_readme_method_cutoffs_run, chess_skill_estimation_attention_experiments_satt_cutoffs_readme_percentile_cutoff_method, chess_skill_estimation_attention_experiments_detector_cutoffs_readme_stratified_heldout_test_sample, chess_skill_estimation_attention_experiments_method_cutoffs_readme_suspicion_cutoffs_json [EXTRACTED 1.00]
- **White-vs-Black separation retrain: pre-registration to verdict** — chess_skill_estimation_attention_analysis_mirror_retrain_preregistration_white_black_mirroring, chess_skill_estimation_attention_analysis_mirror_retrain_preregistration_arm_a_gapweight, chess_skill_estimation_attention_analysis_mirror_retrain_preregistration_arm_b_diffhead, chess_skill_estimation_attention_analysis_mirror_retrain_preregistration_pass_mark, chess_skill_estimation_attention_analysis_heldout_test_eval_mirror_side_check_side_separation_metrics, chess_skill_estimation_attention_analysis_mirror_retrain_results_arm_a_pass, chess_skill_estimation_attention_analysis_mirror_retrain_results_arm_b_fail [EXTRACTED 1.00]
- **Verification that the app's detector ports reproduce the thesis HPC code** — chess_skill_estimation_attention_experiments_detector_parity_readme_a3g_parity_check, chess_skill_estimation_attention_experiments_method_parity_readme_lgbm_cnn_bilstm_parity_check, chess_skill_estimation_attention_experiments_method_parity_readme_a0g_refit, chess_skill_estimation_attention_experiments_method_parity_readme_a4_cpu_vs_gpu_discrepancy, chess_skill_estimation_attention_experiments_detector_parity_readme_extract_detector_features_v2, chess_skill_estimation_attention_experiments_method_cutoffs_progress_scoring_progress_log [INFERRED 0.85]
- **Sampling-identical corpus preprocessing variants (Python fast path, clkscan two-pass replay, one-pass Rust)** — chess_skill_estimation_attention_analysis_corpus_pipeline_bottleneck_verdict_scan_eligibility_predicate, chess_skill_estimation_attention_analysis_corpus_pipeline_fix_plan_has_full_clocks_fast_path, chess_skill_estimation_attention_analysis_faster_scan_research_clkscan, chess_skill_estimation_attention_analysis_faster_scan_research_algorithm_r_replay, chess_skill_estimation_attention_analysis_rust_pass2_preprocessor_onepass_preprocessor, chess_skill_estimation_attention_analysis_rust_pass2_preprocessor_inline_algorithm_r, chess_skill_estimation_attention_analysis_rust_pass2_preprocessor_cpython_mt19937_port [INFERRED 0.85]

## Communities (175 total, 32 thin omitted)

### Community 0 - "vendored chart.umd.js"
Cohesion: 0.04
Nodes (36): at(), b(), beforeDatasetDraw(), beforeDatasetsDraw(), cn(), d(), da(), destroy() (+28 more)

### Community 1 - "vendored chart.umd.js"
Cohesion: 0.06
Nodes (21): Ae(), afterDraw(), afterUpdate(), ba, Bi(), Ci(), configure(), Ee() (+13 more)

### Community 2 - "vendored chess.js"
Cohesion: 0.07
Nodes (62): algebraic(), ATTACKS, BISHOP, BITS, BLACK, Chess(), attacked(), build_move() (+54 more)

### Community 3 - "main.rs"
Cohesion: 0.07
Nodes (38): BufWriter, Args, ClkScan, create(), GameBuf, has_clk(), is_digits(), is_ws() (+30 more)

### Community 4 - "app.js"
Cohesion: 0.06
Nodes (57): applyPlayMoveAfterChessMove(), buildMoreToggleLine(), buildPlayHeaders(), buildPlayPgn(), buildSampleCard(), buildSyncPlugin(), chartLegendPlacement(), chartPalette() (+49 more)

### Community 5 - "synthesize_anomaly_clocks.py"
Cohesion: 0.09
Nodes (48): band_skill(), bin_index(), bucket_of(), categorize_time_control(), ClockParams, compare_with_real(), describe(), draw_time_control() (+40 more)

### Community 6 - "vectors()"
Cohesion: 0.07
Nodes (34): Emit CPython random.Random reference vectors in a dependency-free flat format.…, Byte-exact CPython MT19937 port (init_by_array, high-bit getrandbits, rejection randint), PyRandom, Self, algorithm_r_reproduces_cpython_reservoir(), genrand_uint32_stream_matches_cpython(), getrandbits_matches_cpython_across_widths_and_seeds(), randint_below_matches_cpython() (+26 more)

### Community 7 - "vendored chart.umd.js"
Cohesion: 0.07
Nodes (7): Cs, fe(), ks(), nn(), os(), sn, tn

### Community 8 - "vendored chart.umd.js"
Cohesion: 0.07
Nodes (25): a(), aa(), addElements(), afterDatasetsUpdate(), beforeLayout(), determineDataLimits(), f(), ko (+17 more)

### Community 9 - "mixed_cheater_pooling.py"
Cohesion: 0.06
Nodes (40): draw(), hardneg_check(), pools(), A1 player-level aggregation (no training). Answers a DIFFERENT question from…, condition -> {"cheat": [(cell_key, rate, idx)], "clean": [(band, idx)],…, (n_players, kmax) game indices; each row without replacement over pool…, -> {score_name: {k: {agg: (labels, values, rates)}}}, simulate() (+32 more)

### Community 10 - "vendored chart.umd.js"
Cohesion: 0.07
Nodes (12): buildLookupTable(), En, Fo(), _generate(), getDecimalForValue(), _getTimestampsForTable(), init(), initOffsets() (+4 more)

### Community 11 - "extract_detector_features_v2.py"
Cohesion: 0.08
Nodes (36): Stockfish depth-20 engine features (engine.npz, substituted + matched clean plies), Per-cell shard resumability (--resume and merge), rating_cheap per-ply features (frozen rating-model forward), _as_bool_arr(), _atomic_savez(), _cc(), _cheap_features(), _clamp_cp() (+28 more)

### Community 12 - "extra_arms_common.py"
Cohesion: 0.07
Nodes (34): atomic_savez(), auc(), auc_ci(), boot_indices(), _clean(), conditions(), Data, engine_proximity() (+26 more)

### Community 13 - "vendored chart.umd.js"
Cohesion: 0.08
Nodes (4): As(), ns(), rt(), updateRangeFromParsed()

### Community 14 - "vendored chart.umd.js"
Cohesion: 0.08
Nodes (24): Be(), buildTicks(), draw(), Fn(), g(), getLabelAndValue(), getLabelForValue(), getMaxOverflow() (+16 more)

### Community 15 - "vendored chart.umd.js"
Cohesion: 0.08
Nodes (32): ai(), ao(), average(), beforeDraw(), dataset(), getCenterPoint(), getRange(), hi() (+24 more)

### Community 16 - "vendored chart.umd.js"
Cohesion: 0.07
Nodes (11): Bt(), color(), Ft(), Gt(), It(), jt(), te(), vt() (+3 more)

### Community 17 - "vendored chart.umd.js"
Cohesion: 0.09
Nodes (16): beforeUpdate(), bn(), dn(), e(), ei(), je(), on(), pn() (+8 more)

### Community 18 - "test_format_data.py"
Cohesion: 0.08
Nodes (27): rejected_games write-only reject counter, categorize_time_control(), game_setup_error(), parse_time_control(), Minimal data-formatting utilities adapted from RatingNet. Only the pieces…, Map an estimated game duration in seconds to a Lichess time-control bucket., Return why this PGN can't be analyzed due to its setup, or None if it's a…, Parse a PGN ``TimeControl`` header (``"{base_seconds}+{increment_seconds}"``,… (+19 more)

### Community 19 - "vendored chart.umd.js"
Cohesion: 0.08
Nodes (19): bo, _calculateBarIndexPixels(), _calculateBarValuePixels(), et(), getPixelForTick(), getPixelForValue(), _getRuler(), _getStackCount() (+11 more)

### Community 20 - "170k-verification.md"
Cohesion: 0.10
Nodes (33): 2.45 Elo attention win is a best-val selection effect, Inherited BiLSTM non-causality (retrospective per-move curve), 'Deeper CNN' claimed but not implemented (M2), Glicko-2 label noise floor (L4), Head-of-month sequential sampling (~15-minute window per month), Hierarchical paired bootstrap (seed pairs x test games), Headline MAE as mean of per-batch means (L3), Anomaly head is non-parametric and unreachable from training (+25 more)

### Community 21 - "test_side_separation.py"
Cohesion: 0.09
Nodes (34): gap_weights(), rating_gap_abs_error(), Per-game loss weight that grows with the two players' rating gap. ``w = min(1 +…, Mean absolute per-side error of each game, in rating points. (B,2) -> (B,)., Absolute error of the predicted White-minus-Black difference. (B,2) -> (B,).…, Training loss for the White/Black separation arms, in rating points. Returns…, separation_training_loss(), side_abs_error() (+26 more)

### Community 22 - "generate_anomaly_corpus.py"
Cohesion: 0.12
Nodes (21): _game_metadata(), _import_clock_module(), MaiaPolicyEngine, main(), open_lc0_cheater_engine(), open_stockfish_engine(), OpeningTable, play_one_game() (+13 more)

### Community 23 - "A0g and A4 parity"
Cohesion: 0.10
Nodes (24): lightgbm installed in addition to requirements.txt (for A0g), suspicion_methods response field (s_att, lgbm_a0g, detector_a3g_seed0, cnn_bilstm_a4), Committed detector weights with JSON provenance files, CNN-BiLSTM detector (A4, frozen rating CNN + fine-tuned BiLSTM and gated attention head), LightGBM detector (A0g, 97 pooled per-game features), Per-move detector (A3g seed 0, 2-layer BiGRU over 17 per-ply features, default), suspicion_methods.json (method captions generated by build_method_table.py), Suspicion score (four selectable methods) (+16 more)

### Community 25 - "anomaly-extra-arms-plan.md"
Cohesion: 0.10
Nodes (24): Strong-titled-player brilliancy control set, A3g player-level AUC (k=20 withheld_band mean 0.862), Hard-negative players drift toward cheater players as k grows, Label-blind clock synthesis (--cheater-timing opt-in), Hard-negative strength-mismatch cells, Maia policy sampling (P**(1/tau), tau 0.9, plies 1-16), Per-game engine strength (SF depth 14-20, Lc0 nodes 200-1600), Real-opening seeding from per-band opening tables (+16 more)

### Community 26 - "refreshPlayBoardLocalUi()"
Cohesion: 0.12
Nodes (29): barPositionForSide(), clearPlaySuspicionAndChart(), ensurePlayChess(), fetchPlayLichessBaseline(), liveClockShouldTick(), playClockModeLabel(), playClockShouldTick(), playOverReasonText() (+21 more)

### Community 27 - "vendored chart.umd.js"
Cohesion: 0.11
Nodes (10): ca(), Do(), eo(), getBasePixel(), Gn(), ha, la(), Oe() (+2 more)

### Community 28 - "test_live.py"
Cohesion: 0.12
Nodes (26): _fen_after(), parametrize, Tests for src/live.py: NDJSON parsing, prefix-based PGN bookkeeping, and the…, A real Lichess-style TV feed FEN for the position after these moves., run(), test_iter_ndjson_lines_yields_parsed_objects(), test_live_stream_endpoint_sends_custom_position_error_as_sse_event(), test_prefix_add_move_builds_valid_pgn() (+18 more)

### Community 29 - "Stage-2 Hyperparameter Tuning Pre-Registration"
Cohesion: 0.11
Nodes (27): WebDataset-style shards rejected, Fix Review: Eight Fixes, Methodology Strictness and Anomaly-Corpus Design, Frozen verified split manifest (sha256 over sorted basenames), N1: historical checkpoints invalid under the new sorted split, per_game_errors.csv dump (absolute errors only at review time), --seed / --split_seed separation, Seed-Rerun Test-MAE Results (10 completed 170k checkpoints), Seed-averaged bootstrap conflates test-sampling and training noise (+19 more)

### Community 30 - "anomaly-clock-synthesis-plan.md"
Cohesion: 0.15
Nodes (22): Normalization constants (ratings 1514/366, clocks 273/380), Constant-method control arm fix (no increment accrual), Clock-synthesis self-test regression guard groups, Sigman citation corrections (91,340-game sample, emphasis, .bib journal/author), Maia-band-dependent think-time pacing and dispersion, Phase-knot pacing recalibration to Sigman 40th-move budget, Constant control arm accrues increment (ramp bug), Failed 40th-move budget calibration (~88% vs 74.7-77.9%) (+14 more)

### Community 31 - "api.py"
Cohesion: 0.14
Nodes (25): API checkpoint discovery mismatch (model_55.pth vs preflight_check_2m), _discover_checkpoint(), _discover_gap_weighted_checkpoint(), _find_checkpoint(), lifespan(), _load_cnn_bilstm_detector(), _load_detector(), _load_gap_weighted_model() (+17 more)

### Community 32 - "LichessError"
Cohesion: 0.14
Nodes (23): LichessError, parse_game_id(), Exception, Lichess game-by-ID client. Fetches a game's PGN (with clock annotations) from…, Raised for any Lichess game-ID or fetch failure, with an HTTP status to map to., Extract an 8-character Lichess game ID from a bare ID or a full game URL., _client_returning(), AsyncClient (+15 more)

### Community 33 - "RatingNet Prototype index.html"
Cohesion: 0.11
Nodes (25): Web prototype, Lichess-like one-screen analysis board layout, Chart.js 4.4.4, chess.js 0.13.4, chessboard.js 1.0.0, Hide ratings / Reveal toggle (guess-the-rating demo), jQuery 3.7.1, METRIC_INFO glossary and (i) popovers (+17 more)

### Community 34 - "_run_inference()"
Cohesion: 0.14
Nodes (24): _cache_put(), _labels_for_scores(), LichessGameRequest, _log_prediction(), PGNTextRequest, predict_lichess_game(), predict_pgn_text(), predict_pgn_upload() (+16 more)

### Community 35 - "rerenderForActualRatingToggle()"
Cohesion: 0.14
Nodes (24): actualAndErrorHtml(), actualRatingsVisible(), baselineText(), buildReferenceLineDatasets(), computedGapTitle(), escapeHtml(), formatClock(), formatDeltaHtml() (+16 more)

### Community 36 - "README.md (chess-skill-estimation-attention)"
Cohesion: 0.13
Nodes (17): Anomaly ROC-AUC protocol (game-side unit, stratified, val-calibrated thresholds), S_att attention-weighted rating-deviation suspicion score, S_max and S_mean aggregations, Maia domain-gap hypothesis (H2: model reads Maia-1100 as ~1,900), Earlier v2 evaluation failure (game-level ROC-AUC 0.3438), Decision-point eval: S_att ROC-AUC ~0.555 (runs a-d), Score the prototype's bundled sample PGNs through its own /predict/pgn path., README.md (chess-skill-estimation-attention) (+9 more)

### Community 37 - "limits.py"
Cohesion: 0.10
Nodes (16): Bounded single-worker deployment (20 live streams, 503 + Retry-After queue), bound_analysis_queue(), revalidate_frontend(), AnalysisBusy, AnalysisQueue, LiveStreamLimit, LiveStreamLimiter, Exception (+8 more)

### Community 38 - "renderBoardAtPly()"
Cohesion: 0.13
Nodes (23): buildAttentionRanks(), buildCriticalIndex(), buildFenTimeline(), buildMoveCell(), init(), jumpFromChartEvent(), markActiveMoveListEntry(), preservingWindowScroll() (+15 more)

### Community 39 - "chess_rating_net.py"
Cohesion: 0.11
Nodes (20): ArgumentParser, L1 loss on de-normalised ratings (val loss = MAE in rating points), 2024 IEEE Big Data Cup Chess Puzzle Difficulty Competition, Lichess benchmark dataset (>1M games with clock times), model_55.pth (Omori baseline checkpoint, fallback of last resort), Omori & Tadepalli (Oregon State): rating estimation from moves and clock times, CNN + BiLSTM, MAE 182, Optional training path (chess_rating_net.py --train), Example training config (cnn_bilstm_clocks_all, use_attention false, bahdanau) (+12 more)

### Community 40 - "closed-account-data-readiness-plan.md"
Cohesion: 0.12
Nodes (18): Anomaly credibility traps (circular labels, survivorship, unsigned deviation), Rating-bracket evaluation over 2N player-observations, BOT accounts over-represented among largest rating errors, main(), read_month(), Bot-account check, stage 2 (local): query Lichess POST /api/users for every…, Accounts, not games, as the unit of analysis, Closed-account data ethics (closed != cheated, no consent, no evasion guide) (+10 more)

### Community 41 - "parse_game()"
Cohesion: 0.11
Nodes (22): PGN parser clock/position misalignment (M6), Pipeline hard-reject gates ([%clk] every ply, Rated, Standard, TimeControl), Corpus-Pipeline Bottleneck Verdict, local-preprocess-speed-test report (absent from repository), Scan-phase eligibility predicate (full mainline clock coverage), Scan-phase 12-plane tensor build waste (~81% of scan time), Corpus-Pipeline Fix Plan (skip tensor build during reservoir scan), CLOCK_COMMENT_RE clock-annotation regex (+14 more)

### Community 42 - "Tuned attention arm attn_tuned"
Cohesion: 0.16
Nodes (21): Mirror Side-Check Output (mirror_arm_a, mirror_arm_b, attn_tuned), Side-separation metrics (pred_gap_median, signed_gap_pearson_r, mae_gap_ge_300), Deeper CNN arm5 (val 172.37, not promoted), Tuned attention arm attn_tuned (frozen architecture, test MAE 171.92), Frozen 2.55M split manifest (sha256 dee7d117, 255,000 test games), Pre-Registration: White versus Black Separation Retrain, Arm A: mismatch gap weighting (mirror_arm_a_gapweight), Arm B: difference-aware separate heads (mirror_arm_b_diffhead) (+13 more)

### Community 43 - "GameBlobStore"
Cohesion: 0.12
Nodes (13): ChessGamesDataset, collate_fn(), GameBlobStore, load_or_create_split(), Any, Connection, Dataset, Path (+5 more)

### Community 44 - "resolve_baseline()"
Cohesion: 0.19
Nodes (19): BaselineResolution, parse_header_rating(), Baseline-rating resolution for suspicion scoring. The anomaly module needs a…, Parse a PGN Elo header value, e.g. '1500'. Returns None if missing or non-…, Resolve a side's true rating from the PGN header alone. Unlike…, Resolve a side's baseline rating. Resolution order: 1. ``requested`` (explicit…, resolve_actual_rating(), resolve_baseline() (+11 more)

### Community 46 - "test_rating_model_selector.py"
Cohesion: 0.11
Nodes (8): fixture, Tests for the "Rating model" selector (api.py's MODEL vs GAP_WEIGHTED_MODEL).…, Patch api's loaded-model globals with two distinct tiny models, and disable the…, test_load_gap_weighted_model_mismatched_ratings_mean_raises(), test_load_gap_weighted_model_strict_load_succeeds(), _tiny_model(), two_models(), _write_tiny_checkpoint()

### Community 47 - "ChessEloPredictor"
Cohesion: 0.11
Nodes (18): Demo served random-init attention weights (M1), Attention query-averaging positional artefact (H.1b), Attention front-loading (909-fold ply-1 to ply-90 gap), Pre-fixed case studies: most accurate game and largest swing, ChessEloPredictor, mae_per_item(), Tensor, CNN-BiLSTM rating estimator with optional attention and anomaly branch. The… (+10 more)

### Community 48 - "AnomalyDetector"
Cohesion: 0.12
Nodes (16): Self-referential R_baseline fallback (model's own final-ply prediction), Timing as an adversarial surface (masking limitation), Banned account's inflated rating as R_baseline confound, Deployment Measurements: Per-Move Latency and Offline-vs-Live Agreement, Bidirectional per-ply curve divergence (mean ~174 Elo mid-game), Offline-vs-live final-ply agreement (bit-exact on 100 games), Per-move end-to-end latency (GPU A5000 3.42 ms mean, CPU Xeon 7.73 ms mean), Prefix-recompute live inference (+8 more)

### Community 49 - "corpus_stream_parallel.sh"
Cohesion: 0.27
Nodes (19): attempts_get(), attempts_inc(), ctl(), download_month(), expected_size(), file_size(), free_bytes(), is_failed() (+11 more)

### Community 50 - "vendored chart.umd.js"
Cohesion: 0.14
Nodes (7): ce(), de, dt(), he(), ia(), pa(), qs()

### Community 51 - "test_pipeline_recovery.sh"
Cohesion: 0.17
Nodes (17): MAX_ATTEMPTS retries with terminal FAILED state, G3: an interruption burns a retry (three power cuts make a healthy month FAILED), Recovery-test harness extensions (fake ssh, second host root, pid-targeted kill), MAX_STARTS / attempts separation (G3 fix), bad(), CORPUS_STREAM_TEST_CONFIG, counter(), FAKE_MARKER (+9 more)

### Community 52 - "detector.py"
Cohesion: 0.18
Nodes (17): build_features(), _cheap_features(), Detector, per_ply_arrays(), ndarray, Trained per-move sequence detector (thesis arm A3g, seed 0): the app's main…, is_capture, is_check, material_balance (suspect POV, before the move). Port of…, Port of ``extract_detector_features_v2._traj_features``. (+9 more)

### Community 53 - "cnn_bilstm_detector.py"
Cohesion: 0.18
Nodes (17): A4Model, cnn_trunk(), CnnBilstmDetector, embed_positions(), load_cnn_bilstm_detector(), device, Module, Path (+9 more)

### Community 54 - "lgbm_detector.py"
Cohesion: 0.16
Nodes (15): LgbmDetector, load_lgbm_detector(), load_model_text(), pool_game(), ndarray, Path, LightGBM game-level detector (thesis arm A0g): one of the selectable suspicion…, Parse a LightGBM text model (``Booster.save_model``) with numerical splits. (+7 more)

### Community 55 - "resolve_cutoffs()"
Cohesion: 0.20
Nodes (17): label_for_score(), Any, Plain-language labels for a suspicion score, resolved against a cutoffs file.…, Pick the p75/p95 pair to use for a game's time control. Uses the time-control-…, Classify a score as typical, unusual, or highly unusual. Boundaries: below p75…, resolve_cutoffs(), ResolvedCutoffs, test_label_above_p95_is_highly_unusual() (+9 more)

### Community 56 - "clkscan Rust corpus scanner"
Cohesion: 0.13
Nodes (16): Faster Corpus-Scan Research (measured, not projected), clkscan Rust corpus scanner, niklasf/pgn-reader Rust crate, Raising CONCURRENCY rejected as the alternative, 161x scan speedup (968 vs 156,580 games/s on identical input), Verdict equivalence on 626,765 real games, main(), Path (+8 more)

### Community 57 - "Live mode (follow a"
Cohesion: 0.16
Nodes (18): GitHub repo J4ve/chess-skill-estimation-attention, arm_a_gapweight_best.pth optional checkpoint (RATINGNET_CHECKPOINT_GAP_WEIGHTED), Deployment limits (live stream and analysis concurrency caps, HTTP 503 Retry-After), RATINGNET_CHECKPOINT environment variable, ratingnet.service systemd unit, GET /health endpoint, GET /live/stream/{game_id} SSE endpoint, GET /live/tv SSE endpoint (+10 more)

### Community 58 - "_live_stream_game_events()"
Cohesion: 0.14
Nodes (18): api_index(), _client_key(), health(), live_stream_game(), _live_stream_game_events(), live_tv(), _metered_live_stream(), Follow Lichess TV's currently featured game over Server-Sent Events. (+10 more)

### Community 59 - "build_corpus_store.py"
Cohesion: 0.24
Nodes (17): cmd_build(), cmd_stat(), cmd_verify(), _compress(), connect_rw(), iter_archive(), main(), Connection (+9 more)

### Community 60 - "live.py"
Cohesion: 0.20
Nodes (17): build_update_event(), clock_seconds_for_ply(), iter_ndjson_lines(), AsyncClient, Live game streaming and prefix-based inference bookkeeping. The model's BiLSTM…, Stream an NDJSON endpoint, yielding one parsed object per non-blank line., Run inference over the current prefix and format it as an "update" SSE event., Follow one Lichess game by ID, yielding SSE-formatted text chunks. ``prefix``… (+9 more)

### Community 61 - "submitAnalysis()"
Cohesion: 0.28
Nodes (18): bumpAnalysisRequestSeq(), clearError(), currentBaselines(), currentMinPly(), currentTopK(), loadAndAnalyzeSample(), loadFullGameAnalysisForLiveGame(), onRatingModelChange() (+10 more)

### Community 62 - "main()"
Cohesion: 0.18
Nodes (14): main(), _percentile(), Chapter 4 deployment measurements: per-move latency and offline-vs-live…, summarize(), build_model(), _equivalence_check_one_game(), load_checkpoint_strict_on_trained_parts(), load_game_tensors() (+6 more)

### Community 63 - "train_anomaly_detector.py"
Cohesion: 0.19
Nodes (15): auc_safe(), breakdowns(), computed_scores(), load_npz_games(), main(), make_split(), per_ply_model(), pool_game() (+7 more)

### Community 64 - "ARM64 VPS deployment (Ubuntu"
Cohesion: 0.12
Nodes (15): RatingNet Prototype Hugging Face Space README, Hugging Face Space deployment (Docker SDK, app_port 7860), Thesis: Real-Time Chess Skill Estimation and Anomaly Detection Using an Attention-Augmented CNN-BiLSTM, Deploying the prototype to a small ARM64 server, ARM64 VPS deployment (Ubuntu 22.04 aarch64, live URL), Caddy reverse proxy HTTPS via sslip.io hostname, CPU-only PyTorch wheel index install, Prediction log rotation (logs/predictions.jsonl, size-based logrotate) (+7 more)

### Community 65 - "A3g detector cutoffs run"
Cohesion: 0.26
Nodes (17): Frozen thesis checkpoint attn_tuned (preflight_check_2m/best_model.pth, test MAE 171.92), Computed score S_att (attention-weighted deviation from baseline, rating points), Sample games tab (manifest.json rating and synthetic samples), Suspicion labels (Typical / Unusual / Highly unusual percentile cutoffs), Detector score cutoffs (2026-09-17), A3g detector cutoffs run (2,822 games, 5,644 sides), Held-out TEST evaluation of attn_tuned (analysis/heldout_test_eval/attn_tuned__best.csv), Stratified held-out TEST sample (seed 20260915, up to 700 per fast time control) (+9 more)

### Community 66 - "BahdanauAttention"
Cohesion: 0.17
Nodes (9): E7: three ways to actually wire attention into the rating head. E1 proved the…, chapter3.tex:90 as literally written: y_hat = FC2(Drop(f(FC1(s)))). Attention-…, Keep per-move head; concat the GLOBAL context onto every timestep. Preserves…, Per-move head with a CAUSAL context: at ply t, attend only over plies 1..t.…, W1_GlobalPool, W2_ConcatBroadcast, W3_CausalCumulative, BahdanauAttention (+1 more)

### Community 67 - "compute_critical_moves()"
Cohesion: 0.24
Nodes (15): Critical moves ranking (attention x rating deviation), compute_critical_moves(), Any, Critical-move ranking for human fair-play review. Ranks moves by how much…, Return (critical_moves, warnings) for both sides. Each side is ranked…, _record(), test_anomaly_unavailable_returns_empty_with_warning(), test_invalid_min_ply_raises() (+7 more)

### Community 68 - "format_data_legacy.py"
Cohesion: 0.16
Nodes (16): board_to_array(), categorize_time_control(), game_to_pgn_string(), main(), parse_game(), parse_game_for_inference(), process_pgn_file(), process_pgn_stream() (+8 more)

### Community 69 - "anomaly-corpus-generated.md"
Cohesion: 0.17
Nodes (11): Missing Time key in anomaly corpus (KeyError at load), Synthetic remaining-time clock countdown (per-player Lichess arithmetic), lc0 segfault / hang from GPU contention (repair pass), Clocks hold engine think-time, not a remaining-time countdown, Synthetic anomaly corpus v1 (90,000 games, 9 bands x 2 engines x 5 rates), Bot-vs-bot engine-substitution generation (Maia suspect, engine substitutes), Maia ladder weights (bands 1100-1900, nodes=1), Reference engines: Stockfish 16 (50 ms) and small Lc0 net (nodes=100) (+3 more)

### Community 70 - "reap_stale_claims()"
Cohesion: 0.15
Nodes (16): mkdir claim lock never released (no trap, no metadata), Per-pid staging directory with atomic mv publish, CTL_INFRA_RC=255 ssh transport-failure rule, G10: reused PID hides a stale claim from the heartbeat fallback, G12: abandoned <month>.staging.<pid> directories never pruned, G1: reaper reads an ssh failure as a missing heartbeat and deletes a live claim, G2: ownerless-claim branch applies zero grace, G5: worker never re-checks claim ownership (+8 more)

### Community 71 - "Schema Completeness Check: Stored"
Cohesion: 0.15
Nodes (16): R5 adviser questions, Bot-vs-bot anomaly-validation corpus protocol, Synthetic clock realism and naive-timing stratum, Deviation classification D1-D14 (minor vs major), 9-model Maia ladder (1100-1900), Substitution placement patterns P1 uniform / P2 critical-moment / P3 block-onset, Schema Completeness Check: Stored Fields vs Chapter 4 Plans, Finding 1: stale player-identifier schema claims (+8 more)

### Community 72 - "move_explanation_engine.py"
Cohesion: 0.17
Nodes (14): analyse_game(), game_phase(), load_moves_from_store(), main(), _probe_stockfish_id(), Stockfish pass over every ply of a sample of TEST games: one engine.analyse per…, Reads Moves (UCI) for one game directly from the GameBlobStore sqlite, without…, UCI "id name"/"id author" via a raw pipe, not python-chess (see main()). (+6 more)

### Community 73 - "handleLiveMessage()"
Cohesion: 0.27
Nodes (16): clearLiveReconnectTimer(), closeLiveEventSourceQuietly(), connectLiveEventSource(), extractLichessGameId(), handleLiveMessage(), liveSessionOnScreen(), mergeLiveUpdate(), resetLiveSessionUi() (+8 more)

### Community 74 - "vendored chart.umd.js"
Cohesion: 0.17
Nodes (9): ct(), fs(), ge(), gs(), ps(), vs(), we(), ws (+1 more)

### Community 75 - "vendored chart.umd.js"
Cohesion: 0.18
Nodes (4): generateLabels(), labelColor(), labelPointStyle(), onClick()

### Community 76 - "model()"
Cohesion: 0.17
Nodes (10): Batcher, build_ply_matrix(), _full_band(), predict(), A3 sequence detector on the frozen rating model's per-move outputs. Plan:…, Gathers padded (B, 100, F) batches from the flat ply matrix on GPU., train_one(), predict() (+2 more)

### Community 77 - "move_explanation_case_studies.py"
Cohesion: 0.26
Nodes (13): build_case_json(), load_data(), main(), make_figure(), Select the two illustrative TEST games and build their figures + JSON.…, select_games(), categorize_time_control(), compute_time_spent() (+5 more)

### Community 78 - "main()"
Cohesion: 0.27
Nodes (15): load_checkpoint(), main(), device, Module, Train for one epoch. When ``dense_supervision`` is True (Fix 4), the loss is…, Evaluate on the test set. When ``per_game_csv`` is given, also dumps one row…, Save optimizer/scheduler state, epoch counters, and hyperparameters alongside…, Resume from a checkpoint written by ``save_checkpoint``. (+7 more)

### Community 79 - "extra_arm_a4.py"
Cohesion: 0.24
Nodes (11): Batcher, cache_dir(), cnn_trunk(), load_cache(), load_frozen(), main(), make_model(), A4 end-to-end CNN-BiLSTM detector initialized from the frozen rating model.… (+3 more)

### Community 80 - "Cluster training runners (tmux"
Cohesion: 0.18
Nodes (12): abl_attention_170k run (Bahdanau attention, best-val epoch 18, test MAE 221.7), is_running(), log(), ORDER, hpc_seed_queue.sh script, Cluster training runners, Cluster training runners (tmux sessions, one run per GPU, queues, watcher), hpc_training_heal.sh cron watcher (+4 more)

### Community 81 - "LiveGamePrefix"
Cohesion: 0.16
Nodes (9): LiveGamePrefix, Tracks a growing move/clock prefix for one live game as PGN text. Rendering the…, The most recent recorded clock for one side, or None before its first move., Piece-placement field of the current position's FEN., Shortest legal move sequence (UCI, at most ``max_plies`` long) from the current…, _fake_run_inference(), test_build_update_event_adds_truncation_warning(), test_build_update_event_returns_none_for_empty_prefix() (+1 more)

### Community 83 - "extra_arm_a2.py"
Cohesion: 0.19
Nodes (8): A2 band-normalized feature z-scores, lgbm_arm(), main(), A0 anchor (and A0g grouped-split sensitivity): the v2 LightGBM detector,…, Shared by A0/A0g and A2: pooled features -> LightGBM -> reports. transform(X,…, main(), make_transform(), A2 band-normalized features: anchor LightGBM on rating-derived features…

### Community 84 - "corpus_stream.sh"
Cohesion: 0.31
Nodes (12): Lichess ~99M games/month volume anchor, HPC always-on backbone, local opportunistic placement, Bounded month-parallelism (3-4 HPC workers), Single script with --host hpc|local, download_month(), EXPECTED, file_size(), log() (+4 more)

### Community 85 - "build_opening_tables.py"
Cohesion: 0.27
Nodes (11): band_keys(), _finalize(), iter_archive_games(), iter_store_games(), _loads_fast(), main(), _NoTorchUnpickler, Path (+3 more)

### Community 87 - "SQLite blob store corpus.sqlite"
Cohesion: 0.20
Nodes (12): LMDB, HDF5 and packed-binary alternative formats, SQLite blob store corpus.sqlite (one zlib-compressed pickle per row), ~82 per-month .tar.gz corpus archives (~107 MB each), Trainer refuses resume on mismatched experiment flags, tmux server argv watchdog blind spot, hpc_training_heal.sh cron auto-resume watchdog, PROTECTED_MONTHS 2024-02..07 (pre-username 170k subset), Corpus Pipeline Drivers (corpus-ops README) (+4 more)

### Community 88 - "plot_results_figures.py"
Cohesion: 0.24
Nodes (10): _conditions(), figure_anomaly(), figure_rating(), load_anomaly(), load_rating(), Render the two Chapter 4 results figures from the frozen result JSONs. Figure 1…, Rate conditions live under game_level for A3g/A4, at the top for A0g/S_att., style() (+2 more)

### Community 89 - "preprocess_lichess.py"
Cohesion: 0.24
Nodes (11): Three-rung --scanner rust fallback ladder, has_full_clocks(), is_eligible(), main(), month_label(), Game, Stream-preprocess a Lichess lichess_db_standard_rated_YYYY-MM.pgn.zst dump into…, Extract YYYY-MM from a Lichess dump filename, else 'unknown'. (+3 more)

### Community 90 - "score_cutoffs.py"
Cohesion: 0.29
Nodes (11): load_model(), main(), pkl_path_for(), progress(), Path, Random, Compute suspicion-score (S_att) percentile cutoffs from held-out TEST-partition…, Group held-out test game_ids by time control, restricted to unpacked months,… (+3 more)

### Community 91 - ".forward_causal()"
Cohesion: 0.22
Nodes (8): Attention (B,T,T,d) intermediate sets the batch size, _build_causal_mask(), device, Tensor, Causal-cumulative attention: at ply *t*, attend only over plies 1..t. Full…, Causal-cumulative self-attention: query *t* attends to keys 1..t only. Args:…, Build a (batch, seq, seq) boolean causal + padding mask. Entry ``[b, t, s]`` is…, Global (non-causal) attention — used for diagnostics / return_attention path.…

### Community 92 - "aggregate_extra_arms.py"
Cohesion: 0.25
Nodes (9): fmt(), fmt_diff(), game_rows(), main(), Build the side-by-side results table for the extra anomaly arms from whatever…, -> list of (row label, game_level_report dict), status(), main() (+1 more)

### Community 93 - "Shared predict/* response shape"
Cohesion: 0.24
Nodes (11): API reference, Baseline resolution (request / pgn_header / self_prediction_fallback), per_move records (rating, attention_weight, deviation, clock_seconds, time_spent_seconds), POST /predict/lichess endpoint, POST /predict/pgn endpoint, POST /predict/upload endpoint, Shared predict/* response shape, Critical moves (attention_weight * |deviation|, min_ply) (+3 more)

### Community 94 - "parse_ndjson_line()"
Cohesion: 0.18
Nodes (11): classify_game_stream_message(), headers_from_tv_players(), parse_ndjson_line(), Any, Build minimal PGN-style headers from a TV "featured" event's player list., Parse one NDJSON line, returning None for a blank keep-alive line., Classify a message from GET /api/stream/game/{id}. Returns "final" (game-over…, test_classify_meta_position_final() (+3 more)

### Community 95 - "setSuspicionMethod()"
Cohesion: 0.25
Nodes (11): applyPlayTooShortSuspicionOverride(), applySuspicionMethodInfo(), formatAuc(), loadSuspicionMethodPreference(), loadSuspicionMethodsTable(), methodCaption(), methodInfoText(), renderSuspicionMethodCaption() (+3 more)

### Community 96 - "vendored chart.umd.js"
Cohesion: 0.25
Nodes (4): addBox(), start(), u(), vn()

### Community 97 - "test_detector.py"
Cohesion: 0.20
Nodes (3): det(), parity_case(), fixture

### Community 98 - "corpus-parallel-pipeline-review.md"
Cohesion: 0.24
Nodes (9): Dropped safety properties vs serial corpus_stream.sh (curl -f, resume, validations, cd guard), No preprocess retry; each month fed to xargs once, Peak raw staging ~153-168 GB at CONCURRENCY=5, Corpus Pipeline Hardening: Status Command and Crash Recovery, Stale-heartbeat reaper (STALE_MIN=45), --status one-screen pipeline status, Resilience Review of corpus_stream_parallel.sh, Ungraceful shutdown modes: power off, sleep, lost network (+1 more)

### Community 99 - "Reservoir RNG stream invariance"
Cohesion: 0.22
Nodes (10): Fast-path divergence liveness risk (kept < 30,000 triggers retries then FATAL), Reservoir RNG stream invariance under the eligibility fast path, Torch pickle storage-address byte nondeterminism, Sampling-identical A/B validation protocol, expected_size() function replacing a bash-4 associative array, Algorithm R replay from the eligibility bitstream, Content (not byte) equivalence criterion, Decision B: do not add Site/UTCDate/UTCTime mid-run (+2 more)

### Community 100 - "e5_dense_sup.py"
Cohesion: 0.20
Nodes (5): DS, evaluate(), Dataset, E5: last-ply-only supervision vs dense (all-ply) supervision. The baseline…, Always last-ply MAE in rating points -- the baseline's own metric.

### Community 101 - "hpc_sweep_queue.sh"
Cohesion: 0.29
Nodes (7): hpc_sweep_launch_run.sh script, is_running(), log(), ORDER, record_result(), hpc_sweep_queue.sh script, SWEPT

### Community 102 - "_pgn_to_tensor_inputs()"
Cohesion: 0.31
Nodes (9): _pgn_to_tensor_inputs(), Parse a PGN string and produce model-ready tensors plus move lists., Error-path tests for PGN input handling that do not require the model…, test_chess960_pgn_raises_custom_position_error(), test_empty_pgn_raises_value_error(), test_from_position_pgn_raises_custom_position_error_not_clock_error(), test_pgn_with_clocks_parses_successfully(), test_pgn_without_clocks_raises_value_error() (+1 more)

### Community 103 - "check_corpus_store.py"
Cohesion: 0.33
Nodes (8): check(), games_equal(), _load_store_class(), main(), Structural check is not an identity check: the pickled games carry no self-…, Import GameBlobStore from the trainer without importing torch. The trainer…, _worker(), Full-Corpus Training Data Storage Recommendation

### Community 104 - "MILGRU"
Cohesion: 0.22
Nodes (7): make_model(), load_detector(), MILGRU, device, Path, Tensor, BiGRU + per-ply instance logits + gated-attention MIL pooling. Attribute names…

### Community 105 - "Held-Out Test Eval Results"
Cohesion: 0.28
Nodes (8): Held-Out Test Eval Results (Chapter 4 headline numbers), Omori released checkpoint model_55.pth (zero-shot test MAE 193.47), Omori and Tadepalli (2024) 182 test MAE reference, _import_ratingnet(), main(), Score a trained RatingNet checkpoint on the frozen 255k held-out TEST split.…, Rating normalization constants (mean 1514, std 366), Normalization re-fit diagnostic (mean 1665.7, std 395.8)

### Community 106 - "make_edge_case_pgns.py"
Cohesion: 0.31
Nodes (7): build(), legal_sans(), main(), movetext(), n plies of legal SAN from the start position, deterministically. Uses a knight…, Emit movetext by hand so the exact comment bytes are under our control., (name, pgn_text, expected_eligible)

### Community 107 - "SelfAttention"
Cohesion: 0.22
Nodes (4): Self-contained attention module for the RatingNet extension. Implements…, Simple scaled dot-product self-attention over LSTM outputs. Computes attention…, Global (non-causal) self-attention — used for diagnostics. Returns: output:…, SelfAttention

### Community 109 - "test_cnn_bilstm_detector.py"
Cohesion: 0.25
Nodes (3): case(), det(), fixture

### Community 110 - "test_lgbm_detector.py"
Cohesion: 0.25
Nodes (4): parity_case(), _prob(), fixture, test_tree_walk_follows_lightgbm_missing_value_rules()

### Community 112 - "paired_bootstrap_heldout.py"
Cohesion: 0.36
Nodes (7): Attention effect on held-out test (-3.08 MAE, CI [-3.41, -2.76]), Reproduced baseline arm1 (test MAE 175.00), bootstrap_ci(), load(), main(), parse_pair(), Paired bootstrap over per-game absolute errors on the frozen 255k TEST split.…

### Community 113 - "move_explanation_attention_diagnostics.py"
Cohesion: 0.39
Nodes (7): depth_and_terminal_stats(), load_curves(), main(), observed_mean_attention_by_ply(), Regenerates the two numbers the attention-front-loading caveat in move-…, Every position's depth, cross-checked against whether it is a genuinely…, uniform_baseline_by_ply()

### Community 114 - "move_explanation_regression.py"
Cohesion: 0.43
Nodes (7): build_design(), cluster_bootstrap_ols(), fit_from_weights(), main(), nuisance_design(), per_game_sufficient_stats(), Cluster-bootstrap OLS: what does the live estimate's ply-to-ply change actually…

### Community 115 - "preprocess_fast.py"
Cohesion: 0.32
Nodes (7): main(), Path, Pass 1. Returns (scanned, eligible)., Replay Algorithm R over the eligibility bitstream. Mirrors…, replay_reservoir(), run_clkscan(), Peak RSS drop from 5.86 GiB to 203 MiB

### Community 116 - "scan_equivalence_check.py"
Cohesion: 0.36
Nodes (6): _install_predicates(), main(), Path, Stands in for torch during import; every attribute access is a bug., Import the real is_eligible / has_full_clocks / stream_games., _StubTorch

### Community 117 - "score_detector_cutoffs.py"
Cohesion: 0.46
Nodes (7): load_model(), main(), progress(), Score a stratified sample of held-out TEST-partition games with the RatingNet…, score_game(), stratified_sample(), time_to_seconds()

### Community 118 - "seconds_to_clock_str()"
Cohesion: 0.25
Nodes (6): Append several live moves; ``clock_for_ply`` gives each new ply's clock.…, Append one live move. Returns False (no-op) once the ply cap is hit., Format integer seconds as Lichess's 'H:MM:SS' PGN clock-comment string., seconds_to_clock_str(), test_seconds_to_clock_str_clamps_negative(), test_seconds_to_clock_str_formats_hms()

### Community 119 - "playOnScreen()"
Cohesion: 0.46
Nodes (8): canDragPlayPiece(), clearSquareHighlights(), flipBoard(), highlightCurrentPly(), highlightMoveSquares(), onPlayBoardSnapEnd(), playOnScreen(), setupBoard()

### Community 120 - "compare_processed_dirs.py"
Cohesion: 0.48
Nodes (6): compare_one(), game_files(), load(), main(), Path, Return a list of human-readable differences, empty if identical.

### Community 121 - "lr=3e-4 attention configuration (5-seed"
Cohesion: 0.33
Nodes (7): arm6 baseline + lr 3e-4 confound control (test MAE 173.55), Benitez Q2: learning-rate confound question, Learning Rate 3e-4: 5-Seed Confirmation Results, lr=3e-4 attention configuration (5-seed best-val 213.24 ± 0.79), Paired test gain of 6.50 rating points (t=-4.28, p≈0.013), Mean ± SD mislabeled as confidence intervals, 5-seed test MAE mean error (215.38 stated vs 215.54)

### Community 122 - "DeepTrunk"
Cohesion: 0.29
Nodes (4): baseline_trunk(), DeepTrunk, E6: cost model for the candidate CNN-side changes. Measures parameter count and…, Deeper CNN that preserves board geometry: pool only twice, not 3x. Baseline…

### Community 123 - "method_cutoffs/cutoffs_from_scores.py"
Cohesion: 0.52
Nodes (6): bootstrap_ci(), main(), md5(), percentile(), Turn per_side_scores.csv (all four methods per side) into the combined…, summarize()

### Community 124 - "score_method_cutoffs.py"
Cohesion: 0.48
Nodes (6): load_model(), main(), progress(), Score the detector cutoffs' held-out TEST sample under all four suspicion-score…, score_game(), time_to_seconds()

### Community 126 - "test_suspicion_methods.py"
Cohesion: 0.38
Nodes (3): _cutoffs(), test_combined_cutoffs_file_drops_entries_naming_another_score(), test_suspicion_methods_label_each_score_against_its_own_cutoffs()

### Community 127 - "corpus_composition.py"
Cohesion: 0.53
Nodes (5): categorize_time_control(), classify_blob(), main(), Per-time-control composition and 100-ply-truncation rate for the full corpus.…, _worker()

### Community 128 - "fm-corpus-heal.sh"
Cohesion: 0.67
Nodes (5): block(), classify_pass(), log(), fm-corpus-heal.sh script, stalled()

### Community 129 - "detector_cutoffs/cutoffs_from_scores.py"
Cohesion: 0.60
Nodes (5): bootstrap_ci(), main(), percentile(), Turn per_side_scores.csv (detector score per side) into p75/p95 cutoffs with…, summarize()

### Community 130 - "[dense] pilot training run"
Cohesion: 0.40
Nodes (3): [dense] pilot training run (train 4320 / val 1080 / test 600, best val MAE 275.95), [last] pilot training run (train 4320 / val 1080 / test 600, best val MAE 288.73), Data prep pass (6,000 games kept of 6,062 scanned)

### Community 131 - "satt_cutoffs/cutoffs_from_scores.py"
Cohesion: 0.60
Nodes (5): bootstrap_ci(), main(), percentile(), Turn per_side_scores.csv (S_att score per side) into p75/p95 cutoffs with…, summarize()

### Community 132 - "renderPlayerBars()"
Cohesion: 0.40
Nodes (6): clockSecondsForSide(), gameIsOver(), parseTimeControlHeader(), renderPlayerBars(), setLiveClockAnchor(), sideToMoveAtPly()

### Community 133 - "Full query-key additive (Bahdanau)"
Cohesion: 0.60
Nodes (4): Manuscript attention equations vs code mechanism (M3), Bahdanau et al. (2015) additive attention, Full query-key additive (Bahdanau) attention, Key-only content-based attention variant (superseded)

### Community 134 - ".corpus_done month completion marker"
Cohesion: 0.50
Nodes (5): .corpus_done month completion marker, Atomic .incoming rsync transfer for local months, Claim owner + heartbeat liveness mechanism, G11: completion tested on the local filesystem only, G4: results rsync publishes .corpus_done ahead of the games

### Community 135 - "--scanner rust opt-in flag"
Cohesion: 0.40
Nodes (5): --scanner rust opt-in flag, Conditions for making the Rust scanner the default, zstd stream_reader silent truncation bug, HPC validation and production flip to --scanner rust (2026-08-18), Select-mode truncated-archive refusal

### Community 136 - "extra_arm_a1_on_a3g.py"
Cohesion: 0.40
Nodes (3): pool_size_notes(), A1 player-level aggregation, pointed at A3g's per-game scores. Reuses…, Flag every (condition, pool) whose available game count is below the largest…

### Community 137 - "LiveStreamError"
Cohesion: 0.40
Nodes (3): LiveStreamError, Exception, test_iter_ndjson_lines_maps_404()

### Community 138 - "run_anomaly_corpus_v2.sh"
Cohesion: 0.50
Nodes (3): CUDA_VISIBLE_DEVICES, rate_tag(), run_anomaly_corpus_v2.sh script

### Community 139 - "plot_mixed_cheater_pooling.py"
Cohesion: 0.67
Nodes (3): draw_panel(), main(), Renders the mixed-cheater-pooling chart (AUC vs. games pooled, one line per…

### Community 140 - "rebucket_by_rating.py"
Cohesion: 0.67
Nodes (3): bracket_for(), main(), Re-bucket an existing per-game held-out-test CSV by rating bracket. Does NOT…

### Community 141 - "fm-corpus-autopreprocess.sh"
Cohesion: 1.00
Nodes (3): log(), pass(), fm-corpus-autopreprocess.sh script

### Community 142 - "fm-corpus-zst-cleanup.sh"
Cohesion: 1.00
Nodes (3): log(), pass(), fm-corpus-zst-cleanup.sh script

### Community 143 - "hpc_training_heal.sh"
Cohesion: 1.00
Nodes (3): ensure_store(), log(), hpc_training_heal.sh script

### Community 144 - "load_model_and_sample_games_to_txt()"
Cohesion: 0.83
Nodes (3): calculate_mae(), load_model_and_sample_games_to_txt(), save_game_to_txt()

### Community 145 - "Prefetch per-worker read budget"
Cohesion: 0.67
Nodes (3): Compressed store in page cache vs uncompressed on disk, Prefetch per-worker read budget analysis (85 s vs 228 s epoch), ZIP fresh-handle-per-read overhead artifact hypothesis

## Ambiguous Edges - Review These
- `Frozen 2.55M split manifest (sha256 dee7d117, 255,000 test games)` → `~2.19M-game published corpus (73 months)`  [AMBIGUOUS]
  corpus-ops/README.md · relation: conceptually_related_to
- `lightgbm installed in addition to requirements.txt (for A0g)` → `LightGBM binary GBDT booster (460 trees, 31 leaves, 97 features, lr 0.02)`  [AMBIGUOUS]
  deploy/vps/README.md · relation: conceptually_related_to

## Knowledge Gaps
- **121 isolated node(s):** `extra_arms_launch.sh script`, `extra_arms_run.sh script`, `run_move_explanation.sh script`, `run_v2_engine.sh script`, `run_v2_rating_cheap.sh script` (+116 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 694 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Frozen 2.55M split manifest (sha256 dee7d117, 255,000 test games)` and `~2.19M-game published corpus (73 months)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `lightgbm installed in addition to requirements.txt (for A0g)` and `LightGBM binary GBDT booster (460 trees, 31 leaves, 97 features, lr 0.02)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `README.md (chess-skill-estimation-attention)` connect `README.md (chess-skill-estimation-attention)` to `corpus-parallel-pipeline-review.md`, `compute_critical_moves()`, `limits.py`, `vectors()`, `closed-account-data-readiness-plan.md`, `AnomalyDetector`, `170k-verification.md`, `clkscan Rust corpus scanner`, `anomaly-clock-synthesis-plan.md`?**
  _High betweenness centrality (0.115) - this node is a cross-community bridge._
- **Why does `RatingNet Prototype index.html` connect `RatingNet Prototype index.html` to `Live mode (follow a`, `app.js`, `Shared predict/* response shape`, `A0g and A4 parity`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `ChessEloPredictor` connect `ChessEloPredictor` to `BahdanauAttention`, `score_cutoffs.py`, `chess_rating_net.py`, `extract_detector_features_v2.py`, `SelfAttention`, `main()`, `test_rating_model_selector.py`, `AnomalyDetector`, `load_model_and_sample_games_to_txt()`, `170k-verification.md`, `score_detector_cutoffs.py`, `test_side_separation.py`, `DeepTrunk`, `score_method_cutoffs.py`, `main()`, `api.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `s()` (e.g. with `beforeUpdate()` and `da()`) actually correct?**
  _`s()` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `o()` (e.g. with `ai()` and `da()`) actually correct?**
  _`o()` has 13 INFERRED edges - model-reasoned connections that need verification._
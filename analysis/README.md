# Analysis scripts

The scripts that produced the study's evaluation records, the synthetic
anomaly corpus features and the anomaly detector arms, copied here from the
private manuscript repository so they can be read, diffed and cited alongside
the published artifacts. The artifacts themselves are release downloads, not
repository files: see
[Published artifacts](../README.md#published-artifacts) in the top level
README for each asset, its sha256 and its download URL.

These are research scripts, not a packaged library. They were written to run
on the study's own cluster and on the full corpus, so many of them take
absolute paths, expect data directories that are not in this repository, and
import from the thesis checkout's own `src/`. They are published for reading
and for adapting, not as a turnkey pipeline.

## Where to start

- `paired_bootstrap_heldout.py`, `paired_bootstrap.py`: paired bootstrap over
  the per game error CSVs. `paired_bootstrap_heldout.py` is the one that
  applies to the released held out test CSVs.
- `aggregate_heldout_results.py`, `score_test_split.py`,
  `rebucket_by_rating.py`, `corpus_composition.py`, `naive_floor.py`: the
  held out test evaluation, its per rating band breakdown, the corpus
  composition record and the naive predictor floor.
- `extra_arm_a0.py` through `extra_arm_a4.py`, `extra_arm_a1_on_a3g.py`,
  `extra_arms_common.py`, `aggregate_extra_arms.py`,
  `train_anomaly_detector.py`, `extract_detector_features_v2.py`: the anomaly
  detector arms and the features they were trained on. The app's own ports of
  the A0g, A3g and A4 arms live in `src/` and are parity checked against
  these under `experiments/`.
- `deploy_measure.py`, `sample_test_games.py`, `slow_file_server.py`: the
  deployment latency and offline against live agreement measurements.
- `preprocess_fast.py`, `preprocess_onepass.py`, `resume_pass2.py`,
  `scan_equivalence_check.py`, `compare_processed_dirs.py`,
  `check_corpus_store.py`, `gen_py_random_vectors.py`: the corpus
  preprocessing implementations and the equivalence checks that showed they
  select the same games in the same order.
- `synthesize_anomaly_clocks.py`, `make_edge_case_pgns.py`,
  `bot_check_*.py`, `plot_results_figures.py`: the synthetic clock
  countdown, the edge case fixtures, the bot account error check and the
  manuscript figures.

The shell runners that drove these on the cluster are not copied here,
because they encode that machine's paths and queueing rather than any method.

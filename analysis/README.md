# Analysis scripts, corpus scanner and written record

The scripts that produced the study's evaluation records, the synthetic
anomaly corpus features and the anomaly detector arms, the Rust scanner that
sampled the rating corpus, and the study's own written record of what was
built and what was decided, copied here from the private manuscript repository
so they can be read, diffed and cited alongside the published artifacts. The artifacts themselves are release downloads, not
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

- `run_heldout_test_eval.sh`, `run_v2_rating_cheap.sh`, `run_v2_engine.sh`,
  `extra_arms_launch.sh`, `extra_arms_run.sh`: the shell runners that drove
  the scripts above on the cluster. `test_pipeline_recovery.sh` is different
  in kind: it exercises the corpus driver's claim, staleness and crash
  recovery paths for real, including `SIGKILL`, without downloading a month.

Note that these documents and runners refer to the scripts by their path in
the private repository, `analysis/scripts/<name>.py`. Here they are flat, at
`analysis/<name>.py`.

## The Rust scanner

`scanner-rs/` is `clkscan`, which was the production path for building the
rating corpus. Its own module doc comments are the explanation: `src/main.rs`
for the scan and the command line, `src/reservoir.rs` for Algorithm R, and
`src/pyrandom.rs` for why the CPython random number generator had to be ported
byte for byte rather than approximated. `cargo test` runs the equivalence
suite against vectors the interpreter itself generated.

## The written record

The `*.md` files are the study's working documents: pilots, plans, reviews,
audits and results, each dated and each carrying its own status. They are the
record of how the corpus was made, what ran and what was decided, and several
of them are cited from the scripts and from the top level README.

Where to start depends on the question:

- Corpus construction: `rust-pass2-preprocessor.md` (the three preprocessing
  implementations and the evidence that they agree),
  `corpus-pipeline-bottleneck-verdict.md` and `corpus-pipeline-fix-plan.md`
  (the scan phase fast path), `corpus-pipeline-hardening.md`,
  `corpus-parallel-pipeline-review.md` and `corpus-resilience-review.md` (the
  driver), `corpus-storage-solution.md`, `faster-scan-research.md`.
- Rating model: `170k-verification.md`, `comparison-methodology.md`,
  `fix-review.md`, `seed-rerun-results.md`,
  `stage2-tuning-preregistration.md`, `stage2-sweep-results.md`,
  `stage2-sweep-methodology-audit.md`, `lr3e4-5seed-confirmation-results.md`,
  `heldout-test-eval-results.md`, `bot-account-error-check.md`,
  `deployment-measurements.md`.
- Synthetic anomaly corpora and detectors: `anomaly-corpus-pilot.md`,
  `anomaly-corpus-generated.md`, `anomaly-corpus-v2-notes.md`,
  `anomaly-clock-synthesis-plan.md`, `anomaly-clock-review.md`,
  `anomaly-clock-fixes.md`, `anomaly-clock-recalibrate.md`,
  `anomaly-eval-plan-2026-08-30.md`, `anomaly-eval-results.md`,
  `anomaly-v2-feature-extraction.md`, `anomaly-extra-arms-plan.md`,
  `anomaly-extra-arms-runbook.md`, `a3-player-level-results.md`,
  `schema-completeness-check.md`, `closed-account-data-readiness-plan.md`.

A document's status line is load bearing. Several are explicitly superseded or
obsolete and say so at the top, and a few were written against corpus figures
that a later scope decision changed. Read the status before reading the body.

These documents were written for the study's own use, so they name cluster
paths and commands. Those carry placeholder values here: `<user>@<hpc-host>`
for the SSH destination, `~/.ssh/<key>` for the key, and `~/<workdir>` for the
working directory the code was deployed to.

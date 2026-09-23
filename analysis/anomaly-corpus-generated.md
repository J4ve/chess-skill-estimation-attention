# Anomaly-validation synthetic corpus — generated (2026-08-17)

Deliverable for the Anomaly Validation Protocol, Part 1 (synthetic bot-vs-bot
generation). Follows the pilot in `analysis/anomaly-corpus-pilot.md`, whose
cost/storage extrapolation led to the approved scaled-down target: 1,000
games/case instead of the manuscript's original 5,000/case, keeping the full
9-band × 2-engine × 5-rate factorial coverage.

## Result

All 90 cases (9 Maia bands × 2 engines × 5 substitution rates) generated
successfully, **1,000 games each, exactly as planned — no case short of
target.**

- **Total games: 90,000**
- **Total storage: 22.96 GiB** (24.65 GB decimal) at
  `~/<workdir>/data/anomaly_corpus/` on HPC, one subdirectory per case
  (`{band}_{engine}_r{rate}/game_*.pkl`), e.g. `1500_stockfish16_r015/`.
- Generator/driver: `prototype/src/generate_anomaly_corpus.py` +
  `prototype/src/run_anomaly_corpus_full.sh` (job queue over all 90 cases,
  resumable via a `.done` marker per case directory).

### By Maia band

| Band | Games | Storage |
|------|-------|---------|
| 1100 | 10,000 | 2.35 GiB |
| 1200 | 10,000 | 2.40 GiB |
| 1300 | 10,000 | 2.40 GiB |
| 1400 | 10,000 | 2.53 GiB |
| 1500 | 10,000 | 2.58 GiB |
| 1600 | 10,000 | 2.58 GiB |
| 1700 | 10,000 | 2.76 GiB |
| 1800 | 10,000 | 2.70 GiB |
| 1900 | 10,000 | 2.66 GiB |

### By engine

| Engine | Games | Storage |
|--------|-------|---------|
| Stockfish 16 | 45,000 | 11.29 GiB |
| Lc0 (small ref net) | 45,000 | 11.66 GiB |

### By substitution rate

| Rate | Games | Storage |
|------|-------|---------|
| 0% (clean control) | 18,000 | 5.10 GiB |
| 5% | 18,000 | 4.71 GiB |
| 15% | 18,000 | 4.54 GiB |
| 30% | 18,000 | 4.44 GiB |
| 60% | 18,000 | 4.16 GiB |

(Storage decreases slightly with substitution rate: higher-rate games are
shorter on average — engine-vs-engine-like play ends games faster than the
Maia-vs-Maia clean control, which more often runs out the 100-ply cap.)

## Generation run

Ran on HPC as two passes:

1. **Main sweep**: 3 concurrent workers (`WORKERS=3`), ~7 hours wall-clock,
   88/90 cases succeeded directly.
2. **Repair pass**: 2 cases needed a retry, run sequentially
   (`WORKERS=1`) to avoid the contention that caused the original failures:
   - `1100_lc0_r00` — the lc0 engine process segfaulted (exit code -11)
     partway through (~200/1000 games in), most likely from concurrent CUDA
     context contention across the 3 workers' lc0 processes plus the
     already-running stage-2 training sweep sharing the same 4 GPUs.
   - `1200_lc0_r060` — the process hung (6 hours elapsed, ~2 minutes of
     actual CPU time, stuck at 855/1000 games with zero further progress;
     flagged and confirmed by firstmate's ground-truth check, then killed).
   Both retried cleanly to completion in under 4 minutes each once run
   without concurrent contention, confirming the failures were environment
   contention, not a bug in the generation logic itself.

Both engines ran with `Threads=1` throughout; no case needed more than one
retry.

## Known gap carried over from the pilot (not yet fixed)

- `Clocks` records each move's actual engine think-time (near-zero for the
  fast settings used here — Stockfish `movetime=50ms`, Lc0 `nodes=100`, Maia
  `nodes=1`), not a Lichess-style remaining-time countdown. The trained
  model's clock feature (`mean=273 std=380`) expects the latter, so this
  corpus is not yet directly inference-ready through the model's existing
  clock-conditioned path — synthesizing a plausible base+increment countdown
  per move is still open follow-up work before running ROC-AUC/precision/KS
  evaluation through the trained model in Chapter 4.

## Status

Generation complete and verified (exact game counts, per-case `.done`
markers, storage measured directly from disk). This is the Part 1 synthetic
corpus deliverable; Part 2 (Lichess closed-account sample) remains out of
scope, pending Lichess's reply.

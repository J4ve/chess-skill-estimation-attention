# Anomaly-validation synthetic corpus — Part 1 pilot (2026-08-17)

Scope: PART 1 only (synthetic bot-vs-bot generation via the Maia ladder +
Stockfish 16 + Lc0). PART 2 (Lichess closed-account sample) is out of scope —
the captain has emailed Lichess directly and is awaiting a reply.

Provenance: this was the captain's own priority (task named for dispatch),
not a firm adviser directive. Omori's actual 2026-08-17 wording (see
`contexts/consultation_log.md`, "Update (2026-08-17, synthetic-corpus
directive)") is a soft "maybe it makes sense to try to generate/collect this
synthetic dataset while waiting for their [Lichess's] response" — a
suggestion, not a mandate.

## What was built

- `prototype/src/generate_anomaly_corpus.py` — generates bot-vs-bot games.
  One player ("suspect") is normally a Maia model for a given rating band; at
  each of the suspect's own turns, with probability `--substitution-rate`,
  the move is replaced by a reference engine's move (Stockfish 16 or a small
  Lc0 network) instead. The opponent is always a clean Maia at the same band.
  Every game therefore carries a per-ply ground-truth label
  (`move_is_substituted`) of which moves were engine-injected — something no
  real Lichess game can provide. Output schema mirrors
  `format_data.parse_game` (12-plane board tensors, UCI move strings,
  Lichess-style clock strings) with extra label fields (`suspect_color`,
  `engine`, `substitution_rate`, `maia_band`) appended.
- `prototype/src/run_anomaly_pilot.sh` — drives one Maia band × one engine ×
  all 5 substitution rates (0/5/15/30/60%) at 500 games/case.

## HPC engine setup (now installed at `~/<workdir>/engines/` on HPC)

- **Stockfish 16**: official prebuilt AVX2 Linux binary (HPC CPU supports
  AVX2/BMI2). `engines/stockfish/stockfish-ubuntu-x86-64-avx2`.
- **Lc0 v0.31.2**: no Linux binary is published in GitHub releases (only
  Windows); built from source (`meson`/`ninja`, installed via pip into the
  `ratingnet2` conda env; `protobuf`/`pkg-config` via `conda install -c
  conda-forge`). Auto-detected CUDA and built a `cuda-fp16` GPU backend —
  fine, since a single `nodes=1`/`nodes=100` inference call is a negligible
  add to the GPUs already running the stage-2 sweep (33–41% util at time of
  build). Binary: `engines/lc0/build/release/lc0`.
- **Maia weights**: all 9 ladder bands (1100–1900, step 100) downloaded from
  `CSSLab/maia-chess` GitHub raw, `.pb.gz`, run through `lc0`.
  `engines/maia_weights/maia-{band}.pb.gz`. Move selection: `nodes=1` (the
  standard Maia usage — the network's root policy without search approximates
  human-like play).
  **Komodo is not used anywhere** — Stockfish 16 and Lc0 only, per spec.
- **Lc0 reference net for the "cheater" side**: `744706.pb.gz` (~6.3 MB, an
  older/smaller net from `storage.lczero.org/files/networks-contrib/`), not
  a modern transformer net like BT4 (382 MB) — a BT4-scale net would be far
  too slow for CPU/shared-GPU self-play at this volume. Run at `nodes=100`.
  Stockfish moves are capped at `movetime=50ms`. Both are deliberately fast,
  modest settings chosen for generation feasibility at this scale, not for
  maximum engine strength; both remain far stronger than any human band.
  Both engines run `Threads=1` to stay a light neighbor on the shared box.

## Pilot run (band=1500, engine=Stockfish 16, all 5 rates, 500 games/case)

Ran as a background job on HPC (`~/<workdir>/src/run_anomaly_pilot.sh`),
sequential (one worker), while the stage-2 hyperparameter sweep (10+
concurrent training runs) and the 1.2M-game `corpus_stream.sh` download were
also active on the same 32-core/4-GPU box (load average 42–57 throughout).

| Rate | Games | Wall-clock | s/game |
|------|-------|-----------|--------|
| 0% (clean control, Maia only) | 500 | 295.4s | 0.591 |
| 5% | 500 | 392.0s | 0.784 |
| 15% | 500 | 502.4s | 1.005 |
| 30% | 500 | 617.3s | 1.235 |
| 60% | 500 | 823.7s | 1.647 |
| **Total** | **2,500** | **2,630.8s (43.8 min)** | **avg 1.052** |

Cost scales with substitution rate (Stockfish's 50ms/move is slower than
Maia's `nodes=1`, so more substituted plies → slower games), as expected.

Storage: 2,500 files, 632.2 MB total → **258.96 KB/game** average (dominated
by the raw float32 12×8×8 position tensor per ply, ~100 plies/game — same
encoding as training data, so no surprises there).

**Supplementary Lc0-cheater check** (not the full spec'd pilot; a quick 20-game
sanity/cost check at rate=30% only, to have real numbers for both engines):
1.456 s/game — about 18% slower than Stockfish at the same rate (GPU search
overhead vs. a flat 50ms movetime). Other rates for Lc0 are estimated below
by applying this same ~1.18× ratio to the measured Stockfish curve — not
independently measured, to avoid adding a second concurrent workload to an
already heavily loaded box (load 57/32 at the time).

## Full-scale (450,000-game) extrapolation

Manuscript target: 9 Maia bands × 2 engines × 5 rates × 5,000 games/case =
**450,000 games**.

Per-game cost (measured Stockfish curve; Lc0 curve estimated via the 1.18×
ratio from the one measured Lc0 data point):

| Engine | avg s/game across 5 rates |
|--------|---------------------------|
| Stockfish 16 (measured) | 1.052 |
| Lc0 (1 rate measured, 4 estimated) | 1.220 |
| **Combined average** | **1.136** |

**Compute**: 450,000 games × 1.136s ≈ **511,000s ≈ 142 CPU/GPU-worker-hours
(~5.9 days) on a single sequential worker.** Generation is embarrassingly
parallel (each game independent), but the box is already heavily
oversubscribed (load average 42–57 on 32 cores from the stage-2 sweep +
corpus_stream during this pilot) — running many concurrent workers would
starve those runs. At a **modest** concurrency matching this project's
existing `num_workers ≈ cores / concurrent_runs` practice:

| Concurrent workers | Wall-clock |
|---------------------|-----------|
| 1 (sequential) | ~142 h (~5.9 days) |
| 2 | ~71 h (~3.0 days) |
| 3 | ~47 h (~2.0 days) |

**Storage**: 450,000 games × 258.96 KB ≈ **113.8 GiB (~117 GB)**. `/home` on
HPC currently has 514 GB free out of 7.3 TB (93% full pool-wide, shared
across the whole team) — this would consume **~22% of all remaining free
space**, concurrently with the still-active 1.2M-game `corpus_stream.sh`
download/preprocess pipeline (final footprint not yet known) and the stage-2
sweep's own checkpoint storage. This is the real resource commitment the
task brief flagged — not affordable to commit blind.

## Recommended starting size (if 450k is not affordable)

Scale games/case down 5×, from 5,000 to **1,000**, keeping the full 9-band ×
2-engine × 5-rate factorial coverage (no case is dropped — every
band/engine/rate combination the manuscript wants a data point for still
gets one, just a smaller one):

- Total: 9 × 2 × 5 × 1,000 = **90,000 games**
- Storage: 90,000 × 258.96 KB ≈ **22.7 GiB (~23 GB)** — a modest, easily
  affordable footprint.
- Compute: 90,000 × 1.136s ≈ 102,200s ≈ 28.4h sequential; **~10–14h at 2–3
  modest concurrent workers.**
- 1,000 games/case is still ample sample size for the planned
  ROC-AUC/precision/KS tests in Chapter 4 (the pilot itself only used
  500/case). Can be scaled up per-case later (e.g., just the bands/engines
  that turn out most informative) if time and storage allow, rather than
  committing the full 5,000/case up front.

## Known gaps to fix before real (non-pilot) generation

- **Clock field is currently think-time, not a countdown.** `Clocks` records
  each move's actual engine think-time (near-zero for fast GPU/50ms-movetime
  moves), whereas real Lichess `%clk` annotations are a *remaining-time*
  countdown from a time-control budget. The trained model's clock feature
  (`mean=273 std=380`, see root `CLAUDE.md`) expects the latter. This doesn't
  affect the cost/storage estimate above (same file size either way), but the
  corpus isn't directly inference-ready through the model's existing
  clock-conditioned path until this is fixed — e.g. simulate a plausible
  base+increment budget and decrement it per move, the way real games do.
- Lc0-side per-rate costs above (5/15/60%) are estimated, not measured
  independently — worth a real (not extrapolated) measurement once the
  full-scale go-ahead is decided, since it's cheap to do alongside the first
  real batch.

## Status

Pilot complete, both engines set up and verified on HPC, cost/storage
estimated. **Stopped here per task brief** — did not launch the full-scale
(or even the recommended-starting-size) generation; that requires a
decision on affordability first.

# Deployment measurements — per-move latency and offline-vs-live agreement

Fills the two Chapter 4 TODOs in `chapter4.tex` §"Deployment Measurements": per-move
end-to-end inference latency (mean and tail, with hardware), and numerical agreement
between offline batch scoring and the live path on 100 matched games. Measured
2026-09-13. Read/measure only: nothing under `prototype/` was modified.

## Host, hardware, code version

- **Compute host:** HPC (`<hpc-host>`), work dir `~/<workdir>`, conda env
  `ratingnet2`, `torch 2.13.0+cu130`, cuDNN 9.2.0.
- **GPU arm:** NVIDIA RTX A5000 (24 GB), one of the node's four A5000s, selected because
  it was idle (`nvidia-smi`: 0% util, only the shared `milvus` service resident) — GPU 0
  was excluded because it had another user's process running. Existing tmux sessions
  (`engleak`, `engleak2`, `traindet2`) were left untouched; measurements ran in a new
  session `deploymeas`.
- **CPU-only arm:** Intel(R) Xeon(R) Silver 4309Y @ 2.80GHz, 32 cores visible, default
  PyTorch intra-op threading (no thread-count override). This is an HPC server CPU, not
  a reviewer's laptop; treat the CPU numbers as an upper bound on available compute
  rather than a laptop-equivalent figure — a real laptop CPU is plausibly slower per
  core but the model triggers negligible parallelism at batch size 1 either way (see
  Latency, below).
- **Code version measured:** the HPC's `~/<workdir>/src/` is a **symlink** to
  `prototype/src` in the checked-out repo (`prototype/src -> src`, confirmed with
  `readlink -f`), and `diff -rq prototype/src src` returned no differences (exit 0).
  There is currently no separate hand-copied HPC version to diverge from — the code
  measured is exactly this repo's `prototype/src/` at the commit this branch was cut
  from.
- **Checkpoint measured:** `models/preflight_check_2m/best_model.pth` (the frozen
  tuned-attention arm named in the task brief). Its stored `params` confirm
  `use_attention: true`, `use_anomaly: false`, `bidirectional: true`,
  `split_seed: 42` — matching the frozen split manifest. Loading it with
  `use_attention=True` produced **zero missing and zero unexpected `state_dict` keys**
  for the attention module (`attention.v`, `attention.key_projection.weight`,
  `attention.query_projection.weight` all present and loaded), so the M1 defect from
  `analysis/170k-verification.md` (random-init attention corrupting predictions) does
  **not** reproduce for this checkpoint.

  **Caveat — what the API actually loads by default.** `prototype/src/api.py`'s
  `_discover_checkpoint()` is hardcoded to look for `models/model_55.pth`, not the
  `preflight_check_2m` checkpoint. On the HPC, `models/model_55.pth` exists but is a
  **different, older checkpoint**: its `params` dict has no `use_attention` /
  `use_anomaly` / `split_seed` keys at all, and carries the pre-fix defaults
  (`epochs: 100`, `val_batch_size: 8192`) — it is a plain baseline checkpoint, not the
  tuned-attention arm. If the API were started as-is on this HPC checkout, it would
  silently serve that baseline checkpoint, not the one this task was asked to measure.
  This is a configuration/discovery mismatch, not a blocker: it does not corrupt
  predictions (`api.py` still reads `use_attention`/`use_anomaly` from whatever
  checkpoint it finds, so `model_55.pth` would be served correctly as a baseline), but
  it means "what the prototype loads" and "the frozen configuration" are two different
  files today. This measurement bypasses `_discover_checkpoint()` and loads
  `models/preflight_check_2m/best_model.pth` directly, using the same
  parameter-driven construction logic `api.py::_load_model()` uses.
- **`use_anomaly` override:** the checkpoint's own `params` say `use_anomaly: false`.
  `AnomalyDetector` (`prototype/src/anomaly.py`) has **zero learnable parameters** — it
  is pure post-hoc arithmetic over the rating curve and the attention weights (confirmed
  in `analysis/170k-verification.md` §H.1a) — so there is nothing to load regardless of
  this flag. The measurement scripts build the model with `use_anomaly=True` explicitly
  so `S_att`/suspicion scores can be computed; this changes no loaded weight, and the
  resulting `state_dict` load still reports zero unexpected keys.
- **`S_max` / `S_mean`:** not implemented in `anomaly.py` (only the attention-weighted
  `S_att`/`combined_score` is). Computed in the measurement script from the
  `per_move_deviation` tensor the module already returns (`max` / `mean` over plies,
  summed across white+black), without modifying prototype code.
- **`R_baseline`:** `api.py`'s live default (self-prediction fallback) is a known,
  separately-tracked bug (`analysis/170k-verification.md` §H.1c / M4) that this task was
  not asked to fix. To keep this specific measurement meaningful independent of that bug,
  both the offline and live computations here are given the **same explicit baseline**:
  the game's true pre-game `WhiteElo`/`BlackElo` from the corpus record. This sidesteps
  M4 rather than exercising it — it is not a re-measurement of the baseline bug.

## Method

Scripts: `analysis/scripts/sample_test_games.py`, `analysis/scripts/deploy_measure.py`.
Both import `chess_rating_net` / `format_data` from the prototype's `src/` directory;
neither edits it. Run on the HPC via a `deploymeas` tmux session (job completed in
under two minutes; log tails in the session, not otherwise preserved).

**Game sample.** 100 games drawn with a fixed seed (`random.Random(20260913).sample`)
from `test_files` in the frozen split manifest `data/split_manifest_2p55M_seed42.json`
(internal `sha256` field `dee7d117...`, confirmed matching; 255,000 test games,
`split_seed: 42`). The 100 selected game ids are recorded in
`analysis/deployment_measurements/sampled_test_games_100.json`, alongside the manifest
hash and sample seed, so the sample is reproducible and auditable. Games loaded from
the ready `GameBlobStore` at `data/ratingnet_store_restore/` (a view over
`data/corpus_store_backup.sqlite`). Games capped at 100 plies (`--max-plies 100`,
matching the model's trained sequence cap); sampled game lengths ranged 14–100 plies
(median 68.5, mean 67.3); 16 of the 100 games are long enough to hit the 100-ply cap,
75 reach at least 50 plies.

**Latency.** `chapter3.tex`'s own bidirectionality discussion states the deployed
design "recomputes the bidirectional pass over plies 1 through t at each move
arrival" — i.e., the live path has no incremental/cached state; a new move triggers a
full forward pass over the whole prefix seen so far. This is exactly what was measured:
for ply t = 1..T of each game, one forward pass over `positions[:t]` (batch size 1),
timed individually with `torch.cuda.synchronize()` bracketing on the GPU arm and
`time.perf_counter()` on both. The first 5 games (of the 100) were used as a discarded
warm-up (three forward passes each, at t = 1, T/2, T) before any timed call, to remove
one-time CUDA/cuDNN kernel-selection and JIT overhead from the reported distribution.
6,731 timed forward calls resulted per device (100 games × their respective ply counts).
Latency here is genuinely end-to-end for one new move: CNN trunk → BiLSTM over the
recomputed prefix → attention → rating head → anomaly scoring, all inside the timed
region.

**Offline-vs-live agreement.** For each game: an *offline* pass is one forward call over
the full-length sequence (all T plies at once) — what batch/offline scoring does. The
*live* simulation is the same per-ply loop used for latency, which for t = T is,
tensor-for-tensor, the identical forward computation as the offline pass (same weights,
`model.eval()`, no dropout, same full-length input). Two distinct comparisons are
reported because these two paths are **not** expected to agree everywhere by
construction:

1. **Final-ply agreement** (what `chapter3.tex`'s "within floating-point tolerance"
   claim is about): offline's last-ply output vs. live's t = T step, for the predicted
   white/black rating and for `S_att`, `S_max`, `S_mean`.
2. **Full per-ply curve agreement** (informational, not the manuscript's claim):
   offline's per-ply curve (informed by the whole game, since the LSTM is bidirectional)
   vs. live's per-ply value at each t (informed only by plies 1..t). These are expected
   to diverge mid-game — quantified below rather than hidden.

## Results

### Latency (per-move, end-to-end, batch size 1, post warm-up)

Overall, pooled across all plies 1–100 of all 100 games (n = 6,731 forward calls):

| Device | n | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|---:|
| GPU — NVIDIA RTX A5000 | 6,731 | 3.42 | 3.21 | 5.46 | 7.81 | 11.18 |
| CPU — Xeon Silver 4309Y | 6,731 | 7.73 | 8.60 | 11.94 | 13.26 | 126.02 |

The CPU max (126 ms) is a single outlier (n=6,731; p99 is 13.26 ms), consistent with an
OS/scheduler hiccup rather than a systematic tail — the rest of the CPU distribution is
well-behaved.

Because the live path recomputes the whole prefix on every move, latency grows with ply
(more context = more BiLSTM/CNN work per call), so the pooled numbers above blend short
and long prefixes. Per-brief, latency is also reported at fixed plies 10, 50, 100
(n = number of the 100 games long enough to reach that ply):

| Ply | Device | n | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|---:|---:|
| 10 | GPU | 100 | 2.19 | 2.14 | 2.28 | 3.06 |
| 10 | CPU | 100 | 5.31 | 6.52 | 8.38 | 9.35 |
| 50 | GPU | 75 | 3.80 | 3.79 | 3.96 | 4.08 |
| 50 | CPU | 75 | 8.11 | 9.75 | 10.74 | 11.00 |
| 100 | GPU | 16 | 5.92 | 5.90 | 6.26 | 6.39 |
| 100 | CPU | 16 | 13.00 | 13.04 | 14.31 | 14.76 |

Both devices stay well under 100 ms/move even at the 100-ply cap (GPU p99 ≈ 6.4 ms,
CPU p99 ≈ 14.8 ms at ply 100), i.e. comfortably inside soft-real-time budgets for every
Lichess time control including bullet.

### Offline-vs-live agreement (100 games)

**Final-ply agreement — exact on both devices, all 100 games:**

| Quantity | GPU max abs diff | GPU mean abs diff | CPU max abs diff | CPU mean abs diff |
|---|---:|---:|---:|---:|
| White rating (Elo) | 0.0 | 0.0 | 0.0 | 0.0 |
| Black rating (Elo) | 0.0 | 0.0 | 0.0 | 0.0 |
| S_att | 0.0 | 0.0 | 0.0 | 0.0 |
| S_max | 0.0 | 0.0 | 0.0 | 0.0 |
| S_mean | 0.0 | 0.0 | 0.0 | 0.0 |

Tolerance used: bit-exact (`abs diff == 0.0`) at float32 precision — better than the
"floating-point tolerance" `chapter3.tex:265` asks for. This is expected by
construction here: at the final move, the live path's forward pass is tensor-identical
to the offline pass (same full-length input, same weights, eval mode, no dropout), so
there is no independent source of numerical divergence to produce a nonzero diff on
either device.

**Full per-ply curve — diverges mid-game, quantified, not hidden:**

| Side | max abs diff over games (Elo) | mean abs diff over games (Elo) |
|---|---:|---:|
| White | 1152.14 | 174.19 |
| Black | 1155.79 | 174.83 |

Cause: the LSTM is bidirectional (`bidirectional: true` in the checkpoint's own
params), so offline's per-ply prediction at ply t is informed by plies t+1..T of the
completed game, while the live path's value at ply t is computed from a prefix that
ends at t — it cannot see the future. Divergence is largest early in a game (live has
almost no context yet, offline already has the whole game) and collapses to exactly
zero at t = T, where the two computations become the same input. This is the same
retrospective-vs-causal distinction `chapter3.tex`'s bidirectionality paragraph already
names; it is not a defect in this measurement, and it is a different question from the
final-ply "floating-point tolerance" claim above, which holds exactly.

## Caveats

1. **Checkpoint-discovery mismatch** (above): `api.py::_discover_checkpoint()` finds
   `models/model_55.pth` by default, which on this HPC checkout is an older
   non-attention baseline, not `models/preflight_check_2m/best_model.pth`. This
   measurement targeted the latter directly; if the prototype is deployed as-is, it
   would not currently serve the checkpoint these numbers describe.
2. **CPU arm is a server CPU, not a laptop.** Numbers should be read as "single-request
   CPU-only latency on a modern server core," not as a laptop-equivalent figure.
3. **Ply-100 sample size is small** (n = 16 of the 100 games) because most sampled games
   are shorter than 100 plies; the ply-100 latency figures have wider uncertainty than
   the pooled or ply-10/50 figures.
4. **`R_baseline` sidestep** (above): this measurement does not exercise or validate the
   API's self-referential baseline fallback (M4) — a fresh baseline (true Elo) was
   supplied explicitly to both paths so the agreement result is not confounded by that
   separate, already-tracked bug.
5. No `advisor` tool was available in this environment to consult before/after the
   design, per the task brief's conditional instruction.
6. **GPU choice vs. project convention.** `nvidia-smi` at measurement time showed GPU 0
   running another user's job and GPUs 1-3 idle (0% util, only the shared `milvus`
   service resident), so GPU 1 was used per the task brief's "check `nvidia-smi` and use
   an idle GPU" instruction. `AGENTS.md` separately records a project convention of
   avoiding GPU 1 for some pipelines (reserved historically for `deepcnn_arm5` /
   anomaly-v2 feature extraction, both already complete per that note). The measurement
   job was brief (~2 minutes, low memory) and no conflict was observed, but future
   HPC work in this repo should prefer GPU 0/2/3 once free, consistent with that
   convention.

## Reproduction

```
# on the HPC, conda env `ratingnet2`, from ~/<workdir>
python3 analysis/scripts/sample_test_games.py \
  --manifest data/split_manifest_2p55M_seed42.json \
  --n 100 --seed 20260913 --out games_100.json

python3 analysis/scripts/deploy_measure.py \
  --src-dir src --checkpoint models/preflight_check_2m/best_model.pth \
  --store-dir data/ratingnet_store_restore --games-file games_100.json \
  --device cuda --gpu-index <idle GPU> --warmup-games 5 --out-json full_gpu.json

python3 analysis/scripts/deploy_measure.py \
  --src-dir src --checkpoint models/preflight_check_2m/best_model.pth \
  --store-dir data/ratingnet_store_restore --games-file games_100.json \
  --device cpu --warmup-games 5 --out-json full_cpu.json
```

Raw output: `analysis/deployment_measurements/latency_agreement_gpu_a5000.json`,
`analysis/deployment_measurements/latency_agreement_cpu_xeon4309y.json` (per-game
detail for all 100 games, both latency and agreement), and
`analysis/deployment_measurements/sampled_test_games_100.json` (the 100 game ids plus
the manifest hash and sample seed).

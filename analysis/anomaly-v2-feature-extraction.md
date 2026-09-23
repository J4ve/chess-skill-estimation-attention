# Anomaly corpus v2 - detector feature extraction

Feature extraction for the **supervised** anomaly (cheat) detector over the
completed v2 synthetic anomaly corpus. Detector training is a separate downstream
task and is **not** done here.

- Script: `analysis/scripts/extract_detector_features_v2.py` (v2 variant of
  `analysis/scripts/extract_detector_features.py`, which stays on branch
  `fm/anomaly-trained-detector` and is unchanged).
- Launchers: `analysis/scripts/run_v2_rating_cheap.sh` (GPU),
  `analysis/scripts/run_v2_engine.sh` (CPU, multi-day).
- Runs on the CSPC HPC: `ssh -i ~/.ssh/<key> <user>@<hpc-host>`,
  `cd ~/<workdir>`, `source ~/miniconda3/etc/profile.d/conda.sh && conda activate ratingnet2`.
- Branch `fm/anomaly-v2-feature-extract`, local-only. The `.npz` outputs are **not**
  committed (too big); only the scripts and this note.

## Inputs

| item | value |
|---|---|
| corpus | `~/<workdir>/data/anomaly_corpus_v2/` |
| cells | 134 = 9 bands x 2 engines x 7 rates (126) + 8 `{band}_hardneg` |
| games | 268,000 (2,000 / cell), 63 GiB, every cell has `.done` |
| rates | `r00 r02 r05 r10 r20 r40 r60` (v1 was `00 005 015 030 060`) |
| frozen rating checkpoint | `models/preflight_check_2m/best_model.pth` |

**Frozen checkpoint** = the tuned attention arm named in the freeze decision
(firstmate, 2026-09-04): 4-layer CNN + Bahdanau attention (`attention_dim 64`),
lr 3e-4, dropout 0.5, weight_decay 1e-5, batch 32, 60 epochs, patience 5,
split_seed 42, full 2,550,000-game corpus, **best val loss 172.38**. The
experiment directory is `models/preflight_check_2m/` (the name is a historical
misnomer; `hpc_training_heal.sh:43` and `logs/preflight_check_2m.log` confirm the
hyperparameters and the 172.38 val loss). `best_model.pth` is the
best-validation-epoch checkpoint (epoch 58).

### v2 pickle schema (what changed vs v1)

- `Clocks`: **real remaining-time countdown**, one `"H:MM:SS"` string per ply
  (v1 stored think-time in a sidecar). Read `raw["Clocks"]` directly; no sidecar.
- `move_is_substituted`: per-ply bool, full length (book plies = `False`).
- `engine_differs_from_maia`: per-ply, `None` on plies where no engine move was
  drawn, else `True` / `False` (`False` only when the resample cap was hit).
- `maia_argmax_move`: per-ply UCI string or `None`; non-`None` on every
  suspect ply from the substitution-eligible index (ply 17) onward.
- `engine_setting`: `{"axis": "depth"|"nodes", "value": int}` or `None`
  (r00 / hardneg).
- `suspect_band`, `opponent_band`, `maia_band`, `hard_negative`, `n_book_plies`,
  `opening_line`, `nominal_vs_effective_rate` (dict with `n_eligible_plies`,
  `n_substituted`, `n_differs`, `nominal`, `effective`).
- `Positions`: list of `torch.float32 [12,8,8]` tensors, `len == len(Moves)`,
  capped at 100 plies. `WhiteElo == BlackElo == maia_band`. `Time` = `"base+inc"`.

## Stage 1 - `rating_cheap` (GPU)

Frozen rating-model forward (`return_attention=True`) over **all 134 cells /
268k games**.

Per game (scalars): `band`, `engine`, `rate` (`-1` for hardneg), `is_hardneg`,
`suspect_idx`, `n_plies`, `white_elo`/`black_elo`, `maia_band`, `suspect_band`,
`opponent_band`, `r_base` (= `maia_band`), `game_label` (any substituted ply),
`game_label_differs` (any `engine_differs == True`), `n_sub`, `n_differs`,
`n_eligible` (eligible suspect plies, from the generator dict), `n_sub_attempts`,
`n_suspect`, `nominal_rate` (cell design parameter), `eff_rate`
(`n_differs / n_eligible`, from the generator's `nominal_vs_effective_rate`),
`eff_rate_check` (independent recompute), `nominal_rate_realized`, `opening_len`,
`hard_negative`, `engine_setting_axis` (0 none / 1 depth / 2 nodes),
`engine_setting_value`, `replay_ok`.

Per ply (flat, indexed by `game_start`, length `n_games+1`): `ply_idx`,
`is_suspect_move`, `move_is_substituted`, `engine_differs` (1/0/NaN),
`is_eligible`, `maia_argmax_present`, `r_hat_suspect`, `r_hat_other`, `alpha`
(attention), `d_t = |r_hat_suspect - maia_band|`, `run_mean`, `run_std`,
`first_diff`, `second_diff`, `is_capture`, `is_check`, `material_balance`
(suspect POV), `clock_remaining` (s), `clock_delta` (same-side think time, s).

Clocks standardised with the model's constants (`mean 273`, `std 380`,
confirmed against `src/chess_rating_net.py:136-137` /
`ChessGamesDataset` defaults); ratings de-standardised with `mean 1514`,
`std 366`.

Outputs: `analysis/v2_features/rating_cheap.npz` + `rating_cheap_meta.json`.
GPU 0 only (GPU 1 is the parallel `fullcorpus_baseline_lr3e4_arm6` training arm;
GPUs 0/2/3 are free for this task - GPU 0 chosen).

### Command

```
tmux new -s v2_rating_cheap
cd ~/<workdir> && CUDA_VISIBLE_DEVICES=0 bash analysis/scripts/run_v2_rating_cheap.sh
# marker: analysis/v2_features/.rating_cheap.DONE  (or .rating_cheap.FAILED)
# log:    logs/v2_rating_cheap.log
```

### Result

<!-- FILL: START/DONE timestamps, throughput, n_games, n_plies, size -->
- launched 2026-09-04 02:51 (+08), tmux session `v2_rating_cheap`, GPU 0.
- STATUS: __running / done__ - see `logs/v2_rating_cheap.log` tail and
  `analysis/v2_features/rating_cheap_meta.json` (`n_games` should read 268000,
  134/134 cells).

## Stage 2 - `engine` (CPU, multi-day)

Stockfish 16 (`engines/stockfish/stockfish-ubuntu-x86-64-avx2`) at **fixed
depth 20** (design range 18-20; `chess.engine.Limit(depth=20, time=5.0)` - the
5 s wall cap is a safety valve against pathological positions), **multipv 3**.

Not every ply: per game we analyse **every substituted ply** plus a **matched
sample of clean suspect-side post-book plies** from the same game
(`k = max(n_substituted, --clean_plies_per_game=10)` clean plies, sampled with a
per-game deterministic seed). Games with no substitution (r00, hardneg, and
low-rate games that drew zero substitutions) contribute `--clean_plies_per_game`
clean plies. This keeps the job to ~2-3 M engine evaluations instead of ~15 M.

Per analysed ply: `cp_loss` (best - played, clamped to 2000), `top1_match`,
`top3_match`, `abs_eval` (clamped to 10000; mate score 100000 before clamp),
`eval_swing` (White-POV eval change between successive analysed checkpoints).
Per-ply arrays are written **full length** (`n_plies` per game) with `NaN` at
un-analysed plies plus an `analyzed` mask, so `train_anomaly_detector.py`'s
index join (`eng_pp["cp_loss"][:n_real]`) still works unchanged.

Outputs: `analysis/v2_features/engine.npz` + `engine_meta.json`.

Parallel across `--sf_workers` processes (default 10; the box is 32-core and
`fullcorpus_baseline_lr3e4_arm6` + the OS need headroom - do **not** raise past
~12).

### Command

```
tmux new -s v2_engine
cd ~/<workdir> && SF_WORKERS=10 SF_DEPTH=20 bash analysis/scripts/run_v2_engine.sh
# marker: analysis/v2_features/.engine.DONE  (or .engine.FAILED)
# log:    logs/v2_engine.log
```

### Resumability

Both stages write **one `.npz` shard per cell** under
`analysis/v2_features/<stage>_shards/`. Re-running with `--resume` (both launcher
scripts pass it) skips any cell whose shard already exists and re-merges
`<stage>.npz` from all shards present at the end - so a killed tmux session is
recovered by simply re-running the same launcher. A cell interrupted mid-way is
redone from scratch (cell granularity, ~2000 games).

HPC `/tmp` is NVMe and wiped on reboot; `analysis/v2_features/` lives on
`~/<workdir>` (`/home`, survives reboots). `/home` had ~348 GiB free at
launch; the merged `.npz` files are ~1-2 GiB total (engine.npz is mostly NaN and
compresses hard).

### Result

<!-- FILL: START timestamp, depth, workers, cells done, ETA or DONE, size -->
- STATUS: __pending / running / done__.
- If still running at hand-off: ETA ~__N__ days from
  `logs/v2_engine.log`'s cell rate; marker at
  `analysis/v2_features/.engine.DONE`; resume by re-running
  `bash analysis/scripts/run_v2_engine.sh`.

## Validation done before the full run

- 134 cells discovered, 8 hardneg; band/engine filters correct.
- End-to-end mini-runs of both stages on real HPC pickles (bands 1100/1400/1500/1800,
  a few games/cell): shard write, `--resume` skip, and shard merge all verified;
  per-ply arrays align (`sum(n_plies) == len(flat array)`); `game_start[-1]`
  equals total plies.
- `eff_rate` tracks the design nominal (r05 -> ~0.05, r20 -> ~0.2, r60 -> ~0.45)
  and matches the generator's own `nominal_vs_effective_rate["effective"]`
  (v2's resample-until-differs makes `n_substituted == n_differs`, so realised
  nominal and effective rates nearly coincide - expected).
- `engine_setting_axis` = 1 (depth) for Stockfish cells, 2 (nodes) for lc0,
  0 for r00 / hardneg; `engine_setting_value` carries the depth {14,16,18,20} or
  node {200,400,800,1600} setting.
- Stockfish stage at depth 18: `top1_match` mean ~0.55 on the analysed plies,
  `cp_loss` bounded, `eval_swing` populated.

## Notes for the downstream detector-training task

- `train_anomaly_detector.py` constants need a v2 pass: `HELDOUT_RATE = 15` has no
  v2 equivalent (rates are 0/2/5/10/20/40/60); pick a held-out rate from that set.
  `make_split` keys on `(band, engine, rate)` and will see `rate == -1` for the
  hardneg cells - decide whether hardnegs go in train (as hard negatives, the
  intent) or are held out.
- New per-game columns available for the split / analysis: `is_hardneg`,
  `suspect_band`, `opponent_band`, `engine_setting_axis`, `engine_setting_value`,
  `eff_rate`, `nominal_rate`, `opening_len`.
- `game_label` = "any substituted ply"; `game_label_differs` = "any ply where the
  engine move actually differed from Maia" is the stricter positive.
- Engine features are on a **subset** of plies (substituted + matched clean).
  Pool them with the `analyzed` mask, not over all plies, when building
  suspect-vs-clean contrasts.

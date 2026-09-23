# Synthetic anomaly corpus v2 - implementation notes

Implements `data/anomaly-corpus-v2/design.md` (captain-LOCKED 2026-09-01).
Supersedes v1 (`analysis/anomaly-corpus-generated.md`). Code:

- `prototype/src/build_opening_tables.py` - per-band real-opening frequency
  tables from the 2.55M-game corpus store.
- `prototype/src/generate_anomaly_corpus.py` - rewritten v2 generator.
- `prototype/src/run_anomaly_corpus_v2.sh` - full-sweep driver (resumable).

Not in scope for this task and deliberately not done: feature extraction,
detector training, any `.tex` edit.

## Design decisions, as implemented

### 1. Real-opening seeding
`build_opening_tables.py` walks `data/corpus_store_backup.sqlite` (2,550,000
games; table `games(name, blob)`, blob = `zlib` then `pickle`). Each game's
first 12 UCI plies are counted into every Maia band whose 200-point window
`[B-100, B+100)` contains the mean of WhiteElo/BlackElo (the two edge bands are
half-open so no game is dropped; the 100-point overlap between adjacent windows
is intentional and smooths the tail). Games under 12 plies are skipped. Output:
`openings_band{1100..1900}.json`, each `{band, line_plies, n_games,
n_unique_lines, coverage, lines:[[uci_str,count],...]}` sorted count-desc. All
unique lines are kept (unique lines per band <= games per band, so no
truncation is needed at this corpus size).

Per game the generator samples a line with weight `count ** opening_temperature`
(`--opening-temperature`, default **0.7**). This is the power-law reading of the
design's "temperature 0.7 so the long tail still appears": `T<1` *flattens*
toward rare lines (a singleton line is ~15x more likely than under raw
frequency weighting). It is not the softmax-temperature reading (`count**(1/T)`,
which would sharpen); the design's stated intent ("long tail still appears")
picks the power-law form. Then a per-game cutoff in `[6, 10]` plies
(`--opening-cutoff-min/max`) is replayed and Maia takes over. Opening selection
is seeded from `(global_seed, "opening", band, game_index)` only - identical
across the rate arms of a band so clocks/openings never confound the across-rate
comparison (same rule the clock module already uses).

### 2. Maia policy sampling over the opening
lc0 v0.31.2 on the HPC exposes **no** search-time temperature option (only
`PolicyTemperature`, a monotone reshaping that does not change the argmax at
`nodes=1`). So `MaiaPolicyEngine` drives lc0 over raw UCI with
`VerboseMoveStats` on, reads the root policy prior `P` for every legal move from
the `info string <uci> ... (P: xx.xx%)` lines, and samples a move with weight
`P ** (1/tau)`, `tau = --maia-temp` (default **0.9**). Plies 1..16
(`--maia-temp-cutoff-ply`, 1-based) sample; from ply 17 the move is Maia's
argmax (lc0 `bestmove`). Applies to both suspect and opponent, symmetric, and
cut before the middlegame so the clean baseline is honest argmax Maia.

### 3. Substitution only from ply 17
`--sub-eligible-from-ply` default 17. Placement stays uniform-random: on each
eligible suspect ply a Bernoulli(`substitution_rate`) draw increments an `owed`
counter. While `owed > 0` the engine move is compared to Maia's argmax; on a
*differ* it is played (`move_is_substituted=True`, `engine_differs_from_maia=
True`, `owed -= 1`). On agreement, **resample-until-differs** (default on,
`--resample-cap` 8) defers: Maia's move is played and `owed` kept, up to 8
consecutive deferrals, after which the engine move is played anyway
(`engine_differs_from_maia=False`). This makes the per-game effective rate track
the nominal rate; both are recorded in `nominal_vs_effective_rate` and the mean
`effective/nominal` ratio is printed in the run SUMMARY (design target >= 0.9).
`maia_argmax_move` is recorded per ply for every eligible suspect ply (for
post-hoc effective-rate recompute), `None` elsewhere.

### 4. Per-game engine strength
`--engine-strength sample` (default): Stockfish search **depth** drawn per game
from `{14,16,18,20}` with a `--stockfish-time-cap-ms` (default 1500) wall-time
ceiling as a safety valve; Lc0 **nodes** from `{200,400,800,1600}`. Recorded as
`engine_setting = {"axis": "depth"|"nodes", "value": N}`. Seeded from
`(global_seed, "engstr", band, engine, game_index)` (matched across rate arms
within an engine). The design also lists a Stockfish `movetime` axis as an
alternative; depth was chosen ("more reproducible"). The unbounded depth 18-20
analysis the design mentions is a *feature-extraction* step, out of scope here;
the 1.5 s cap keeps generation wall-time bounded (depth 20 in a middlegame is
~1.2 s uncapped, measured on the HPC).

### 5. Simulated remaining-time clocks (no sidecar)
A time control `(base, inc)` is drawn per game from
`ClockParams().tc_mix` (the corpus TC mix) and a Lichess-style countdown is
written straight into `Clocks` by importing
`analysis/scripts/synthesize_anomaly_clocks.py`
(`game_streams`, `draw_time_control`, `synthesize_game_clocks`) - single-sourced,
not copied. `substituted=None` is passed so the stream is **label-blind**
(identical model for Maia and substituted moves), and `band` is passed so the
Sigman-et-al. band-dependent pacing/dispersion applies. `Time` is written as
`"base+inc"` so games load straight through `ChessGamesDataset` (which also
needs the `Time` key v1 lacked). `--clock-mode constant` produces the flat-clock
leak-check variant (constant clock, no increment accrual), the v2 equivalent of
v1's `data/anomaly_clocks_constant/`.

### 6. Hard-negative strength-mismatch cells
`--hard-negative` (requires `--engine none`): the suspect plays a **stronger**
Maia band `min(1900, B + d)`, `d` sampled per game from `{200,400,600}`, against
a clean band-`B` opponent, no substitution, labelled clean
(`hard_negative=True`, `suspect_band` records the actual stronger band,
`WhiteElo/BlackElo` stay at `B` - the naive pre-game baseline). Forces a
detector to key on move provenance, not "the suspect plays above the nominal
band". One cell per band 1100..1800 (band 1900 has no stronger band; 8 cells).

### 7. Rate grid
`0, 2, 5, 10, 20, 40, 60 %` (LOCKED). Rate 0 is generated with `--engine none`
(no engine process), once per `(band, engine arm)` so each engine's ROC curve
has its own matched control, matching the design's `9 x 2 x 7` factorial count.

### 8. Volume / storage - HPC disk check
Checked 2026-09-01: `~/<workdir>` is on `/dev/sda1` (`/home`), **412 GiB
free, 95 % used, shared** across the whole box; `~/<workdir>` itself is
145 GiB. v1 was ~256 KB/game.

- 3,000/cell -> ~278k games main + 24k hard-neg ~= 302k games ~= **~74 GiB**
- 2,000/cell -> ~202k main + 16k hard-neg ~= **~50 GiB**

Decision: **2,000 games/cell.** The absolute free space would technically fit
3,000/cell, but the filesystem is shared, already at 95 %, and the downstream
feature-extraction pass (rating-model forward + deeper Stockfish) has an
un-budgeted footprint on the same disk; the design pre-authorises 2,000/cell as
the "disk is tight" fallback and captain decision 2 says decide once with no
later top-up. `GAMES_PER_CELL` in the driver is the single knob if this is
revised before launch.

### 9. Per-game metadata keys
`WhiteElo, BlackElo` (= band), `Result`, `Positions`, `Moves`, `Clocks`,
`Time` ("base+inc"), `suspect_color`, `engine`, `engine_setting`,
`substitution_rate` (nominal), `maia_band`, `suspect_band`, `opponent_band`,
`hard_negative`, `move_is_substituted` (per ply), `engine_differs_from_maia`
(per ply, `None` off eligible plies), `maia_argmax_move` (per ply),
`nominal_vs_effective_rate` (`{n_suspect_plies, n_eligible_plies, n_substituted,
n_differs, nominal, effective}`), `time_control` (`{base, inc}`), `opening_line`
(seeded prefix) + `opening_line_full` (12-ply), `maia_temperature`,
`maia_temp_cutoff_ply`, `clock_mode`, `corpus_version` ("v2"), `global_seed`,
`game_index`, `n_book_plies`.

## HPC run

- Opening tables built on HPC: `~/<workdir>/data/opening_tables/`
  (`python prototype/src/build_opening_tables.py --store
  data/corpus_store_backup.sqlite --out data/opening_tables`).
- Sweep output: `~/<workdir>/data/anomaly_corpus_v2/{cell}/game_*.pkl`,
  logs in `~/<workdir>/logs/anomaly_corpus_v2/`.
- Launched in tmux session `anomaly_v2` with `CUDA_VISIBLE_DEVICES=0,2,3`
  (keeps lc0 off GPU 1, where `deepcnn_arm5` trains), `WORKERS=3`.
- Resumable: per-cell `.done` markers; re-running the driver skips finished
  cells.

## Known limitations / follow-ups (not blockers for generation)

- lc0 policy-sampling RNG is Python-side and fully seeded; the lc0 process
  itself is only used for a deterministic `nodes=1` evaluation, so games are
  reproducible from `(global_seed, band, game_index)` up to engine
  non-determinism in the Stockfish/Lc0 substitution search (depth-limited, so
  low).
- The design's "compare against real Lichess clock traces" gate
  (`analysis/anomaly-clock-synthesis-plan.md` section 8) is still open and is a
  Chapter 4 prerequisite, unchanged by this task.
- Effective/nominal ratio is measured per run from the SUMMARY line; if a rate
  arm comes in below the design's 0.9 target the grid may need biasing lower
  (design decision 7 anticipates this).

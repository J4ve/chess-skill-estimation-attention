# Real-Time Chess Skill Estimation and Anomaly Detection Using an Attention-Augmented CNN-BiLSTM

BS Computer Science thesis (Camarines Sur Polytechnic Colleges). This fork extends the CNN-BiLSTM rating-estimation baseline of Omori & Tadepalli (2024) with a deeper CNN, an attention mechanism, a move-level anomaly-detection module, and a real-time web prototype.

## Thesis abstract

This thesis develops a deep-learning system that estimates a chess player's skill move-by-move in real time from board states and clock times, and simultaneously flags moves that deviate suspiciously from the player's established level as possible engine assistance. The system extends the CNN-BiLSTM rating-estimation baseline of Omori and Tadepalli with a deeper convolutional network, an attention mechanism, and a move-level anomaly-detection module, and packages the result as a real-time web prototype for human fair-play review.

**Based on the baseline paper:** *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM* by Michael Omori and Prasad Tadepalli (Oregon State University). https://arxiv.org/abs/2409.11506

## Current build (2026-09-15)

This is Objective 3 of the thesis: a real-time, web-based prototype that
surfaces per-move ratings, suspicion scores, and critical moves to a human
fair-play reviewer, taking a PGN upload, a Lichess game identifier, a
built-in sample game, or a live ongoing Lichess game as input.

- **Model** (`src/chess_rating_net.py`, `src/attention.py`, `src/anomaly.py`):
  a CNN-BiLSTM rating estimator with causal-cumulative Bahdanau attention and
  an attention-weighted anomaly-detection branch. The API serves the frozen
  thesis checkpoint by default: the tuned-attention arm, test MAE 171.92 (see
  "Frozen weights" below).
- **FastAPI service** (`src/api.py`, `src/live.py`): PGN text, PGN upload,
  and Lichess game-ID endpoints; per-side baseline resolution with a
  reported source; a ranked `critical_moves` list for fast reviewer triage;
  clean 4xx errors instead of a bare 500 for bad input; and two
  Server-Sent-Events endpoints that follow a Lichess game (or Lichess TV)
  move by move (see "Live mode" below).
- **Web prototype** (`src/static/`): a no-build-step HTML/CSS/JS page served
  by the same FastAPI app, covering PGN, Lichess ID, sample games, and live
  input modes. Results render as a Lichess-style analysis board: a
  chessboard with player bars, a two-column move list, a per-move metrics
  panel, and a full-width rating chart synced to the current ply. See "Web
  prototype" below.

Manuscript and experimental plan: [J4ve/cs_thesis](https://github.com/J4ve/cs_thesis).

## Frozen weights

The API serves the frozen thesis checkpoint by default: `models/preflight_check_2m/best_model.pth`,
the tuned attention arm (test MAE 171.92). It is **not committed** to git (see
`.gitignore`).

- Expected location: `models/preflight_check_2m/best_model.pth`
- Override: set `RATINGNET_CHECKPOINT` to load an exact path instead.
- Fallback: if the frozen thesis checkpoint is not found and no override is
  set, the API falls back to Omori's released baseline checkpoint
  (`model_55.pth`, Option C "both" weight strategy) and logs a warning, since
  that checkpoint has no attention or anomaly branch.
  - Expected location: `models/model_55.pth`
  - Download: [Google Drive folder](https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt)

## Quick start (web prototype, local machine)

PyTorch needs Python 3.12 or 3.13. A venv outside the repo keeps its
interpreter symlinks clear of any repo sync tooling.

```bash
python3.12 -m venv ~/venvs/ratingnet-web
source ~/venvs/ratingnet-web/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Point at the frozen thesis checkpoint (read-only, never copied into the repo)
export RATINGNET_CHECKPOINT=/path/to/best_model.pth
# Or copy it into place instead of setting the env var:
#   mkdir -p models/preflight_check_2m
#   cp /path/to/best_model.pth models/preflight_check_2m/best_model.pth

python src/api.py
```

The service binds to `http://0.0.0.0:8000` by default (set `PORT` to
override). Open **http://localhost:8000/** in a browser for the web
prototype, or use the API directly:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict/pgn \
  -H "Content-Type: application/json" \
  -d '{"pgn": "[Event \"Demo\"]\n[WhiteElo \"1500\"]\n[BlackElo \"1500\"]\n[TimeControl \"300+0\"]\n\n1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} *"}'

curl -X POST http://localhost:8000/predict/upload -F "file=@game.pgn"

curl -X POST http://localhost:8000/predict/lichess \
  -H "Content-Type: application/json" \
  -d '{"game_id": "abcd1234"}'
```

## Web prototype

`src/static/` is a static, no-build-step page (plain HTML/CSS/JS) served by
FastAPI's `StaticFiles` at `/static`, with `GET /` returning `index.html`
directly.

The page has five input tabs: paste PGN, upload a `.pgn` file, enter a
Lichess game ID/URL, pick a **sample game** (see "Sample games" below), or
watch a game **live** (see "Live mode" below); plus optional per-side
baseline, critical-move top-k, and min-ply overrides shared across tabs.
Once a game is analyzed, the input form collapses (click its header to
reopen it for a new game) and a Lichess-style analysis board takes over: a
chessboard with a player bar above and below showing each side's baseline
and current rating estimate, a move list in Lichess's two-column style, a
per-move metrics panel, and a rating chart across the full width with a
marker that follows the current ply. See "Using the analysis board" below
for controls.

### Using the analysis board

**Board and player bars.** The bottom player bar always matches the board's
current orientation; the "Flip" button swaps orientation and the bars follow.
Each bar shows the player's baseline rating, the model's rating estimate at
the ply currently shown, and, when the source game recorded one, the
player's actual rating and the signed error between the estimate and that
actual rating (for example "estimate 1935, actual 1979, off by -44"). The
actual rating comes only from the PGN `WhiteElo`/`BlackElo` header and is
independent of any baseline override: overriding the baseline changes only
the suspicion-score comparison, never the actual-rating display. The results
header likewise shows each side's final error, and the metrics panel and
sample cards below show the same actual/error pair for the currently shown
ply and for the whole sample.

**Reveal/hide toggle.** The "Hide actual ratings" switch next to the results
header masks every actual-rating value and error across the bars, metrics
panel, chart, and sample card, showing a "Reveal" button in its place; this
supports a guess-the-rating demo where a reviewer steps through the game
watching only the live estimate before revealing the answer. Because the
baseline equals the actual rating in most games, hiding also masks baseline
numbers wherever they would give it away (shown as "baseline hidden"); the
suspicion-score bars and numbers stay visible either way, since they are not
themselves a rating value. The choice is remembered per browser via
`localStorage` (defaulting to shown) and does not affect what the API
returns, only what the page displays.

**What do these numbers mean?** Every label with a small "i" next to it
(baseline, actual rating, error, deviation, attention, critical move,
suspicion score, and more) shows a one-or-two-sentence plain-language
explanation on hover, tap, or keyboard focus. The same explanations are also
listed together in the collapsible "What do these numbers mean?" section
near the results header, for a reviewer who wants the full list at once. The
wording lives in one place, `METRIC_INFO` in `src/static/app.js`.

**Moves panel.** Lists every move in standard two-column notation (move
number, White, Black). Click a move to jump the board there. The move
currently shown is highlighted and kept scrolled into view. A red star marks
a critical move (see "Move details" below).

**Rating chart.** Full width, below the board and move list, like Lichess's
evaluation graph. A vertical line and a dot on each curve track the current
ply; small ring markers show critical moves. Click or drag on the chart to
jump the board to that ply. Each side's baseline is drawn as a dashed
reference line; if that side's actual rating is also known and matches the
baseline (the usual case), the two are drawn as one line labeled "actual
rating (baseline)" rather than as duplicates, and only when they differ does
a separate, visually distinct actual-rating line appear. While actual
ratings are hidden, these reference lines are omitted from the chart
entirely rather than drawn with masked values, so the rating-estimate curves
are all that remain visible.

**Controls and keys.**

| Control | Key | Effect |
| --- | --- | --- |
| `«` | `Home` / `ArrowUp` | Jump to the start of the game |
| `‹` | `ArrowLeft` | Step back one ply |
| `›` | `ArrowRight` | Step forward one ply |
| `»` | `End` / `ArrowDown` | Jump to the end of the game |
| Play/Pause | - | Autoplay forward at about one move per second; stops at the end or on any manual navigation |
| Flip | - | Swap board orientation and the player bars |

Keyboard shortcuts are ignored while typing in a text field.

**Move details panel.** For the currently shown ply: the move in SAN, each
side's rating estimate and its change from the previous ply, each side's
deviation from its baseline, the move's attention weight with its rank and
percentile among all plies in the game (for example "top 5% of attention"),
and whether the move is flagged critical with its rank. At the start of the
game this shows each side's baseline instead.

**Critical moves.** Ranked per side by `attention_weight * |deviation|`,
excluding plies before a minimum ply (default 10, configurable per request;
see "API reference") since the model has little context in the opening and
early plies would otherwise dominate the list.

### Suspicion labels

Next to each suspicion bar, a label chip classifies that side's S_att against
percentile cutoffs computed from ordinary, finished, rated games in the
thesis's held-out TEST-partition corpus (games the model never trained on):
**Typical** (below the 75th percentile of those games), **Unusual** (75th to
95th percentile), or **Highly unusual: worth a human review** (above the
95th percentile). Small ticks on the bar track mark the p75 and p95 cutoff
positions so the bar reads visually, and a caption under the bars states the
comparison is against ordinary games, "not evidence of engine use on its
own." Cutoffs use the game's own time-control bucket (bullet/blitz/rapid/
classical/ultrabullet, derived from the PGN `TimeControl` header) when that
bucket has at least ~300 sides scored; otherwise they fall back to the
overall cutoff across all time controls. This is the same wording, cutoffs
file shape, and boundary logic pytest covers in `tests/test_suspicion_labels.py`
and `tests/test_format_data.py`.

The cutoffs live in `src/static/suspicion_cutoffs.json`, computed on the HPC
corpus checkout by scoring a stratified sample of held-out test games with
this exact model + `AnomalyDetector` code path (same checkpoint, same
baseline resolution) so the numbers are directly comparable to what a viewer
sees here. As of this writing the file is **provisional**: the full
~2,718-game stratified sample run was stopped early, so it currently reflects
1,200 scored games (bullet and blitz fully covered; rapid partial; classical
and ultrabullet not yet scored, both currently falling back to the overall
cutoff). The app shows a banner under the suspicion bars whenever the served
cutoffs are provisional, naming how many games back them; the banner
disappears once the file is regenerated from the completed run. Method,
the full percentile/CI table, and a sanity check running the 12 bundled
sample games through this same pipeline are recorded outside this repo in
the cutoffs report generated alongside this file.

Labels never claim cheating and never say "likely" or name a player as
suspicious: S_att separated engine-substituted synthetic games only weakly in
thesis evaluation (ROC-AUC 0.555, see "Suspicion score" above), and a clean
synthetic sample game (`synthetic: false alarm`, see "Sample games" below)
lands solidly in "Highly unusual" despite having no engine-substituted moves
at all. A label describes how uncommon a score is among ordinary games, not
a verdict about the players.

The API optionally returns `white_suspicion_label`/`black_suspicion_label`
and the cutoffs actually applied (see "API reference"), so the labelling
logic is testable independent of the UI. `suspicion_labels.py` takes a score
and a cutoffs object with no reference to S_att specifically, so a future
trained detector can reuse it with its own cutoffs file in the same shape.

### Sample games

The "Sample games" tab loads and analyzes a game in one click, reusing the
same analysis board as every other input mode. Cards are grouped by
`src/static/samples/manifest.json`'s `group` field:

- **Rating: Bullet / Blitz / Rapid** - for each time control, a
  best-predicted, typical, and worst-predicted game, picked by this
  checkpoint's held-out test error (`analysis/heldout_test_eval/`), not by
  outcome or player identity. This is an honest spread, not a highlight
  reel: the worst-predicted games are shown because the model does worse on
  them, most often master-strength games underrepresented in the training
  distribution or very short/aborted games. Descriptions only ever
  characterize level and time control; they never call out a specific
  player. Source games are real, finished, public Lichess games (CC0, no
  rights reserved), so usernames in the PGN headers are real and unchanged,
  but nothing in this tab or its descriptions labels any player's games as
  suspicious.
- **Synthetic anomaly: caught / false alarm / missed** - three games from
  this project's own synthetic anomaly corpus (a Maia policy network's
  moves with some fraction substituted for a stronger engine's,
  `src/generate_anomaly_corpus.py`), one from each outcome bucket at the
  suspicion-score threshold used in thesis evaluation. These games are
  synthetic end to end (no real player), and each substituted ply is known
  ground truth: it renders as a small square marker on the move list and
  the rating chart (next to the existing critical-move star/ring markers),
  and the panel above the board shows the substitution rate, the Maia
  rating band, the substitution engine, and this game's saved suspicion
  score (`S_att`) from thesis evaluation, next to the score this deployment
  computes live, so the two can be compared directly. A neutral note on
  every synthetic card explains why both sides can score high here: the
  rating model reads Maia's play as coming from a player well above its
  nominal band (for example Maia 1100 read as roughly 2280), which is the
  known reason the computed suspicion score is weak on these samples.

Every card also shows its saved held-out test error (rating games) or saved
`S_att` (synthetic games) alongside the app's own live estimate, so a viewer
can see whether this deployment's numbers match the thesis evaluation's, and
the actual rating(s) used for evaluation. The "Hide actual ratings" toggle
(see "Using the analysis board") masks the actual-rating and test-error
facts on this card the same way it masks the board and chart.

`manifest.json` documents its own schema in a `schema_notes` field; new
entries only need `id`, `title`, `description`, and `pgn_path`, with every
provenance field (`source`, `selection_label`, test errors, actual ratings,
`substituted_plies`, and the synthetic-only fields) optional.

### Live mode

The "Live" tab follows an ongoing Lichess game move by move: paste a game ID
or URL and click "Watch game", or click "Watch Lichess TV" to follow
whichever game Lichess is currently featuring (switching automatically when
TV switches games). Both use Server-Sent Events against this app's own
`GET /live/stream/{game_id}` and `GET /live/tv` endpoints, which in turn
follow Lichess's public streaming API (`GET /api/stream/game/{id}`,
`GET /api/tv/feed`) with no token required. Each update carries
`white_actual_rating`/`black_actual_rating` straight from whatever rating
Lichess's own stream reports for that side (the game's PGN header when
following a game by ID, or the featured player's live rating when following
TV), null when the stream did not report one, and displayed and masked by
the reveal/hide toggle exactly like a batch result.

**Prefix-estimate semantics (read this before trusting the live curve).**
The model's BiLSTM is not incremental: scoring ply *t* means running the
whole model over plies 1..*t* from scratch (see "Limitations"). Live mode
makes this workable by re-running that same batch pipeline on the growing
PGN prefix every time a new move arrives, and takes only the estimate at
the *final* ply of that run. The chart plots and freezes that one value per
move; earlier points are never revised, even though a from-scratch rerun
over a longer prefix would compute slightly different values for them. This
is why the live curve is labeled **"live (move-by-move) estimate"** and can
differ from the full-game curve you'd get by re-analyzing the finished game
normally - the full-game curve is the non-causal, most-accurate view; the
live curve is what the model could have told you in the moment. Suspicion
score and critical moves are likewise computed on the prefix so far and
marked provisional, and can shift as the game continues. Per-move inference
time on this deployment's CPU is shown in the status line after each move.

Once the game ends, a **"Show full-game analysis"** button appears and
switches to the normal, non-live analysis view for the finished game.

**Delay.** Lichess itself delays a spectator's view of an ongoing game by a
few moves as an anti-cheating measure; the status line says so once
connected. This is a Lichess-side delay, not something this app can reduce.

**Following behavior.** The board auto-follows the newest move while you're
viewing the latest ply. Stepping back (via the board controls, keyboard, or
the chart) stops following and shows a "Jump to live" button; new moves
still arrive and grow the chart, but the board stays where you left it
until you jump back. Stopping, or switching to a different game or tab,
cleanly closes the stream. On a dropped connection to this app's own SSE
endpoint the page reconnects once automatically, then shows a persistent
error if that also fails; a dropped upstream connection to Lichess is
retried once server-side, transparently, before that same error surfaces.

### Roadmap (not built)

- Optional Stockfish evaluation plotted alongside the rating curve, to give a
  reviewer both signals (engine correlation and rating-estimate anomaly) on
  the same timeline.

### Vendored libraries

No CDN is used, so the page works offline at the defense. Each library is
vendored under `src/static/vendor/` with its own license file:

| Library | Version | Source | License |
| --- | --- | --- | --- |
| jQuery | 3.7.1 | https://www.npmjs.com/package/jquery | MIT |
| chessboard.js | 1.0.0 | https://github.com/oakmac/chessboardjs (npm: `@chrisoakman/chessboardjs`) | MIT |
| chessboard.js piece images | v1.0.0 tag | https://github.com/oakmac/chessboardjs/tree/v1.0.0/website/img/chesspieces/wikipedia | MIT |
| chess.js | 0.13.4 | https://www.npmjs.com/package/chess.js | BSD-2-Clause |
| Chart.js | 4.4.4 | https://www.npmjs.com/package/chart.js | MIT |

chess.js 0.13.4 is the last version published as a plain ES module with no
bundler required (later versions still ship as ESM; this version was picked
because it is a small, stable release with no build step needed beyond a
native `<script type="module">` import). chessboard.js needs jQuery as a peer
dependency; both load as classic scripts before `app.js`, which imports
chess.js as a native ES module.

## API reference

`GET /api` returns this list as JSON at runtime.

| Endpoint | Description |
| --- | --- |
| `GET /health` | Model load status, device, and checkpoint path. |
| `GET /` | The web prototype (`src/static/index.html`). |
| `GET /api` | Machine-readable endpoint list. |
| `POST /predict/pgn` | Body `{pgn, white_baseline?, black_baseline?}`. Query `top_k` (default 5), `min_ply` (default 10). |
| `POST /predict/upload` | Multipart form: `file` (`.pgn`), `white_baseline?`, `black_baseline?`. Query `top_k`, `min_ply`. |
| `POST /predict/lichess` | Body `{game_id, white_baseline?, black_baseline?}`, where `game_id` is a bare 8-character Lichess ID or a full game URL. Query `top_k`, `min_ply`. |
| `GET /live/stream/{game_id}` | Server-Sent Events. Follows one Lichess game (ongoing or just-finished) move by move. Query `top_k`, `min_ply`, `white_baseline?`, `black_baseline?`. See "Live mode". |
| `GET /live/tv` | Server-Sent Events. Follows Lichess TV's currently featured game, switching automatically when TV switches games. Query `top_k`, `min_ply`. See "Live mode". |

All three `predict/*` endpoints return the same shape:

- `headers`: the PGN headers (`White`, `Black`, `WhiteElo`, `Result`, ...).
- `white_baseline`, `black_baseline`: the resolved per-side baseline rating.
- `white_baseline_source`, `black_baseline_source`: one of `request` (caller
  supplied it), `pgn_header` (parsed from `WhiteElo`/`BlackElo`), or
  `self_prediction_fallback` (no usable baseline anywhere, so the model's own
  final-ply prediction was used instead; see "Limitations").
- `white_actual_rating`, `black_actual_rating`: the side's real rating,
  parsed from the PGN `WhiteElo`/`BlackElo` header only (`"?"` and any other
  non-numeric or missing value is `null`). Independent of the baseline: a
  `white_baseline`/`black_baseline` request override never changes this
  field, since it is a fixed ground-truth fact about the game rather than a
  reviewer-supplied value used for suspicion scoring.
- `warnings`: readable strings for anything that degrades the result (a
  fallback baseline, or an anomaly branch that is not available).
- `white_final_rating`, `black_final_rating`: the model's final-ply rating
  estimate per side.
- `white_suspicion_score`, `black_suspicion_score`, `combined_suspicion_score`:
  attention-weighted deviation from baseline, summed over the game.
- `white_suspicion_label`, `black_suspicion_label`: one of `typical`,
  `unusual`, or `highly_unusual`, from comparing that side's suspicion score
  against percentile cutoffs computed on ordinary held-out test games (see
  "Suspicion labels" below). `null` when the served checkpoint has no anomaly
  branch or no cutoffs file has been generated.
- `suspicion_cutoffs_used`: the `p75`/`p95` cutoff values actually applied
  (`source` is `time_control` or `overall`, `time_control` is the bucket
  derived from the PGN `TimeControl` header, or `null` if it couldn't be
  parsed). `null` under the same conditions as the labels above.
- `per_move`: one record per ply (`ply`, `move` as SAN, `uci`, `white_rating`,
  `black_rating`, `attention_weight`, `white_deviation`, `black_deviation`,
  `clock_seconds`, `time_spent_seconds`). `clock_seconds` is that ply's mover's
  remaining clock, parsed from the PGN's `[%clk ...]` comments (one of the
  model's own inputs, not a derived value). `time_spent_seconds` is that
  side's previous remaining clock minus `clock_seconds` plus any `TimeControl`
  increment; `null` for a side's first move when `TimeControl` couldn't be
  parsed (see `format_data.compute_time_spent`).
- `critical_moves`: up to `top_k` entries per side, ranked by
  `attention_weight * |deviation|` for that side (`ply`, `move`, `side`,
  `attention_weight`, `deviation`, `weighted_score`), excluding plies before
  `min_ply`. Empty (with a warning) when the served checkpoint has no
  attention/anomaly branch.
- `critical_moves_min_ply`: the `min_ply` value actually used for the
  `critical_moves` ranking above.
- `ongoing`, `provisional`: `true` when the game has no final result yet
  (PGN `Result` header is `*`).

Bad input (a PGN with no `[%clk ...]` annotations, unparseable PGN text, or a
malformed Lichess ID) returns HTTP 422 with a readable `detail`, never a bare
500. A Lichess game that does not exist returns 404; hitting Lichess's rate
limit returns 429.

The two `/live/*` endpoints stream `text/event-stream`; each `data:` line is
one JSON object with a `type`:

- `{"type": "status", "state": ..., "message": ...}` - a status update.
  `state` is one of `connecting`, `connected`, `reconnecting`, `capped`
  (the game passed 100 plies, the model's training cap), `finished`, or
  `error`.
- `{"type": "update", "result": {...}}` - `result` has the same shape as
  the `predict/*` endpoints above, plus `inference_ms` (this move's
  inference wall time), `lichess_game_id`, and `live: true`. `per_move`
  here covers only the plies known so far; a client that wants a frozen,
  never-revised chart should keep only the newest row of each update and
  append it to its own array, as `src/static/app.js`'s `mergeLiveUpdate`
  does, rather than replacing its data with `per_move` wholesale.
- `{"type": "error", "detail": ...}` - a terminal error; the stream closes
  after this.

## Used as a submodule

This repository is the `prototype/` submodule of
[J4ve/cs_thesis](https://github.com/J4ve/cs_thesis), the manuscript and
experiment-plan repository. Code changes land here first; the thesis repo
then bumps its submodule pointer to pick them up. Do not edit thesis-repo
files from within this repository's history.

## Limitations

- **Suspicion score is a supplementary flag, not a verdict.** It is meant to
  help a human fair-play reviewer decide where to look, not to accuse a
  player automatically. See Barnes and Hernandez-Castro (2015) on the
  false-positive risk of single-game move analysis.
- **Mid-game values are provisional.** For an ongoing game (`Result` header
  `*`), Lichess itself delays the export by a few moves, and any suspicion
  score computed before the game ends should be read as provisional: it can
  shift once more moves are known.
- **Every move re-runs the full bidirectional model over the whole prefix.**
  The model's BiLSTM has a backward pass that needs a completed sequence;
  there is no incremental/streaming inference here; scoring ply *t* means
  running the model over plies 1..*t* from scratch. This is fine for the
  batch PGN/Lichess-ID flows in this prototype. Live mode (see "Live mode")
  works around it by re-running the batch pipeline on the growing prefix
  and taking only the final-ply estimate each time, so the live curve is a
  sequence of independent from-scratch runs, not a true incremental
  inference; it can differ from the full-game curve for the same finished
  game.
- **Self-prediction fallback baselines are close to meaningless.** If neither
  the caller nor the PGN headers supply a baseline rating, the suspicion
  score compares the model's own final-ply prediction against itself. The
  API always reports which baseline source was used
  (`white_baseline_source`/`black_baseline_source`) so this is visible, not
  silent.

## Training path (optional)

```bash
python src/chess_rating_net.py --train --data_dir data/processed_games \
  --experiment cnn_bilstm_clocks_all --epochs 60 --lr 1e-4 --batch_size 32 \
  --model_dir models --resume models/cnn_bilstm_clocks_all/latest.pth
```

A YAML config is available at `example_config.yaml`.

---

## Upstream baseline (original README)

**Authors**: Michael Omori (ORCID: [0009-0000-4632-9272](https://orcid.org/0009-0000-4632-9272)), Prasad Tadepalli (ORCID: [0000-0003-2736-3912](https://orcid.org/0000-0003-2736-3912))
Oregon State University, Corvallis OR, USA

### Abstract

Current chess rating systems update ratings incrementally and may not always accurately reflect a player's true strength at all times, especially for rapidly improving players or very rusty players. To overcome this, we explore a method to estimate player ratings directly from game moves and clock times. We compiled a benchmark dataset from Lichess with over one million games, encompassing various time controls and including move sequences and clock times. Our model architecture comprises a CNN to learn positional features, which are then integrated with clock-time data into a Bidirectional LSTM, predicting player ratings after each move. The model achieved an MAE of 182 rating points on the test data. Additionally, we applied our model to the 2024 IEEE Big Data Cup Chess Puzzle Difficulty Competition dataset, predicted puzzle ratings and achieved competitive results. This model is the first to use no hand-crafted features to estimate chess ratings and also the first to output a rating prediction after each move. Our method highlights the potential of using move-based rating estimation for enhancing rating systems and potentially other applications such as cheating detection.

### Installation and Setup

```bash
conda create --name rating_env python=3.8
conda activate rating_env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn==1.3.2
conda install tensorboard
pip install chess==1.10.0
```

You can download games from https://database.lichess.org/ in the .pgn.zst format.
Put them in data/game_zips.
Next run
```bash
sh format.sh
```
You can change the year and months in that file. This will run src/format_data_legacy.py (the original eval-annotation-requiring preprocessor kept alongside the current src/format_data.py, which the rest of this repo now uses) which converts the games into a format suitable for the cnn input.
The converted game data will be saved in data/processed_games.
Download model_55.pth and put it in models/cnn_bilstm_clocks_all.
We also provide a direct download from google drive at this link: https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt?usp=sharing
You can run the code with
```bash
python src/chess_rating_net.py
```
python src/game_analysis.py will output analyzed games with the rating predictions.

## License

This project is licensed under the MIT license - see LICENSE.

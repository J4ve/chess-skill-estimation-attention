# Real-Time Chess Skill Estimation and Anomaly Detection Using an Attention-Augmented CNN-BiLSTM

BS Computer Science thesis (Camarines Sur Polytechnic Colleges). This fork extends the CNN-BiLSTM rating-estimation baseline of Omori & Tadepalli (2024) with a deeper CNN, an attention mechanism, a move-level anomaly-detection module, and a real-time web prototype.

## Thesis abstract

This thesis develops a deep-learning system that estimates a chess player's skill move-by-move in real time from board states and clock times, and simultaneously flags moves that deviate suspiciously from the player's established level as possible engine assistance. The system extends the CNN-BiLSTM rating-estimation baseline of Omori and Tadepalli with a deeper convolutional network, an attention mechanism, and a move-level anomaly-detection module, and packages the result as a real-time web prototype for human fair-play review.

**Based on the baseline paper:** *Chess Rating Estimation from Moves and Clock Times Using a CNN-LSTM* by Michael Omori and Prasad Tadepalli (Oregon State University). https://arxiv.org/abs/2409.11506

## Current build (2026-09-15)

This is Objective 3 of the thesis: a real-time, web-based prototype that
surfaces per-move ratings, suspicion scores, and critical moves to a human
fair-play reviewer, taking a PGN upload or a Lichess game identifier as
input.

- **Model** (`src/chess_rating_net.py`, `src/attention.py`, `src/anomaly.py`):
  a CNN-BiLSTM rating estimator with causal-cumulative Bahdanau attention and
  an attention-weighted anomaly-detection branch. The API serves the frozen
  thesis checkpoint by default: the tuned-attention arm, test MAE 171.92 (see
  "Frozen weights" below).
- **FastAPI service** (`src/api.py`): PGN text, PGN upload, and Lichess
  game-ID endpoints; per-side baseline resolution with a reported source;
  a ranked `critical_moves` list for fast reviewer triage; clean 4xx errors
  instead of a bare 500 for bad input.
- **Web prototype** (`src/static/`): a no-build-step HTML/CSS/JS page served
  by the same FastAPI app, covering both required input modes (PGN and
  Lichess ID). Results render as a Lichess-style analysis board: a
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
directly. It covers the two input modes committed for this stage of the
thesis (PGN upload/paste and a Lichess game identifier); move-by-move live
streaming of an ongoing game is a later step, not implemented here (see
"Limitations").

The page has three input tabs (paste PGN, upload a `.pgn` file, or enter a
Lichess game ID/URL), optional per-side baseline, critical-move top-k, and
min-ply overrides. Once a game is analyzed, the input form collapses (click
its header to reopen it for a new game) and a Lichess-style analysis board
takes over: a chessboard with a player bar above and below showing each
side's baseline and current rating estimate, a move list in Lichess's
two-column style, a per-move metrics panel, and a rating chart across the
full width with a marker that follows the current ply. See "Using the
analysis board" below for controls.

### Using the analysis board

**Board and player bars.** The bottom player bar always matches the board's
current orientation; the "Flip" button swaps orientation and the bars follow.
Each bar shows the player's baseline rating and the model's rating estimate
at the ply currently shown.

**Moves panel.** Lists every move in standard two-column notation (move
number, White, Black). Click a move to jump the board there. The move
currently shown is highlighted and kept scrolled into view. A red star marks
a critical move (see "Move details" below).

**Rating chart.** Full width, below the board and move list, like Lichess's
evaluation graph. A vertical line and a dot on each curve track the current
ply; small ring markers show critical moves. Click or drag on the chart to
jump the board to that ply.

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

### Roadmap (not built)

- Live game streaming from Lichess, so an ongoing game updates move by move
  instead of requiring a re-submitted PGN. See "Limitations" below for why
  this is not a small addition (the model has no incremental/streaming mode).
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

All three `predict/*` endpoints return the same shape:

- `headers`: the PGN headers (`White`, `Black`, `WhiteElo`, `Result`, ...).
- `white_baseline`, `black_baseline`: the resolved per-side baseline rating.
- `white_baseline_source`, `black_baseline_source`: one of `request` (caller
  supplied it), `pgn_header` (parsed from `WhiteElo`/`BlackElo`), or
  `self_prediction_fallback` (no usable baseline anywhere, so the model's own
  final-ply prediction was used instead; see "Limitations").
- `warnings`: readable strings for anything that degrades the result (a
  fallback baseline, or an anomaly branch that is not available).
- `white_final_rating`, `black_final_rating`: the model's final-ply rating
  estimate per side.
- `white_suspicion_score`, `black_suspicion_score`, `combined_suspicion_score`:
  attention-weighted deviation from baseline, summed over the game.
- `per_move`: one record per ply (`ply`, `move` as SAN, `uci`, `white_rating`,
  `black_rating`, `attention_weight`, `white_deviation`, `black_deviation`).
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
  batch PGN/Lichess-ID flows in this prototype, but is why true move-by-move
  live streaming of an ongoing game is out of scope for this stage (see
  "Web prototype").
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

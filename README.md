# Real-Time Chess Skill Estimation and Anomaly Detection Using an Attention-Augmented CNN-BiLSTM

BS Computer Science thesis (Camarines Sur Polytechnic Colleges). This fork
extends the CNN-BiLSTM rating-estimation baseline of Omori & Tadepalli (2024)
with a deeper CNN, an attention mechanism, a move-level anomaly-detection
module, and a real-time web prototype for human fair-play review.

The prototype estimates each player's rating move by move from board states
and clock times, and gives each side a suspicion score, from any of four
methods the thesis tried, as a pointer for a human reviewer. It takes a PGN, a
Lichess game ID, a built-in sample game, or a live ongoing Lichess game.

**Based on the baseline paper:** *Chess Rating Estimation from Moves and Clock
Times Using a CNN-LSTM* by Michael Omori and Prasad Tadepalli (Oregon State
University). https://arxiv.org/abs/2409.11506
Manuscript and experimental plan: [J4ve/cs_thesis](https://github.com/J4ve/cs_thesis).

## The app

Following a live Lichess TV game, switching suspicion methods mid-game:

![Demo: following a live game, switching suspicion methods](docs/screenshots/demo.gif)

![Analysis view, light theme](docs/screenshots/analysis-light.png)

The same view in dark theme, and the sample-game picker:

| Dark theme | Sample games |
| --- | --- |
| ![Analysis view, dark theme](docs/screenshots/analysis-dark.png) | ![Sample game picker](docs/screenshots/samples-tab.png) |

## What it does

- **Rating estimate per move** (`src/chess_rating_net.py`, `src/attention.py`):
  a CNN-BiLSTM with causal-cumulative Bahdanau attention. The API serves the
  frozen thesis checkpoint: the tuned-attention arm, test MAE 171.92.
- **Suspicion score per side, four selectable methods** (weights in
  `src/models/`). A Method dropdown on the suspicion card switches between
  them for the loaded game without re-running the rating model, and remembers
  the choice. ROC-AUC on synthetic games from rating bands withheld from
  training (from `src/static/suspicion_methods.json`):

  | Method (page label) | Code, thesis arm | Withheld bands | 2% engine moves | 60% engine moves |
  | --- | --- | --- | --- | --- |
  | Computed score (first method tried) | `src/anomaly.py`, S_att | 0.506 | 0.497 | 0.520 |
  | Trained detector (LightGBM) | `src/lgbm_detector.py`, A0g | 0.688 | 0.539 | 0.811 |
  | **Per-move detector (A3g, default)** | `src/detector.py`, A3g | **0.752** | 0.576 | 0.892 |
  | Full model (CNN-BiLSTM) | `src/cnn_bilstm_detector.py`, A4 | 0.705 | 0.547 | 0.838 |

  ![Suspicion card](docs/screenshots/suspicion-card.png)

  The Method dropdown open, showing all four choices:

  ![Suspicion method dropdown open](docs/screenshots/method-selector.png)

- **Typical / Unusual / Highly unusual labels**, from each method's own
  percentile cutoffs over the same 2,822 ordinary held-out test games, per
  time control. A label says how uncommon a score is among ordinary games,
  never that anyone cheated.
- **Critical moves**: per side, the moves ranked highest by attention times
  rating deviation, so a reviewer knows where to look first. This ranking
  does not change with the selected method.
- **Live mode**: follows an ongoing Lichess game (or Lichess TV) move by move
  over Server-Sent Events.
- **Sample games**: held-out test games across time controls, plus synthetic
  caught / false-alarm / missed examples, in one click.

## Quick start (web prototype, local machine)

PyTorch needs Python 3.12 or 3.13. A venv outside the repo keeps its
interpreter symlinks clear of any repo sync tooling.

> `RATINGNET_CHECKPOINT` points at the frozen thesis checkpoint on disk; it
> is read-only and never copied into the repo. See [Weights](#weights) below
> for the alternative of placing the file directly under `models/`.

```bash
python3.12 -m venv ~/venvs/ratingnet-web
source ~/venvs/ratingnet-web/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export RATINGNET_CHECKPOINT=/path/to/best_model.pth

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

## Weights

The rating checkpoint is **not committed** to git. Download
`attn_tuned_best.pth` from the release below, then either set
`RATINGNET_CHECKPOINT` to its path or place it where the API looks for it by
default:

```bash
mkdir -p models/preflight_check_2m
curl -L -o models/preflight_check_2m/best_model.pth \
  https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/attn_tuned_best.pth
sha256sum models/preflight_check_2m/best_model.pth
# ff6370e477a0ea7264933adbf517f94068ed1205126ad284d8158ae6124275a0
```

The trained suspicion detectors (per-move, LightGBM, CNN-BiLSTM head) are
small and **are** committed, with their provenance, at `src/models/`. See
[docs/development.md](docs/development.md) for the fallback checkpoint,
submodule use, and the training path.

## Published artifacts

The frozen rating checkpoint and the computations behind the reported
evaluation numbers are published as a GitHub release,
[v0.1-weights-attn-tuned](https://github.com/J4ve/chess-skill-estimation-attention/releases/tag/v0.1-weights-attn-tuned).
The analysis scripts that produced them are in this repository at
[analysis/](analysis/) rather than in the release, so they can be browsed and
diffed.

### Two names that mislead, read this before loading anything

> **`models/preflight_check_2m/best_model.pth` is the tuned attention arm.**
> The directory name says "preflight check" but that experiment is the frozen
> reported architecture: learning rate 3e-4, Bahdanau attention with
> `attention_dim` 64, trained on the full 2,550,000 game corpus. Its identity
> in the evaluation records is `attn_tuned`, and the release publishes the same
> bytes under the unambiguous name `attn_tuned_best.pth`.

> **`models/model_55.pth` is not one of the thesis arms.** It is a plain
> baseline checkpoint that predates the corrections this study applied: its
> stored `params` carry no `use_attention` and no `split_seed`, and its
> `epochs` and `val_batch_size` are pre-fix defaults. `src/api.py` still names
> it as a last-resort fallback when no thesis checkpoint is found, which is a
> known defect and not an endorsement. It is deliberately **not** published,
> because a reader who scored games with it would get numbers that match no
> reported result.

### Assets

Every asset carries its sha256 below, and `SHA256SUMS.txt` in the release
repeats them in `sha256sum -c` format, covering both the downloads and the
files inside the two archives as they extract.

| Asset | What it is | sha256 |
| --- | --- | --- |
| [`attn_tuned_best.pth`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/attn_tuned_best.pth) | The `attn_tuned` rating checkpoint, 9,379,970 bytes, best validation epoch 58 of a 60 epoch run | `ff6370e477a0ea7264933adbf517f94068ed1205126ad284d8158ae6124275a0` |
| [`heldout_test_per_game_errors.tar.gz`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/heldout_test_per_game_errors.tar.gz) | Per game absolute and signed errors over the 255,000 game held out test partition, one CSV per arm for five arms, about 95 MB unpacked | `a87293fa72cd5620953869eefdb3e52198e3dd19930d0b93c0e144d0835ae919` |
| [`eval_records.tar.gz`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/eval_records.tar.gz) | The held out evaluation JSONs, the paired bootstrap output, corpus composition, the anomaly detector arm results, and the deployment latency and agreement measurements | `68a6d2c9413b416bc5b01067c7c853c85912c9f3513cbe073be5935c839cbc4a` |
| [`attn_tuned__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/attn_tuned__best.json) | `attn_tuned` evaluation record and `arch_params`, loose so it reads without the archive | `c7e086259b7fafb83c8bebc8cb68d57388a24982eac781259e9e3fc18f90b66c` |
| [`attn_untuned__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/attn_untuned__best.json) | `attn_untuned` evaluation record and `arch_params` | `85a0fd424ddac37106a07d3ed2321ce4d2a08bda206049ebec9430d6f81212f1` |
| [`baseline__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/baseline__best.json) | `baseline` evaluation record and `arch_params` | `f233636fabbfef3eef27bee18e92fa6d899da4894eaeea1f384afd53fc05f5ba` |
| [`baseline_lr3e4__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/baseline_lr3e4__best.json) | `baseline_lr3e4` evaluation record and `arch_params` | `e47674dd866fca2723f77045aaf420ad2ea33a81f992fa66d70e7682b96d2f38` |
| [`deepcnn__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/deepcnn__best.json) | `deepcnn` evaluation record and `arch_params` | `2cd4697aecb9afb0f7ef867d28045bdc8079d932f6caa8e89776d3487d00e2ed` |
| [`lowdropout__best.json`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/lowdropout__best.json) | `lowdropout` evaluation record and `arch_params` | `06778437d4c9580792d1c00626c1689f737db82e050bfe1bfd2f1bc21f163693` |
| [`SHA256SUMS.txt`](https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/SHA256SUMS.txt) | Checksums for every asset above and for every file inside the two archives | listed in the release notes |

### Seeds

The seeds are recorded in every `arch_params` block and are repeated here
because nothing else in this repository states them:

- **Training seed 0** for every full corpus arm. It is passed as `--seed`.
- **Data split seed 42**, the `--split_seed` that fixes the 72 / 18 / 10
  train, validation and test partition of the corpus. Where the thesis says
  "seed 42" it means this split seed, never a training seed.
- **Seeds 0 through 4** for the separate 170,000 game seed variance study,
  which trained five baseline and five attention runs to check whether the
  attention difference survived seed noise.

### Reproducing a reported number

The checkpoint is a dictionary, not a bare state dict. Its architecture lives
under `params`, which is what `src/api.py`'s own `_load_model()` reads, so a
reader never has to guess how it was configured:

```python
import torch
from chess_rating_net import ChessEloPredictor   # run with src/ on sys.path

ckpt = torch.load("attn_tuned_best.pth", map_location="cpu", weights_only=False)
p = ckpt["params"]
print(p["use_attention"], p["attention_type"], p["attention_dim"], p["learning_rate"])
print(p["seed"], p["split_seed"], ckpt["best_epoch"], ckpt["best_val_loss"])

model = ChessEloPredictor(
    conv_filters=p["conv_filters"], lstm_layers=p["lstm_layers"],
    dropout_rate=p["dropout_rate"], lstm_h=p["lstm_h"], fc1_h=p["fc1_h"],
    bidirectional=p["bidirectional"], use_attention=p["use_attention"],
    attention_type=p["attention_type"], attention_dim=p["attention_dim"],
    use_anomaly=p["use_attention"],
)
model.load_base_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
```

That prints `True bahdanau 64 0.0003` and `0 42 58 172.3787906438345`, which
is how the published file identifies itself as the `attn_tuned` arm without
trusting its directory name.

Predictions are produced in normalized units and multiplied back by
`ratings_std` 366 and shifted by `ratings_mean` 1514, the constants the
baseline paper used. They are stored in the checkpoint and in every
evaluation record, so a reader never has to guess them.

The per game CSVs let the headline test MAE of any published arm be
recomputed without a GPU or the corpus. Each row is one held out game, with
`white_err` and `black_err` in rating points, and the arm's MAE is the mean
over both columns:

```python
import csv, statistics
errs = []
for row in csv.DictReader(open("heldout_test_per_game_errors/attn_tuned__best.csv")):
    errs += [float(row["white_err"]), float(row["black_err"])]
print(len(errs) // 2, statistics.fmean(errs))
```

That prints `255000 171.9167760980392`, which is the
`overall_mae_rating_points` recorded in `attn_tuned__best.json` down to the
order in which the terms are summed. The rows also carry `time_control`,
`white_elo` and `black_elo`, so per time control and per rating band
breakdowns follow from the same files, as does a paired bootstrap between two
arms: `analysis/paired_bootstrap_heldout.py` is this study's paired bootstrap
over these CSVs, and `bootstrap.json` inside `eval_records.tar.gz` records the
resample count, the seed and the per game error definition it used.

### This release is partial

Only the `attn_tuned` checkpoint is published. The other five arms the study
trained are on the institutional cluster and have not been transferred yet:

| Arm | Checkpoint on the cluster | Published here |
| --- | --- | --- |
| `attn_tuned` | `models/preflight_check_2m/best_model.pth` | weights, per game errors, evaluation record |
| `attn_untuned` | `models/fullcorpus_attention_untuned_arm2/best_model.pth` | per game errors, evaluation record |
| `baseline` | `models/fullcorpus_baseline_arm1/best_model.pth` | per game errors, evaluation record |
| `baseline_lr3e4` | `models/fullcorpus_baseline_lr3e4_arm6/best_model.pth` | per game errors, evaluation record |
| `deepcnn` | `models/fullcorpus_deepcnn_arm5/best_model.pth` | per game errors, evaluation record |
| `lowdropout` | `models/diag_fullcorpus_lowdropout/best_model.pth` | evaluation record only |

`lowdropout` is the one arm with no per game CSV, so five CSVs cover six
arms. A later release will add the missing weights.

## Documentation

- [docs/web-prototype.md](docs/web-prototype.md) - the page itself: layout,
  analysis board, suspicion score and labels, sample games, live mode,
  roadmap, vendored libraries.
- [docs/api.md](docs/api.md) - endpoints, every response field, and error
  behaviour.
- [docs/development.md](docs/development.md) - weights, submodule use,
  training path, and the upstream baseline's own README.
- [analysis/](analysis/) - the study's own evaluation, detector and corpus
  scripts, with a guide to which one produced which published record.
- [experiments/detector_parity/](experiments/detector_parity/) - the app's
  per-move detector path reproduces the thesis HPC code to within 6e-07.
- [experiments/method_parity/](experiments/method_parity/) - the same check
  for the LightGBM (4.8e-07) and CNN-BiLSTM (2.6e-05) ports.
- [experiments/method_cutoffs/](experiments/method_cutoffs/) - how every
  method's suspicion cutoffs were computed, with bootstrap CIs.

## Tests

```bash
pytest tests/
```

needs no rating checkpoint and runs in a few seconds; it covers every scoring
path against fixtures from the thesis HPC code, the cutoffs and label logic,
and the method-switch response.

**Regression suite (run after any model, detector or cutoffs change).**
`tests/test_browser_regression.py` loads all 12 bundled samples under all 4
suspicion methods (48 cases) in headless Chrome and checks for console errors,
that the score and label render, and that each score matches its
parity-checked value in `tests/regression/expected_sample_scores.json`. It is
opt-in because it needs the checkpoint, Playwright and a Chrome binary:

```bash
pip install playwright
RATINGNET_BROWSER_REGRESSION=1 \
RATINGNET_CHECKPOINT=/path/to/best_model.pth \
RATINGNET_CHROME=/path/to/chrome \
pytest tests/test_browser_regression.py
```

It starts its own server on a free port (or set `RATINGNET_REGRESSION_URL` to
test a running one); `RATINGNET_CHROME` can be left out if Playwright's own
Chromium is installed (`playwright install chromium`). After a deliberate
change (an arm swap, new cutoffs), regenerate the expected file with
`experiments/method_parity/build_regression_expected.py` (for a new or
re-trained detector, re-run its parity check first so the expected scores
come from the HPC code path) and review the diff.

## Limitations

> **Suspicion score is a supplementary flag, not a verdict.** It is meant to
> help a human fair-play reviewer decide where to look, not to accuse a
> player automatically. See Barnes and Hernandez-Castro (2015) on the
> false-positive risk of single-game move analysis.

- **The trained detectors learned from synthetic games only.** Their measured
  AUCs come from Maia games with inserted engine moves, not from confirmed
  real cheating cases, and all are weak when only a few moves are engine
  moves (0.54 to 0.58 at 2 percent). None of their per-move outputs locate
  the engine moves, so the app does not use them. The four methods often
  disagree on the same game.

> **Mid-game values are provisional.** For an ongoing game (`Result` header
> `*`), Lichess itself delays the export by a few moves, and any suspicion
> score computed before the game ends should be read as provisional: it can
> shift once more moves are known.

- **Every move re-runs the full bidirectional model over the whole prefix.**
  The model's BiLSTM has a backward pass that needs a completed sequence;
  there is no incremental/streaming inference here; scoring ply *t* means
  running the model over plies 1..*t* from scratch. This is fine for the
  batch PGN/Lichess-ID flows in this prototype. Live mode (see [docs/web-prototype.md](docs/web-prototype.md#live-mode))
  works around it by re-running the batch pipeline on the growing prefix
  and taking only the final-ply estimate each time, so the live curve is a
  sequence of independent from-scratch runs, not a true incremental
  inference; it can differ from the full-game curve for the same finished
  game.

> **Self-prediction fallback baselines are close to meaningless.** If neither
> the caller nor the PGN headers supply a baseline rating, the suspicion
> score compares the model's own final-ply prediction against itself. The
> API always reports which baseline source was used
> (`white_baseline_source`/`black_baseline_source`) so this is visible, not
> silent.

## License

This project is licensed under the MIT license - see LICENSE.

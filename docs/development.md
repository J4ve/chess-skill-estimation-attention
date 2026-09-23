# Development, weights and deployment

[Back to the README](../README.md)

## Frozen weights

The API serves the frozen thesis checkpoint by default: `models/preflight_check_2m/best_model.pth`,
the tuned attention arm (test MAE 171.92). Despite the directory name, that
experiment **is** the reported architecture; its name in the evaluation
records is `attn_tuned`. It is **not committed** to git (see `.gitignore`);
download it from the release, where it is published under the unambiguous
name `attn_tuned_best.pth` with its sha256. See
[Published artifacts](../README.md#published-artifacts).

- Expected location: `models/preflight_check_2m/best_model.pth`
- Override: set `RATINGNET_CHECKPOINT` to load an exact path instead.
- Fallback: if the frozen thesis checkpoint is not found and no override is
  set, the API falls back to Omori's released baseline checkpoint
  (`model_55.pth`, Option C "both" weight strategy) and logs a warning, since
  that checkpoint has no attention or anomaly branch.
  - Expected location: `models/model_55.pth`
  - Download: [Google Drive folder](https://drive.google.com/drive/folders/164qXisHsNAKSM6R7ZMeTeJjPpnZ7s5Rt)
  - **`model_55.pth` is not one of this study's arms.** Its stored `params`
    carry no `use_attention` and no `split_seed`, and its `epochs` (100) and
    `val_batch_size` (8192) are pre-fix defaults, so anything scored with it
    reproduces no reported number. It is deliberately left out of the
    release. That `_discover_checkpoint()` still names it is a fallback of
    last resort, not an endorsement.

The trained suspicion detectors are small and **are committed**, each with a
JSON provenance file next to it (HPC source paths and script hashes, split,
seed, headline AUCs, and whatever the port needs, such as the exact feature
list and standardization constants):

- `src/models/detector_a3g_seed0.pt` (about 2.2 MB): the per-move detector,
  the default method. If it is missing, the API serves S_att in the
  top-level `*_suspicion_*` fields and says so in `warnings`.
- `src/models/detector_lgbm_a0g.txt` (about 1.6 MB): the LightGBM detector as
  a LightGBM text model, read by `src/lgbm_detector.py`'s own tree walker (no
  `lightgbm` package needed). The thesis run kept no model file, so this is a
  re-fit that reproduces every stored thesis score exactly; see
  `experiments/method_parity/`.
- `src/models/detector_cnn_bilstm_a4.pt` (about 1.6 MB): the CNN-BiLSTM
  detector's BiLSTM, attention and head. Its CNN trunk is the served rating
  checkpoint's own (frozen in training), so this method needs the thesis
  checkpoint too.

A missing LightGBM or CNN-BiLSTM file only makes that method unavailable
(`"available": false` in `suspicion_methods`, "Not available on this server"
on the page). Tests and the browser regression suite are described in the
[README](../README.md#tests).

## Used as a submodule

This repository is the `prototype/` submodule of
[J4ve/cs_thesis](https://github.com/J4ve/cs_thesis), the manuscript and
experiment-plan repository. Code changes land here first; the thesis repo
then bumps its submodule pointer to pick them up. Do not edit thesis-repo
files from within this repository's history.

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


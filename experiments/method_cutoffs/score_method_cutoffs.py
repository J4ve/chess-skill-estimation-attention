"""
Score the detector cutoffs' held-out TEST sample under all four suspicion-score
methods, for per-method Typical / Unusual / Highly unusual cutoffs (HPC, CPU).

Reuses the exact 2,822 games of the per-move detector's cutoffs run
(scratch/detector-default/cutoffs/per_side_scores.csv, copied here as
reference_scores.csv) instead of re-sampling, so all four methods are compared
on the same games. The rating-model pass and S_att mirror
api._run_inference; the three trained detectors run through this repo's own
src/detector.py, src/lgbm_detector.py and src/cnn_bilstm_detector.py, copied
into ./src next to this script. Each side is scored against its PGN-header
rating. The per-move detector's scores are checked against the reference run.

Writes per_side_scores.csv and progress.txt next to this script.
"""

import csv
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import cnn_bilstm_detector  # noqa: E402
import detector as detector_module  # noqa: E402
import lgbm_detector  # noqa: E402
from chess_rating_net import ChessEloPredictor  # noqa: E402

RATINGS_MEAN = 1514.0
RATINGS_STD = 366.0
CLOCKS_MEAN = 273.0
CLOCKS_STD = 380.0
MAX_PLIES = 100

THESIS2 = Path.home() / "Bacsain" / "thesis2"
CHECKPOINT_PATH = THESIS2 / "models" / "preflight_check_2m" / "best_model.pth"
PROCESSED_DIR = THESIS2 / "data" / "processed_games"
METHODS = ("s_att", "lgbm_a0g", "detector_a3g_seed0", "cnn_bilstm_a4")


def progress(msg):
    with (HERE / "progress.txt").open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def load_model():
    device = torch.device("cpu")
    saved = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    params = saved["params"]
    use_attention = params.get("use_attention", False)
    model = ChessEloPredictor(
        conv_filters=params.get("conv_filters", 32),
        lstm_layers=params.get("lstm_layers", 3),
        dropout_rate=params.get("dropout_rate", 0.5),
        lstm_h=params.get("lstm_h", 64),
        fc1_h=params.get("fc1_h", 32),
        bidirectional=params.get("bidirectional", True),
        use_attention=use_attention,
        attention_type=params.get("attention_type", "bahdanau"),
        attention_dim=params.get("attention_dim", 64),
        use_anomaly=params.get("use_anomaly", False) or use_attention,
        deeper_cnn=params.get("deeper_cnn", False),
    ).to(device)
    model.load_base_state_dict(saved["model_state_dict"], strict=False)
    model.eval()
    return model, device


def time_to_seconds(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def score_game(model, device, dets, gid):
    month, tail = gid.split("_game_")
    with (PROCESSED_DIR / month / f"game_{tail}.pkl").open("rb") as f:
        d = pickle.load(f)
    positions = d["Positions"][:MAX_PLIES]
    moves = list(d.get("Moves") or [])[:MAX_PLIES]
    secs = [time_to_seconds(c) for c in d["Clocks"][:MAX_PLIES]]
    white_elo, black_elo = float(d.get("WhiteElo")), float(d.get("BlackElo"))

    positions_t = torch.stack(positions).unsqueeze(0).to(device)
    clocks_t = torch.tensor([(c - CLOCKS_MEAN) / CLOCKS_STD for c in secs], dtype=torch.float).unsqueeze(0)
    lengths = torch.tensor([positions_t.size(1)], dtype=torch.int)
    with torch.no_grad():
        out = model(positions_t, clocks_t, lengths, return_attention=True, baseline=None)
    preds = out["per_move_preds"].squeeze(0).cpu() * RATINGS_STD + RATINGS_MEAN
    attn = out["attention_weights"]
    anomaly = model.anomaly_detector(
        predictions=preds.unsqueeze(0),
        baseline=torch.tensor([[white_elo, black_elo]], dtype=torch.float),
        attention_weights=attn.squeeze(0).unsqueeze(0),
    )
    args = (preds.numpy(), attn.squeeze(0).cpu().numpy(), moves, secs, white_elo, black_elo)
    return {
        "s_att": (anomaly["white_score"].item(), anomaly["black_score"].item()),
        "lgbm_a0g": lgbm_detector.score_game(dets["lgbm_a0g"], *args),
        "detector_a3g_seed0": detector_module.score_game(dets["detector_a3g_seed0"], *args),
        "cnn_bilstm_a4": cnn_bilstm_detector.score_game(dets["cnn_bilstm_a4"], model, positions_t, clocks_t),
    }


def main():
    torch.set_num_threads(8)
    progress("starting")
    with (HERE / "reference_scores.csv").open() as f:
        reference = list(csv.DictReader(f))
    progress(f"reference games: {len(reference)}")
    model, device = load_model()
    dets = {
        "lgbm_a0g": lgbm_detector.load_lgbm_detector(),
        "detector_a3g_seed0": detector_module.load_detector(),
        "cnn_bilstm_a4": cnn_bilstm_detector.load_cnn_bilstm_detector(),
    }
    progress("models loaded")
    cols = ["game_id", "time_control", "num_plies"] + [f"{s}_{m}" for m in METHODS for s in ("white", "black")]
    max_ref_diff, t0, errors = 0.0, time.time(), defaultdict(int)
    with (HERE / "per_side_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, ref in enumerate(reference, start=1):
            try:
                r = score_game(model, device, dets, ref["game_id"])
            except Exception as exc:  # recorded, never silently dropped
                errors[type(exc).__name__] += 1
                progress(f"ERROR {ref['game_id']}: {exc!r}")
                continue
            max_ref_diff = max(
                max_ref_diff,
                abs(r["detector_a3g_seed0"][0] - float(ref["white_score"])),
                abs(r["detector_a3g_seed0"][1] - float(ref["black_score"])),
            )
            row = {"game_id": ref["game_id"], "time_control": ref["time_control"], "num_plies": ref["num_plies"]}
            for m in METHODS:
                digits = 4 if m == "s_att" else 6
                row[f"white_{m}"] = round(r[m][0], digits)
                row[f"black_{m}"] = round(r[m][1], digits)
            w.writerow([row[c] for c in cols])
            if i % 200 == 0:
                f.flush()
                progress(f"scored {i} elapsed={time.time() - t0:.0f}s max|A3g - reference|={max_ref_diff:.2e}")
    progress(f"DONE games={len(reference)} errors={dict(errors)} max|A3g - reference|={max_ref_diff:.2e}")


if __name__ == "__main__":
    main()

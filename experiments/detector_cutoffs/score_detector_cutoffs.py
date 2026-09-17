"""
Score a stratified sample of held-out TEST-partition games with the RatingNet
web prototype's trained detector (src/detector.py, arm A3g seed 0), for the
detector's Typical / Unusual / Highly unusual percentile cutoffs.

Same sample definition, seed and filters as scratch/suspicion-cutoffs/score_cutoffs.py
(the S_att cutoffs run): bullet/blitz/rapid up to 700 games each (the S_att run used 600; raised to reach about 3,000 games), classical and
ultrabullet their whole pool, months restricted to non-empty processed_games
directories, at least 20 plies, numeric positive Elo headers. The rating model
forward pass mirrors api._pgn_to_tensor_inputs/_run_inference; the detector
code is the prototype's own src/detector.py copied into ./src next to this script.
S_att is recorded too, for reference only.

Run on CPU. Writes per_side_scores.csv and progress.txt next to this script.
"""

import csv
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import detector as detector_module  # noqa: E402
from chess_rating_net import ChessEloPredictor  # noqa: E402

RATINGS_MEAN = 1514.0
RATINGS_STD = 366.0
CLOCKS_MEAN = 273.0
CLOCKS_STD = 380.0
MAX_PLIES = 100
MIN_PLIES = 20

THESIS2 = Path.home() / "Bacsain" / "thesis2"
CSV_PATH = THESIS2 / "analysis" / "heldout_test_eval" / "attn_tuned__best.csv"
CHECKPOINT_PATH = THESIS2 / "models" / "preflight_check_2m" / "best_model.pth"
PROCESSED_DIR = THESIS2 / "data" / "processed_games"
OUT_DIR = HERE

SEED = 20260915
TARGET_PER_TC = {"bullet": 700, "blitz": 700, "rapid": 700, "classical": 10_000, "ultrabullet": 10_000}


def progress(msg):
    with (OUT_DIR / "progress.txt").open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def load_model():
    device = torch.device("cpu")
    saved = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    params = saved["params"]
    use_attention = params.get("use_attention", False)
    use_anomaly = params.get("use_anomaly", False) or use_attention
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
        use_anomaly=use_anomaly,
        deeper_cnn=params.get("deeper_cnn", False),
    ).to(device)
    model.load_base_state_dict(saved["model_state_dict"], strict=False)
    model.eval()
    return model, device


def stratified_sample(rng):
    unpacked = set()
    for p in PROCESSED_DIR.iterdir():
        if p.is_dir() and any(True for _ in p.iterdir()):
            unpacked.add(p.name)
    progress(f"usable months: {sorted(unpacked)}")
    by_tc = defaultdict(list)
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            gid = row["game_id"]
            if gid.split("_game_")[0] in unpacked:
                by_tc[row["time_control"]].append(gid)
    sample = {}
    for tc, target in TARGET_PER_TC.items():
        pool = list(by_tc.get(tc, []))
        rng.shuffle(pool)
        sample[tc] = pool[:target]
        progress(f"sample: {tc} pool={len(pool)} selected={len(sample[tc])}")
    return sample


def time_to_seconds(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def score_game(model, device, det, gid, tc):
    month, tail = gid.split("_game_")
    path = PROCESSED_DIR / month / f"game_{tail}.pkl"
    if not path.exists():
        return "missing_pkl"
    with path.open("rb") as f:
        d = pickle.load(f)
    positions = d["Positions"][:MAX_PLIES]
    clocks_raw = d["Clocks"][:MAX_PLIES]
    moves = list(d.get("Moves") or [])[:MAX_PLIES]
    if len(positions) < MIN_PLIES:
        return "too_short"
    if len(positions) != len(clocks_raw) or len(moves) != len(positions):
        return "position_clock_mismatch"
    try:
        white_elo = float(d.get("WhiteElo"))
        black_elo = float(d.get("BlackElo"))
    except (TypeError, ValueError):
        return "unparseable_elo"
    if white_elo <= 0 or black_elo <= 0:
        return "non_positive_elo"

    secs = [time_to_seconds(c) for c in clocks_raw]
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
    w, b = detector_module.score_game(
        det, preds.numpy(), attn.squeeze(0).cpu().numpy(), moves, secs, white_elo, black_elo
    )
    return {
        "game_id": gid, "time_control": tc, "num_plies": len(positions),
        "white_score": round(w, 6), "black_score": round(b, 6),
        "white_s_att": round(anomaly["white_score"].item(), 4), "black_s_att": round(anomaly["black_score"].item(), 4),
    }


def main():
    torch.set_num_threads(8)
    progress("starting")
    sample = stratified_sample(random.Random(SEED))
    progress(f"total sampled: {sum(len(v) for v in sample.values())}")
    model, device = load_model()
    det = detector_module.load_detector()
    progress("models loaded")
    scored, skips, t0 = 0, defaultdict(int), time.time()
    with (OUT_DIR / "per_side_scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        cols = ["game_id", "time_control", "num_plies", "white_score", "black_score", "white_s_att", "black_s_att"]
        w.writerow(cols)
        for tc, gids in sample.items():
            for gid in gids:
                try:
                    r = score_game(model, device, det, gid, tc)
                except Exception as exc:  # a bad pickle must not stop the run
                    r = f"error_{type(exc).__name__}"
                if isinstance(r, str):
                    skips[r] += 1
                    continue
                w.writerow([r[c] for c in cols])
                scored += 1
                if scored % 200 == 0:
                    f.flush()
                    progress(f"scored {scored} skipped {sum(skips.values())} elapsed={time.time() - t0:.0f}s")
    progress(f"DONE scored={scored} skipped={sum(skips.values())} reasons={dict(skips)}")


if __name__ == "__main__":
    main()

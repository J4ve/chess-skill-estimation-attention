"""
Compute suspicion-score (S_att) percentile cutoffs from held-out TEST-partition
games, using the exact model + AnomalyDetector code path the RatingNet web
prototype uses (prototype/src/anomaly.py, chess_rating_net.py, attention.py on
this HPC checkout are byte-identical to the fork's src/, diffed 2026-09-15;
format_data.py differs only in an SAN-tracking field the tensors below don't need).

Run on CPU. Writes into scratch/suspicion-cutoffs/:
  per_side_scores.csv   raw per-side S_att records (game_id, time_control, side, score)
  progress.txt          progress marker for tmux polling
"""

import csv
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

THESIS2 = Path.home() / "Bacsain" / "thesis2"
SRC_DIR = THESIS2 / "prototype" / "src"
sys.path.insert(0, str(SRC_DIR))

from chess_rating_net import ChessEloPredictor  # noqa: E402

RATINGS_MEAN = 1514.0
RATINGS_STD = 366.0
CLOCKS_MEAN = 273.0
CLOCKS_STD = 380.0
MAX_PLIES = 100
MIN_PLIES = 20

CSV_PATH = THESIS2 / "analysis" / "heldout_test_eval" / "attn_tuned__best.csv"
CHECKPOINT_PATH = THESIS2 / "models" / "preflight_check_2m" / "best_model.pth"
PROCESSED_DIR = THESIS2 / "data" / "processed_games"
OUT_DIR = THESIS2 / "scratch" / "suspicion-cutoffs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260915
_DRY_RUN_N = int(os.environ.get("SUSCUT_DRY_RUN_N", "0"))
# bullet/blitz/rapid: capped at ~600 each per the brief. classical/ultrabullet:
# "whatever exists" (their pools, ~690/~228 games in the unpacked months, are
# both already well under 600), so no cap is applied there.
TARGET_PER_TC = {
    "bullet": _DRY_RUN_N or 600,
    "blitz": _DRY_RUN_N or 600,
    "rapid": _DRY_RUN_N or 600,
    "classical": _DRY_RUN_N or 10_000,
    "ultrabullet": _DRY_RUN_N or 10_000,
}


def progress(msg: str) -> None:
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


def stratified_sample(rng: random.Random):
    """Group held-out test game_ids by time control, restricted to unpacked
    months, then sample up to TARGET_PER_TC per bucket.

    A "processed_games/<month>" directory can exist but be entirely empty
    (2021-04..2021-10 on this HPC checkout: created, never populated), so
    non-emptiness is checked, not mere directory existence.
    """
    unpacked_months = set()
    for p in PROCESSED_DIR.iterdir():
        if not p.is_dir():
            continue
        try:
            next(p.iterdir())
        except StopIteration:
            continue
        unpacked_months.add(p.name)
    progress(f"usable (non-empty) months: {sorted(unpacked_months)}")
    by_tc = defaultdict(list)
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row["game_id"]
            month = game_id.split("_game_")[0]
            if month not in unpacked_months:
                continue
            by_tc[row["time_control"]].append(game_id)

    sample = {}
    for tc, target in TARGET_PER_TC.items():
        pool = list(by_tc.get(tc, []))
        rng.shuffle(pool)
        sample[tc] = pool[:target]
        progress(f"sample: {tc} pool={len(pool)} selected={len(sample[tc])}")
    return sample


def pkl_path_for(game_id: str) -> Path:
    month, tail = game_id.split("_game_")
    return PROCESSED_DIR / month / f"game_{tail}.pkl"


def time_to_seconds(time_str: str) -> int:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def score_game(model, device, game_id: str, time_control: str):
    path = pkl_path_for(game_id)
    if not path.exists():
        return "missing_pkl"

    with path.open("rb") as f:
        d = pickle.load(f)

    positions = d["Positions"][:MAX_PLIES]
    clocks_raw = d["Clocks"][:MAX_PLIES]
    if len(positions) < MIN_PLIES:
        return "too_short"
    if len(positions) != len(clocks_raw):
        return "position_clock_mismatch"

    try:
        white_elo = float(d.get("WhiteElo"))
        black_elo = float(d.get("BlackElo"))
    except (TypeError, ValueError):
        return "unparseable_elo"
    if white_elo <= 0 or black_elo <= 0:
        return "non_positive_elo"

    positions_t = torch.stack(positions).unsqueeze(0).to(device)
    clocks = [(time_to_seconds(c) - CLOCKS_MEAN) / CLOCKS_STD for c in clocks_raw]
    clocks_t = torch.tensor(clocks, dtype=torch.float).unsqueeze(0).to(device)
    lengths = torch.tensor([positions_t.size(1)], dtype=torch.int)

    with torch.no_grad():
        outputs = model(positions_t, clocks_t, lengths, return_attention=True, baseline=None)

    per_move_preds = outputs["per_move_preds"].squeeze(0).cpu()
    raw_attention = outputs["attention_weights"]
    per_move_preds_orig = per_move_preds * RATINGS_STD + RATINGS_MEAN

    baseline = torch.tensor([[white_elo, black_elo]], dtype=torch.float)
    anomaly = model.anomaly_detector(
        predictions=per_move_preds_orig.unsqueeze(0),
        baseline=baseline,
        attention_weights=raw_attention.squeeze(0).unsqueeze(0) if raw_attention is not None else None,
    )
    return {
        "game_id": game_id,
        "time_control": time_control,
        "num_plies": len(positions),
        "white_score": round(anomaly["white_score"].item(), 4),
        "black_score": round(anomaly["black_score"].item(), 4),
    }


def main():
    progress("starting")
    rng = random.Random(SEED)
    sample = stratified_sample(rng)
    total = sum(len(v) for v in sample.values())
    progress(f"total sampled game_ids: {total}")

    model, device = load_model()
    progress("model loaded")

    csv_out = OUT_DIR / "per_side_scores.csv"
    scored = 0
    skip_reasons = defaultdict(int)
    t0 = time.time()
    with csv_out.open("w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["game_id", "time_control", "num_plies", "white_score", "black_score"])
        for tc, game_ids in sample.items():
            for gid in game_ids:
                result = score_game(model, device, gid, tc)
                if isinstance(result, str):
                    skip_reasons[result] += 1
                    continue
                writer.writerow(
                    [result["game_id"], result["time_control"], result["num_plies"],
                     result["white_score"], result["black_score"]]
                )
                scored += 1
                if scored % 200 == 0:
                    out_f.flush()
                    elapsed = time.time() - t0
                    progress(f"scored {scored} skipped {sum(skip_reasons.values())} elapsed={elapsed:.0f}s")
    skipped = sum(skip_reasons.values())
    progress(f"DONE scored={scored} skipped={skipped} reasons={dict(skip_reasons)}")


if __name__ == "__main__":
    main()

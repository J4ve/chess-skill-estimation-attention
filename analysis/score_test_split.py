"""Score a trained RatingNet checkpoint on the frozen 255k held-out TEST split.

Loads a checkpoint, rebuilds the exact architecture from the hyperparameters
saved *inside* that checkpoint (``ckpt["params"]``), loads the weights with
``strict=True`` (any key mismatch is a hard error, so an architecture flag can
never be silently wrong), runs the deterministic test partition from
``split_manifest.json`` (split_seed 42), and reports MAE in RATING POINTS plus a
per-game absolute-error CSV for the downstream paired bootstrap.

The per-game CSV schema matches ``chess_rating_net.test(per_game_csv=...)`` and
``analysis/scripts/paired_bootstrap.py``:
    game_id, white_err, black_err, white_signed_err, black_signed_err,
    time_control, white_elo, black_elo   (errors in Elo points)

Usage (HPC):
    python analysis/scripts/score_test_split.py \
        --checkpoint models/preflight_check_2m/best_model.pth \
        --data_dir /tmp/ratingnet_store \
        --src_dir src \
        --out_csv analysis/heldout_test_eval/preflight_check_2m__best.csv \
        --out_json analysis/heldout_test_eval/preflight_check_2m__best.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

EXPECTED_MANIFEST_SHA = "dee7d11779df12bfc1242445079f9903b1c48b0787103db63f7b410bd9ed2b3b"
EXPECTED_TEST_GAMES = 255_000


def _import_ratingnet(src_dir: str | None):
    candidates = []
    if src_dir:
        candidates.append(src_dir)
    here = Path(__file__).resolve()
    candidates += [
        str(here.parents[2] / "prototype" / "src"),
        str(here.parents[2] / "src"),
        "src",
        "prototype/src",
    ]
    for c in candidates:
        if c and (Path(c) / "chess_rating_net.py").is_file():
            sys.path.insert(0, str(Path(c).resolve()))
            import chess_rating_net as crn  # noqa: E402

            return crn, str(Path(c).resolve())
    raise FileNotFoundError(f"chess_rating_net.py not found in any of: {candidates}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default="/tmp/ratingnet_store")
    ap.add_argument("--src_dir", default=None, help="dir containing chess_rating_net.py")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--label", default=None, help="free-form tag stored in the JSON")
    args = ap.parse_args()

    crn, src_used = _import_ratingnet(args.src_dir)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path}: no 'model_state_dict' (not a save_checkpoint payload)")
    if "params" not in ckpt:
        raise KeyError(
            f"{ckpt_path}: no 'params' block; cannot rebuild the exact architecture. "
            f"Refusing to guess."
        )
    p = ckpt["params"]

    ratings_mean = float(p.get("ratings_mean", 1514))
    ratings_std = float(p.get("ratings_std", 366))

    model = crn.ChessEloPredictor(
        conv_filters=int(p["conv_filters"]),
        lstm_layers=int(p["lstm_layers"]),
        dropout_rate=float(p["dropout_rate"]),
        lstm_h=int(p["lstm_h"]),
        fc1_h=int(p["fc1_h"]),
        bidirectional=bool(p["bidirectional"]),
        use_attention=bool(p["use_attention"]),
        attention_type=p.get("attention_type", "bahdanau"),
        attention_dim=int(p.get("attention_dim", 64)),
        use_anomaly=bool(p.get("use_anomaly", False)),
        deeper_cnn=bool(p.get("deeper_cnn", False)),
    ).to(device)
    # strict: an architecture-flag mismatch shows up as missing/unexpected keys.
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    store = crn.GameBlobStore.open_if_present(args.data_dir)
    if store is None:
        all_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".pkl"))
        train_names, val_names, test_names, manifest_hash = crn.load_or_create_split(
            args.data_dir, all_files, args.split_seed
        )
        test_files = [os.path.join(args.data_dir, f) for f in test_names]
    else:
        all_files = store.names()
        train_names, val_names, test_names, manifest_hash = crn.load_or_create_split(
            args.data_dir, all_files, args.split_seed
        )
        test_files = test_names

    print(f"src={src_used} device={device}")
    print(f"checkpoint={ckpt_path}  epoch={ckpt.get('epoch')}  best_epoch={ckpt.get('best_epoch')}")
    print(f"arch params: {json.dumps({k: p[k] for k in sorted(p)}, default=str)}")
    print(f"manifest_sha256={manifest_hash}  n_test={len(test_files)}")
    if manifest_hash != EXPECTED_MANIFEST_SHA:
        raise SystemExit(
            f"manifest sha mismatch: got {manifest_hash}, expected {EXPECTED_MANIFEST_SHA}"
        )
    if len(test_files) != EXPECTED_TEST_GAMES:
        raise SystemExit(f"test partition is {len(test_files)}, expected {EXPECTED_TEST_GAMES}")

    test_dataset = crn.ChessGamesDataset(
        test_files, ratings_mean=ratings_mean, ratings_std=ratings_std, store=store
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=crn.collate_fn,
        num_workers=args.num_workers,
    )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    criterion = torch.nn.L1Loss()
    t0 = time.time()
    test_loss_batchmean, loss_by_tc = crn.test(
        model,
        test_loader,
        device,
        criterion,
        ratings_mean=ratings_mean,
        ratings_std=ratings_std,
        per_game_csv=args.out_csv,
    )
    elapsed = time.time() - t0

    # Exact, unbiased MAE straight from the per-game dump (mean over every
    # per-side absolute error). test_loss_batchmean is the same quantity but
    # averaged batch-by-batch, so it is off by a hair on the short last batch.
    n_games = 0
    sum_w = sum_b = 0.0
    per_game_mean_errs = []
    tc_counts: dict[str, int] = {}
    with open(args.out_csv) as f:
        for row in csv.DictReader(f):
            w = float(row["white_err"])
            b = float(row["black_err"])
            sum_w += w
            sum_b += b
            per_game_mean_errs.append((w + b) / 2.0)
            tc_counts[row["time_control"]] = tc_counts.get(row["time_control"], 0) + 1
            n_games += 1

    mae_white = sum_w / n_games
    mae_black = sum_b / n_games
    overall_mae = (sum_w + sum_b) / (2 * n_games)
    mean_of_pergame = sum(per_game_mean_errs) / n_games  # == overall_mae, sanity

    result = {
        "label": args.label,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_epoch": ckpt.get("best_epoch"),
        "checkpoint_best_val_loss": ckpt.get("best_val_loss"),
        "src_dir_used": src_used,
        "manifest_sha256": manifest_hash,
        "n_test_games": n_games,
        "ratings_mean": ratings_mean,
        "ratings_std": ratings_std,
        "arch_params": {k: p[k] for k in sorted(p)},
        "overall_mae_rating_points": overall_mae,
        "mae_white_rating_points": mae_white,
        "mae_black_rating_points": mae_black,
        "per_game_mean_abs_err_mean": mean_of_pergame,
        "test_loss_batchmean_rating_points": test_loss_batchmean,
        "mae_by_time_control": loss_by_tc,
        "n_by_time_control": tc_counts,
        "eval_seconds": elapsed,
        "per_game_csv": str(Path(args.out_csv).resolve()),
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\noverall test MAE (rating points) = {overall_mae:.4f}")
    print(f"  white {mae_white:.4f} | black {mae_black:.4f} | batch-mean {test_loss_batchmean:.4f}")
    print(f"  by tc: {json.dumps({k: round(v, 2) for k, v in loss_by_tc.items()})}")
    print(f"  n={n_games}  elapsed={elapsed:.0f}s")
    print(f"wrote {args.out_csv}\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

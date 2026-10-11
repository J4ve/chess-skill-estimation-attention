"""Live (prefix-only) per-ply rating curves for a sample of TEST games.

For each sampled game and for each of two checkpoints (the served attn_tuned
arm and the mirror_arm_a_gapweight comparison arm), recomputes the model's
*live* estimate at every ply by re-running the full forward pass on the
growing prefix positions[:t] and keeping only the last ply's output -- this
is exactly the prefix-recompute design in prototype/src/live.py and the same
loop analysis/deploy_measure.py already used to measure offline-vs-live
agreement. It is NOT the bidirectional offline per-ply output (which also
sees the rest of the game and is discarded here, only used for the
same-tensor sanity check at t == T).

Imports the prototype's model/data code directly from --src-dir (this
worktree's src/, copied onto the HPC), the same pattern deploy_measure.py
uses, so the frozen trainer on the cluster is never touched.

Usage:
    python move_explanation_curves.py \
        --src-dir ~/move_explain_src \
        --store-dir /tmp/ratingnet_store \
        --games-file sample_games.json \
        --served-checkpoint models/preflight_check_2m/best_model.pth \
        --mirror-a-checkpoint models/mirror_arm_a_gapweight/best_model.pth \
        --device cuda --gpu-index 2 \
        --max-plies 100 \
        --out-jsonl curves.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RATINGS_MEAN, RATINGS_STD = 1514.0, 366.0
CLOCKS_MEAN, CLOCKS_STD = 273.0, 380.0


def build_model(ChessEloPredictor, params: dict, device):
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
        separate_heads=params.get("separate_heads", False),
    ).to(device)
    return model, use_attention, use_anomaly


def load_checkpoint_strict_on_trained_parts(path: Path, device, ChessEloPredictor, torch):
    """Load a checkpoint, building the architecture exactly as api.py does, and
    raise instead of silently warning if any parameter the checkpoint's own
    params say it trained (attention, or anomaly beyond the parameter-free
    module) is missing from the saved state dict."""
    saved = torch.load(path, map_location=device, weights_only=False)
    params = saved["params"]
    model, use_attention, use_anomaly = build_model(ChessEloPredictor, params, device)
    result = model.load_state_dict(saved["model_state_dict"], strict=False)
    # anomaly_detector is parameter-free (see chess_rating_net.py), so it is
    # the only prefix allowed to be missing; any other missing key (a conv,
    # LSTM, fc or attention parameter) means the architecture built here
    # does not actually match the checkpoint, and must raise, not warn.
    served_missing = [k for k in result.missing_keys if not k.startswith("anomaly_detector.")]
    if served_missing:
        raise RuntimeError(
            f"{path}: checkpoint is missing parameter(s): {served_missing}"
        )
    if result.unexpected_keys:
        raise RuntimeError(f"{path}: checkpoint has unexpected keys: {result.unexpected_keys}")
    model.eval()
    return model, params


def load_game_tensors(store, time_to_seconds, gid: str, max_plies: int, torch):
    info = store.load(gid)
    clocks_str = list(info.get("Clocks", []))
    clocks_raw = [time_to_seconds(c) for c in clocks_str]
    positions = torch.stack(info["Positions"])[:max_plies]
    clocks_t = torch.tensor(
        [(c - CLOCKS_MEAN) / CLOCKS_STD for c in clocks_raw], dtype=torch.float
    )[:max_plies]
    return positions, clocks_t, clocks_raw[:max_plies], clocks_str[:max_plies], info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--games-file", required=True)
    ap.add_argument("--served-checkpoint", required=True)
    ap.add_argument("--mirror-a-checkpoint", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Debug: only process the first N games")
    ap.add_argument(
        "--equivalence-check-n",
        type=int,
        default=0,
        help="For the first N games, also call src.api._run_inference on reconstructed PGN "
        "prefixes and assert agreement with this script's own forward_prefix loop.",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src_dir).resolve()))
    import torch  # noqa: E402

    from chess_rating_net import ChessEloPredictor, GameBlobStore  # noqa: E402
    from format_data import time_to_seconds  # noqa: E402

    if args.device == "cuda":
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)
        device = torch.device(f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda")
    else:
        device = torch.device("cpu")

    served_model, served_params = load_checkpoint_strict_on_trained_parts(
        Path(args.served_checkpoint), device, ChessEloPredictor, torch
    )
    mirror_a_model, mirror_a_params = load_checkpoint_strict_on_trained_parts(
        Path(args.mirror_a_checkpoint), device, ChessEloPredictor, torch
    )
    print(f"Loaded served params: {served_params}", flush=True)
    print(f"Loaded mirror_a params: {mirror_a_params}", flush=True)

    store = GameBlobStore.open_if_present(args.store_dir)
    if store is None:
        raise FileNotFoundError(f"No corpus.sqlite under {args.store_dir}")

    games_meta = json.loads(Path(args.games_file).read_text())
    game_ids: list[str] = games_meta["game_ids"]
    if args.limit:
        game_ids = game_ids[: args.limit]

    @torch.no_grad()
    def forward_prefix(model, positions_t, clocks_t, t: int):
        # return_attention=True always: with it False, forward() returns a bare
        # (per_move_preds, last_step_preds) tuple instead of the dict (see
        # ChessEloPredictor.forward's own docstring). Always requesting the
        # dict is free (same compute either way) and keeps one code path.
        pos = positions_t[:t].unsqueeze(0).to(device)
        clk = clocks_t[:t].unsqueeze(0).to(device)
        lengths = torch.tensor([t], dtype=torch.int)
        out = model(pos, clk, lengths, return_attention=True, baseline=None)
        per_move = out["per_move_preds"].squeeze(0).cpu() * RATINGS_STD + RATINGS_MEAN
        attn = out["attention_weights"]
        attn_list = attn.squeeze(0).cpu().tolist() if attn is not None else None
        return per_move, attn_list

    def live_curve(model, positions_t, clocks_t, T: int, want_attention_at_final: bool):
        white_curve, black_curve = [None] * T, [None] * T
        final_attention = None
        for t in range(1, T + 1):
            per_move, attn_list = forward_prefix(model, positions_t, clocks_t, t)
            white_curve[t - 1] = round(per_move[-1, 0].item(), 4)
            black_curve[t - 1] = round(per_move[-1, 1].item(), 4)
            if t == T:
                final_attention = attn_list
                full_offline_white = [round(x, 4) for x in per_move[:, 0].tolist()]
                full_offline_black = [round(x, 4) for x in per_move[:, 1].tolist()]
        return white_curve, black_curve, final_attention, full_offline_white, full_offline_black

    equivalence_report = []
    n_done = 0
    with open(args.out_jsonl, "w") as out_f:
        for gid in game_ids:
            positions_t, clocks_t, clocks_raw, clocks_str, info = load_game_tensors(
                store, time_to_seconds, gid, args.max_plies, torch
            )
            T = positions_t.size(0)
            if T < 1:
                continue

            served_white, served_black, served_attn, off_w, off_b = live_curve(
                served_model, positions_t, clocks_t, T, want_attention_at_final=True
            )
            mirror_white, mirror_black, _, _, _ = live_curve(
                mirror_a_model, positions_t, clocks_t, T, want_attention_at_final=False
            )

            # Sanity: the live curve's last point IS the offline full-game pass
            # (same tensors, same forward call) -- must match to float tolerance.
            assert abs(served_white[-1] - off_w[-1]) < 1e-3, (gid, served_white[-1], off_w[-1])
            assert abs(served_black[-1] - off_b[-1]) < 1e-3, (gid, served_black[-1], off_b[-1])

            record = {
                "game_id": gid,
                "n_plies": T,
                "white_elo": info.get("WhiteElo"),
                "black_elo": info.get("BlackElo"),
                "time_control": info.get("Time"),
                "result": info.get("Result"),
                "moves_uci": list(info["Moves"])[:T],
                "clocks_seconds": clocks_raw,
                "clocks_str": clocks_str,
                "served": {
                    "live_white": served_white,
                    "live_black": served_black,
                    "offline_white": off_w,
                    "offline_black": off_b,
                    "offline_attention": [round(a, 6) for a in served_attn] if served_attn else None,
                },
                "mirror_arm_a_gapweight": {
                    "live_white": mirror_white,
                    "live_black": mirror_black,
                },
            }
            out_f.write(json.dumps(record) + "\n")
            n_done += 1

            if n_done <= args.equivalence_check_n:
                eq = _equivalence_check_one_game(
                    gid, info, clocks_str, T, served_white, served_black,
                    args.src_dir, served_model, device,
                )
                equivalence_report.append(eq)
                print(f"[equivalence] {gid}: {eq}", flush=True)

            if n_done % 50 == 0:
                print(f"{n_done}/{len(game_ids)} games done", flush=True)

    print(f"Wrote {n_done} game records to {args.out_jsonl}", flush=True)
    if equivalence_report:
        Path(args.out_jsonl + ".equivalence.json").write_text(json.dumps(equivalence_report, indent=2))


def _equivalence_check_one_game(
    gid, info, clocks_str, T, served_white, served_black, src_dir, served_model, device
):
    """Rebuild PGN prefixes from the stored game and call the real
    src.api._run_inference on each one, comparing its white/black final
    rating against this script's own forward_prefix loop. This is what lets
    the write-up say the LIVE curve equals "what the prototype shows"."""
    import chess
    import chess.pgn

    sys.path.insert(0, str(Path(src_dir).resolve()))
    import api  # noqa: E402 -- the prototype's own inference module

    # api.MODEL / api.DEVICE are normally set by a FastAPI startup event; here
    # we point them at the already-loaded served model so _run_inference's
    # own parsing + tensorization path (not this script's) produces the
    # comparison numbers.
    api.MODEL = served_model
    api.DEVICE = device

    checks = []
    sample_plies = sorted({1, 2, max(1, T // 4), max(1, T // 2), max(1, 3 * T // 4), T})
    for t in sample_plies:
        if t > T:
            continue
        game = chess.pgn.Game()
        game.headers["WhiteElo"] = str(info.get("WhiteElo") or "?")
        game.headers["BlackElo"] = str(info.get("BlackElo") or "?")
        game.headers["TimeControl"] = str(info.get("Time") or "-")
        game.headers["Result"] = "*"
        node = game
        for i in range(t):
            mv = chess.Move.from_uci(info["Moves"][i])
            node = node.add_variation(mv)
            if i < len(clocks_str):
                node.comment = f"[%clk {clocks_str[i]}]"
        pgn_text = str(game)
        result = api._run_inference(pgn_text, None, None, top_k=5)
        api_white = result["white_final_rating"]
        api_black = result["black_final_rating"]
        own_white = served_white[t - 1]
        own_black = served_black[t - 1]
        checks.append(
            {
                "ply": t,
                "api_white": api_white,
                "own_white": own_white,
                "white_abs_diff": abs(api_white - own_white),
                "api_black": api_black,
                "own_black": own_black,
                "black_abs_diff": abs(api_black - own_black),
            }
        )
    max_diff = max((max(c["white_abs_diff"], c["black_abs_diff"]) for c in checks), default=0.0)
    return {"game_id": gid, "max_abs_diff": max_diff, "checks": checks}


if __name__ == "__main__":
    main()

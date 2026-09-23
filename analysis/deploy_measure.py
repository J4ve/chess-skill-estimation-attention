"""Chapter 4 deployment measurements: per-move latency and offline-vs-live agreement.

Imports the prototype's model/data code (``chess_rating_net``, ``format_data``)
directly from ``--src-dir`` rather than editing it. Intended to run on the HPC
where torch and the trained checkpoint live.

What it measures, for each game in --games-file:

  * "offline" pass: one forward pass over the whole game (length T), exactly
    what batch/offline scoring does.
  * "live" simulation: T forward passes, one per move arrival, each on the
    growing prefix positions[:t] (t = 1..T) -- the prefix-recompute inference
    design chapter3.tex Sec. 3 (bidirectionality) describes as the deployed
    design. Batch size is always 1. Each forward call is timed individually
    (post warm-up) to build the per-move latency distribution.

Because the LSTM is bidirectional, the offline pass at ply t sees plies
t+1..T that the live prefix at time t cannot see. The two are therefore only
expected to agree exactly at the final ply (t == T), where the live prefix
*is* the whole game -- both computations are then literally the same forward
pass on the same tensors. The script reports both:
  (a) final-ply agreement (offline vs. live's last step) -- the quantity
      chapter3.tex's "floating-point tolerance" claim is about, and
  (b) full per-ply agreement (offline curve vs. live's per-ply curve) --
      expected to diverge mid-game by construction; quantified, not hidden.

S_max / S_mean are not implemented in prototype/src/anomaly.py (only the
attention-weighted S_att / "combined_score" is). This script computes S_max
and S_mean itself from the per_move_deviation tensor anomaly.py already
returns, without modifying prototype code.

R_baseline: api.py's live baseline defaults to the model's own final-ply
prediction when no explicit baseline is supplied (a known open bug, "M4" /
Sec. H.1c in analysis/170k-verification.md -- not fixed here, out of scope).
To keep this agreement measurement meaningful independent of that bug, both
the offline and live computations in this script are given the SAME
explicit baseline: the game's true pre-game WhiteElo/BlackElo from the
corpus record. This sidesteps M4 rather than exercising it.

use_anomaly override: the frozen checkpoint's stored params carry
use_anomaly=False (the anomaly head has zero learnable parameters -- see
analysis/170k-verification.md Sec. H.1a -- so there is nothing to load
regardless). This script builds the model with use_anomaly=True explicitly
so S_att/S_max/S_mean can be computed; this does not change any loaded
weight, only whether the (parameter-free) scoring module is attached.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"n": 0}
    s = sorted(latencies_ms)
    return {
        "n": len(s),
        "mean_ms": statistics.fmean(s),
        "p50_ms": _percentile(s, 0.50),
        "p95_ms": _percentile(s, 0.95),
        "p99_ms": _percentile(s, 0.99),
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="Path to the prototype src/ directory to import")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store-dir", required=True, help="Directory holding corpus.sqlite (GameBlobStore)")
    ap.add_argument("--games-file", required=True, help="Output of sample_test_games.py")
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--warmup-games", type=int, default=3)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--skip-agreement", action="store_true", help="Latency-only run (used for the CPU arm)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.src_dir).resolve()))
    import torch  # noqa: E402

    from chess_rating_net import ChessEloPredictor, GameBlobStore  # noqa: E402
    from format_data import time_to_seconds  # noqa: E402

    if args.device == "cuda":
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)
        device = torch.device(f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda")
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        import subprocess

        try:
            device_name = subprocess.run(
                ["lscpu"], capture_output=True, text=True, check=True
            ).stdout
            device_name = next(
                (l.split(":", 1)[1].strip() for l in device_name.splitlines() if l.startswith("Model name")),
                "unknown CPU",
            )
        except Exception:
            device_name = "unknown CPU"

    RATINGS_MEAN, RATINGS_STD = 1514.0, 366.0
    CLOCKS_MEAN, CLOCKS_STD = 273.0, 380.0

    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = saved["params"]

    model = ChessEloPredictor(
        conv_filters=params.get("conv_filters", 32),
        lstm_layers=params.get("lstm_layers", 3),
        dropout_rate=params.get("dropout_rate", 0.5),
        lstm_h=params.get("lstm_h", 64),
        fc1_h=params.get("fc1_h", 32),
        bidirectional=params.get("bidirectional", True),
        use_attention=params.get("use_attention", False),
        attention_type=params.get("attention_type", "bahdanau"),
        attention_dim=params.get("attention_dim", 64),
        use_anomaly=True,  # override: parameter-free module, see module docstring
    ).to(device)
    load_result = model.load_state_dict(saved["model_state_dict"], strict=False)
    model.eval()

    store = GameBlobStore.open_if_present(args.store_dir)
    if store is None:
        raise FileNotFoundError(f"No corpus.sqlite found under {args.store_dir}")

    games_meta = json.loads(Path(args.games_file).read_text())
    game_ids: list[str] = games_meta["game_ids"]

    def load_game_tensors(name: str):
        game_info = store.load(name)
        clocks_raw = [time_to_seconds(c) for c in game_info.get("Clocks", [])]
        clocks = [(c - CLOCKS_MEAN) / CLOCKS_STD for c in clocks_raw]
        clocks_t = torch.tensor(clocks, dtype=torch.float)[: args.max_plies]
        positions_t = torch.stack(game_info["Positions"])[: args.max_plies]
        white_elo = float(game_info["WhiteElo"])
        black_elo = float(game_info["BlackElo"])
        return positions_t, clocks_t, white_elo, black_elo, game_info

    @torch.no_grad()
    def forward_prefix(positions_t, clocks_t, t: int, baseline: "torch.Tensor"):
        pos = positions_t[:t].unsqueeze(0).to(device)
        clk = clocks_t[:t].unsqueeze(0).to(device)
        lengths = torch.tensor([t], dtype=torch.int)
        out = model(pos, clk, lengths, return_attention=True, baseline=baseline.to(device))
        per_move = out["per_move_preds"].squeeze(0).cpu() * RATINGS_STD + RATINGS_MEAN
        attn = out["attention_weights"]
        anomaly_out = out["anomaly"]
        deviation = anomaly_out["per_move_deviation"].squeeze(0).cpu()  # (t, 2)
        white_att_score = anomaly_out["white_score"].item()
        black_att_score = anomaly_out["black_score"].item()
        white_dev = deviation[:, 0]
        black_dev = deviation[:, 1]
        s_att = white_att_score + black_att_score
        s_max = (white_dev.max().item() + black_dev.max().item())
        s_mean = (white_dev.mean().item() + black_dev.mean().item())
        return {
            "white_rating": per_move[-1, 0].item(),
            "black_rating": per_move[-1, 1].item(),
            "s_att": s_att,
            "s_max": s_max,
            "s_mean": s_mean,
            "full_curve_white": per_move[:, 0].tolist(),
            "full_curve_black": per_move[:, 1].tolist(),
        }

    def time_one_forward(positions_t, clocks_t, t: int, baseline: "torch.Tensor") -> tuple[float, dict]:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = forward_prefix(positions_t, clocks_t, t, baseline)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, result

    # ---- Warm-up (discarded from timing) ----
    warmup_ids = game_ids[: args.warmup_games]
    for gid in warmup_ids:
        positions_t, clocks_t, white_elo, black_elo, _ = load_game_tensors(gid)
        baseline = torch.tensor([[white_elo, black_elo]], dtype=torch.float)
        T = positions_t.size(0)
        for t in (1, max(1, T // 2), T):
            forward_prefix(positions_t, clocks_t, t, baseline)

    # ---- Main measurement loop ----
    all_latencies_ms: list[float] = []
    ply_latencies: dict[int, list[float]] = {10: [], 50: [], 100: []}

    per_game_agreement = []
    missing_attention_keys = [k for k in load_result.missing_keys if k.startswith("attention.")]

    for gid in game_ids:
        positions_t, clocks_t, white_elo, black_elo, game_info = load_game_tensors(gid)
        T = positions_t.size(0)
        if T < 1:
            continue
        baseline = torch.tensor([[white_elo, black_elo]], dtype=torch.float)

        # Offline: single full-length forward pass.
        offline = forward_prefix(positions_t, clocks_t, T, baseline)

        if args.skip_agreement:
            # Latency-only: still walk every ply for the latency distribution,
            # but skip the O(T) python-side bookkeeping used for agreement.
            for t in range(1, T + 1):
                lat_ms, _ = time_one_forward(positions_t, clocks_t, t, baseline)
                all_latencies_ms.append(lat_ms)
                if t in ply_latencies:
                    ply_latencies[t].append(lat_ms)
            continue

        live_white_curve = [None] * T
        live_black_curve = [None] * T
        live_final = None
        for t in range(1, T + 1):
            lat_ms, result = time_one_forward(positions_t, clocks_t, t, baseline)
            all_latencies_ms.append(lat_ms)
            if t in ply_latencies:
                ply_latencies[t].append(lat_ms)
            live_white_curve[t - 1] = result["white_rating"]
            live_black_curve[t - 1] = result["black_rating"]
            if t == T:
                live_final = result

        # Final-ply agreement (the floating-point-tolerance claim).
        final_white_diff = abs(offline["full_curve_white"][-1] - live_final["white_rating"])
        final_black_diff = abs(offline["full_curve_black"][-1] - live_final["black_rating"])
        final_s_att_diff = abs(offline["s_att"] - live_final["s_att"])
        final_s_max_diff = abs(offline["s_max"] - live_final["s_max"])
        final_s_mean_diff = abs(offline["s_mean"] - live_final["s_mean"])

        # Full per-ply agreement (expected to diverge; quantified below).
        white_diffs = [
            abs(o - l) for o, l in zip(offline["full_curve_white"], live_white_curve)
        ]
        black_diffs = [
            abs(o - l) for o, l in zip(offline["full_curve_black"], live_black_curve)
        ]

        per_game_agreement.append(
            {
                "game_id": gid,
                "n_plies": T,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "final_ply": {
                    "white_rating_abs_diff": final_white_diff,
                    "black_rating_abs_diff": final_black_diff,
                    "s_att_abs_diff": final_s_att_diff,
                    "s_max_abs_diff": final_s_max_diff,
                    "s_mean_abs_diff": final_s_mean_diff,
                    "offline_white_rating": offline["full_curve_white"][-1],
                    "offline_black_rating": offline["full_curve_black"][-1],
                    "offline_s_att": offline["s_att"],
                    "offline_s_max": offline["s_max"],
                    "offline_s_mean": offline["s_mean"],
                },
                "per_ply_curve": {
                    "white_abs_diff_max": max(white_diffs),
                    "white_abs_diff_mean": statistics.fmean(white_diffs),
                    "black_abs_diff_max": max(black_diffs),
                    "black_abs_diff_mean": statistics.fmean(black_diffs),
                },
            }
        )

    out: dict[str, Any] = {
        "device": args.device,
        "device_name": device_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_params": params,
        "missing_state_dict_keys": load_result.missing_keys,
        "missing_attention_keys": missing_attention_keys,
        "unexpected_state_dict_keys": load_result.unexpected_keys,
        "use_anomaly_override": True,
        "n_games": len(game_ids),
        "max_plies": args.max_plies,
        "latency_overall_ms": summarize(all_latencies_ms),
        "latency_at_ply": {str(k): summarize(v) for k, v in ply_latencies.items()},
    }
    if not args.skip_agreement:
        n = len(per_game_agreement)
        final_white = [g["final_ply"]["white_rating_abs_diff"] for g in per_game_agreement]
        final_black = [g["final_ply"]["black_rating_abs_diff"] for g in per_game_agreement]
        final_s_att = [g["final_ply"]["s_att_abs_diff"] for g in per_game_agreement]
        final_s_max = [g["final_ply"]["s_max_abs_diff"] for g in per_game_agreement]
        final_s_mean = [g["final_ply"]["s_mean_abs_diff"] for g in per_game_agreement]
        ply_white_max = [g["per_ply_curve"]["white_abs_diff_max"] for g in per_game_agreement]
        ply_white_mean = [g["per_ply_curve"]["white_abs_diff_mean"] for g in per_game_agreement]
        ply_black_max = [g["per_ply_curve"]["black_abs_diff_max"] for g in per_game_agreement]
        ply_black_mean = [g["per_ply_curve"]["black_abs_diff_mean"] for g in per_game_agreement]
        out["agreement_summary"] = {
            "n_games": n,
            "final_ply": {
                "white_rating_abs_diff": {"max": max(final_white), "mean": statistics.fmean(final_white)},
                "black_rating_abs_diff": {"max": max(final_black), "mean": statistics.fmean(final_black)},
                "s_att_abs_diff": {"max": max(final_s_att), "mean": statistics.fmean(final_s_att)},
                "s_max_abs_diff": {"max": max(final_s_max), "mean": statistics.fmean(final_s_max)},
                "s_mean_abs_diff": {"max": max(final_s_mean), "mean": statistics.fmean(final_s_mean)},
            },
            "full_per_ply_curve_bidirectional_lookahead": {
                "white_rating_abs_diff": {
                    "max_over_games": max(ply_white_max),
                    "mean_over_games": statistics.fmean(ply_white_mean),
                },
                "black_rating_abs_diff": {
                    "max_over_games": max(ply_black_max),
                    "mean_over_games": statistics.fmean(ply_black_mean),
                },
                "note": (
                    "Expected to be large: offline per-ply predictions see the "
                    "whole game (bidirectional LSTM); live per-ply predictions "
                    "at ply t only see plies 1..t. Only the final ply (t==T) is "
                    "the same computation in both paths."
                ),
            },
        }
        out["per_game_agreement"] = per_game_agreement

    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json}")
    print(json.dumps(out.get("latency_overall_ms", {}), indent=2))
    if "agreement_summary" in out:
        print(json.dumps(out["agreement_summary"]["final_ply"], indent=2))


if __name__ == "__main__":
    main()

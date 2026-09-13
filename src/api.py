"""
FastAPI inference service for the RatingNet prototype.

The service imports model classes directly (no shelling out to
``python src/chess_rating_net.py``) and loads a checkpoint once at startup,
then exposes a PGN upload/text endpoint that returns per-move rating
predictions and attention weights.

By default it serves the frozen thesis checkpoint
(``models/preflight_check_2m/best_model.pth``, the tuned attention arm).
Set ``RATINGNET_CHECKPOINT`` to load an exact path instead. If the thesis
checkpoint is not found and no override is set, it falls back to Omori's
released baseline (``model_55.pth``) and logs a warning, since that
checkpoint has no attention or anomaly branch.

Predictions and attention weights are logged to ``logs/predictions.jsonl``
for reproducibility and fairness-review audit trails.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Allow imports both when running `python src/api.py` and via `from src.api import ...`.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import chess.pgn
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chess_rating_net import ChessEloPredictor
from format_data import board_to_array, parse_game, time_to_seconds


# Baseline normalization constants (must match training; see audit report).
RATINGS_MEAN = 1514.0
RATINGS_STD = 366.0
CLOCKS_MEAN = 273.0
CLOCKS_STD = 380.0
MAX_PLIES = 100

MODEL: ChessEloPredictor | None = None
MODEL_PARAMS: dict[str, Any] | None = None
DEVICE: torch.device | None = None
CHECKPOINT_PATH: Path | None = None

# Lightweight cache: keyed by (white_name, black_name, pgn_hash) -> result.
RESULT_CACHE: dict[str, Any] = {}
CACHE_MAX_SIZE = 128

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _log_prediction(payload: dict[str, Any]) -> None:
    """Append a JSONL audit record with a timestamp."""
    record = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **payload}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# The frozen thesis checkpoint: the tuned-attention arm (test MAE 171.92),
# preferred over Omori's plain baseline whenever it is available.
FROZEN_CHECKPOINT_REL = Path("models") / "preflight_check_2m" / "best_model.pth"
# Omori's released baseline: no attention, no anomaly branch. Only served
# when the frozen thesis checkpoint above cannot be found.
BASELINE_CHECKPOINT_REL = Path("models") / "model_55.pth"


def _find_checkpoint(relative: Path) -> Path | None:
    """Search the standard candidate roots for ``relative``, or return None."""
    candidates = [
        _SRC_DIR.parent / relative,
        Path(relative),
        Path("prototype") / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _discover_checkpoint() -> Path:
    """Return the checkpoint path to serve.

    Resolution order:
    1. ``RATINGNET_CHECKPOINT`` env var, if set. Loaded exactly as given;
       raises if the file does not exist (never silently falls back).
    2. The frozen thesis checkpoint (``FROZEN_CHECKPOINT_REL``).
    3. Omori's baseline ``model_55.pth``, with a logged warning that the
       thesis model is not being served.
    """
    override = os.environ.get("RATINGNET_CHECKPOINT")
    if override:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(
                f"RATINGNET_CHECKPOINT is set to '{override_path}', but that file does not exist."
            )
        return override_path.resolve()

    frozen = _find_checkpoint(FROZEN_CHECKPOINT_REL)
    if frozen is not None:
        return frozen

    baseline = _find_checkpoint(BASELINE_CHECKPOINT_REL)
    if baseline is not None:
        logging.warning(
            "Frozen thesis checkpoint (%s) not found; falling back to Omori's "
            "baseline model_55.pth. This serves the plain baseline model, not "
            "the thesis attention model.",
            FROZEN_CHECKPOINT_REL,
        )
        return baseline

    raise FileNotFoundError(
        "No checkpoint found. Set RATINGNET_CHECKPOINT to an exact path, or "
        f"place the frozen thesis checkpoint at {FROZEN_CHECKPOINT_REL} "
        f"(preferred) or the baseline at {BASELINE_CHECKPOINT_REL}."
    )


def _load_model() -> tuple[ChessEloPredictor, dict[str, Any], torch.device, Path]:
    """Load the discovered checkpoint once and build the model from its stored params."""
    checkpoint_path = _discover_checkpoint()
    logging.info("Loading checkpoint from %s", checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = saved["params"]

    # Serve the architecture recorded in the checkpoint's own params, with
    # defaults matching the plain baseline for older checkpoints that lack a
    # key. This lets both the frozen thesis checkpoint (attention, anomaly,
    # deeper CNN all enabled) and the plain baseline load with their weights
    # applied exactly, instead of leaving randomly initialized modules that
    # would silently corrupt every prediction.
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
        use_anomaly=params.get("use_anomaly", False),
        deeper_cnn=params.get("deeper_cnn", False),
    ).to(device)

    model.load_base_state_dict(saved["model_state_dict"], strict=False)
    model.eval()
    logging.info("Model loaded on %s", device)
    return model, params, device, checkpoint_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup hook: load the model once and keep it in memory."""
    global MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH
    _setup_logging()
    MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH = _load_model()
    logging.info("Serving checkpoint: %s", CHECKPOINT_PATH)
    yield
    MODEL = None


app = FastAPI(
    title="RatingNet Prototype Inference API",
    description="PGN-upload demo for move-by-move chess rating estimation and anomaly scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


class PGNTextRequest(BaseModel):
    pgn: str
    white_baseline: float | None = None
    black_baseline: float | None = None


def _pgn_to_tensor_inputs(pgn_text: str):
    """Parse a PGN string and produce model-ready tensors plus move list."""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Could not parse PGN text")

    game_info = parse_game(game, max_plies=MAX_PLIES)
    if game_info is None:
        raise ValueError("PGN has no usable clock annotations")

    positions = torch.stack(game_info["Positions"])  # (seq, 12, 8, 8)
    clocks = [time_to_seconds(c) for c in game_info["Clocks"]]
    clocks = [(c - CLOCKS_MEAN) / CLOCKS_STD for c in clocks]
    clocks = torch.tensor(clocks, dtype=torch.float)

    positions = positions.unsqueeze(0)  # (1, seq, 12, 8, 8)
    clocks = clocks.unsqueeze(0)  # (1, seq)
    lengths = torch.tensor([positions.size(1)], dtype=torch.int)
    return positions, clocks, lengths, game_info["Moves"], game.headers


def _stable_cache_key(*parts: str) -> str:
    """Stable cache key across Python process restarts."""
    digest = hashlib.md5("|".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest


def _run_inference(
    pgn_text: str,
    white_baseline: float | None,
    black_baseline: float | None,
) -> dict[str, Any]:
    assert MODEL is not None and DEVICE is not None

    positions, clocks, lengths, moves, headers = _pgn_to_tensor_inputs(pgn_text)
    positions = positions.to(DEVICE)
    clocks = clocks.to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(
            positions,
            clocks,
            lengths,
            return_attention=True,
            baseline=None,
        )

    per_move_preds = outputs["per_move_preds"].squeeze(0).cpu()  # (seq, 2)
    attention_weights = outputs["attention_weights"].squeeze(0).cpu().tolist() if outputs["attention_weights"] is not None else []

    # De-standardize ratings back to original Elo scale.
    per_move_preds_orig = per_move_preds * RATINGS_STD + RATINGS_MEAN

    # Use provided baselines; fall back to final predicted rating for each side.
    if white_baseline is None:
        white_baseline = per_move_preds_orig[-1, 0].item()
    if black_baseline is None:
        black_baseline = per_move_preds_orig[-1, 1].item()
    baseline = torch.tensor([[white_baseline, black_baseline]], dtype=torch.float)

    # Anomaly scoring is only available when the served checkpoint has an anomaly
    # branch. Omori's baseline model_55.pth does not, so emit neutral scores there.
    seq_len = per_move_preds.size(0)
    if MODEL.anomaly_detector is not None:
        # Compute anomaly scores using the attention-weighted deviation formula.
        # Predictions must be de-standardized so deviations are on the same Elo
        # scale as the baseline ratings.
        anomaly = MODEL.anomaly_detector(
            predictions=per_move_preds_orig.unsqueeze(0),
            baseline=baseline,
            attention_weights=outputs["attention_weights"].squeeze(0).unsqueeze(0) if outputs["attention_weights"] is not None else None,
        )
        white_deviation = anomaly["white_deviation"].squeeze(0)
        black_deviation = anomaly["black_deviation"].squeeze(0)
        white_score = anomaly["white_score"].item()
        black_score = anomaly["black_score"].item()
        combined_score = anomaly["combined_score"].item()
    else:
        white_deviation = torch.zeros(seq_len)
        black_deviation = torch.zeros(seq_len)
        white_score = 0.0
        black_score = 0.0
        combined_score = 0.0

    move_records = []
    for i in range(seq_len):
        move_records.append(
            {
                "ply": i + 1,
                "move": moves[i] if i < len(moves) else None,
                "white_rating": round(per_move_preds_orig[i, 0].item(), 2),
                "black_rating": round(per_move_preds_orig[i, 1].item(), 2),
                "attention_weight": round(attention_weights[i], 6) if i < len(attention_weights) else None,
                "white_deviation": round(white_deviation[i].item(), 4),
                "black_deviation": round(black_deviation[i].item(), 4),
            }
        )

    return {
        "headers": dict(headers),
        "white_baseline": round(white_baseline, 2),
        "black_baseline": round(black_baseline, 2),
        "white_final_rating": round(per_move_preds_orig[-1, 0].item(), 2),
        "black_final_rating": round(per_move_preds_orig[-1, 1].item(), 2),
        "white_suspicion_score": round(white_score, 4),
        "black_suspicion_score": round(black_score, 4),
        "combined_suspicion_score": round(combined_score, 4),
        "per_move": move_records,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint_path": str(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None,
    }


@app.post("/predict/pgn")
async def predict_pgn_text(request: PGNTextRequest):
    """Submit PGN text and receive per-move ratings and attention weights."""
    cache_key = _stable_cache_key("pgn_text", request.pgn, str(request.white_baseline), str(request.black_baseline))
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    try:
        result = _run_inference(
            request.pgn,
            request.white_baseline,
            request.black_baseline,
        )
    except Exception as exc:
        logging.exception("Inference failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})

    _log_prediction({"source": "pgn_text", "cache_key": cache_key, **result})
    response = {"status": "ok", **result}
    if len(RESULT_CACHE) >= CACHE_MAX_SIZE:
        RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
    RESULT_CACHE[cache_key] = response
    return response


@app.post("/predict/upload")
async def predict_pgn_upload(
    file: UploadFile = File(...),
    white_baseline: float | None = Form(None),
    black_baseline: float | None = Form(None),
):
    """Upload a .pgn file and receive per-move ratings and attention weights."""
    content = await file.read()
    pgn_text = content.decode("utf-8", errors="replace")
    cache_key = _stable_cache_key("pgn_upload", file.filename, pgn_text, str(white_baseline), str(black_baseline))
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    try:
        result = _run_inference(pgn_text, white_baseline, black_baseline)
    except Exception as exc:
        logging.exception("Inference failed")
        return JSONResponse(status_code=400, content={"error": str(exc)})

    _log_prediction({"source": "pgn_upload", "filename": file.filename, "cache_key": cache_key, **result})
    response = {"status": "ok", **result}
    if len(RESULT_CACHE) >= CACHE_MAX_SIZE:
        RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
    RESULT_CACHE[cache_key] = response
    return response


@app.get("/")
async def root():
    return {
        "message": "RatingNet Prototype API",
        "endpoints": {
            "health": "/health",
            "predict_text": "POST /predict/pgn",
            "predict_upload": "POST /predict/upload",
        },
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

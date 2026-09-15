"""
FastAPI inference service for the RatingNet prototype.

The service imports model classes directly (no shelling out to
``python src/chess_rating_net.py``) and loads a checkpoint once at startup,
then exposes PGN upload/text/Lichess-ID endpoints that return per-move rating
predictions, attention weights, suspicion scores, and critical moves for a
human fair-play reviewer.

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
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from baseline import resolve_actual_rating, resolve_baseline
from chess_rating_net import ChessEloPredictor
from critical_moves import DEFAULT_MIN_PLY, DEFAULT_TOP_K, compute_critical_moves
from format_data import board_to_array, parse_game, time_control_bucket, time_to_seconds
from lichess_client import LichessError, fetch_game_pgn, parse_game_id
from live import LiveGamePrefix, stream_game, stream_tv
from suspicion_labels import label_for_score, resolve_cutoffs


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

# Suspicion-score percentile cutoffs (src/static/suspicion_cutoffs.json), loaded once
# at startup. None when the file is missing, in which case suspicion labels are
# omitted from responses rather than guessed. See "Suspicion labels" in the README.
SUSPICION_CUTOFFS: dict[str, Any] | None = None
SUSPICION_CUTOFFS_PATH = Path(__file__).resolve().parent / "static" / "suspicion_cutoffs.json"

# Lightweight cache: keyed by a hash of the request inputs -> result.
RESULT_CACHE: dict[str, Any] = {}
CACHE_MAX_SIZE = 128

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"
STATIC_DIR = Path(__file__).resolve().parent / "static"


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
    # key. This lets both the frozen thesis checkpoint (attention enabled;
    # deeper_cnn absent, so off) and the plain baseline load with their
    # weights applied exactly, instead of leaving randomly initialized
    # modules that would silently corrupt every prediction.
    use_attention = params.get("use_attention", False)
    # AnomalyDetector has no learnable parameters (pure post-hoc arithmetic
    # over the rating curve and attention weights), and load_base_state_dict
    # already skips anomaly_detector.* keys, so it is always safe to build
    # one whenever attention is enabled, even if the checkpoint itself was
    # saved with use_anomaly=False.
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
    logging.info("Model loaded on %s", device)
    return model, params, device, checkpoint_path


def _load_suspicion_cutoffs() -> dict[str, Any] | None:
    if not SUSPICION_CUTOFFS_PATH.exists():
        logging.warning(
            "No suspicion cutoffs file at %s; suspicion_score labels will be omitted "
            "from responses until one is generated.",
            SUSPICION_CUTOFFS_PATH,
        )
        return None
    with SUSPICION_CUTOFFS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup hook: load the model once and keep it in memory."""
    global MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH, SUSPICION_CUTOFFS
    _setup_logging()
    MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH = _load_model()
    logging.info("Serving checkpoint: %s", CHECKPOINT_PATH)
    SUSPICION_CUTOFFS = _load_suspicion_cutoffs()
    yield
    MODEL = None


app = FastAPI(
    title="RatingNet Prototype Inference API",
    description="Web prototype for move-by-move chess rating estimation and anomaly scoring.",
    version="0.2.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def revalidate_frontend(request, call_next):
    # The page and its assets change together on every prototype update. Without
    # this, a browser can pair a fresh index.html (served directly at "/") with a
    # stale cached app.js, leaving new controls on the page with no handlers.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


class PGNTextRequest(BaseModel):
    pgn: str
    white_baseline: float | None = None
    black_baseline: float | None = None


class LichessGameRequest(BaseModel):
    game_id: str
    white_baseline: float | None = None
    black_baseline: float | None = None


def _pgn_to_tensor_inputs(pgn_text: str):
    """Parse a PGN string and produce model-ready tensors plus move lists."""
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Could not parse PGN text")

    game_info = parse_game(game, max_plies=MAX_PLIES)
    if game_info is None:
        raise ValueError(
            "PGN has no usable clock annotations. This model requires per-move "
            "[%clk ...] comments; export the game with clock times included."
        )

    positions = torch.stack(game_info["Positions"])  # (seq, 12, 8, 8)
    clocks = [time_to_seconds(c) for c in game_info["Clocks"]]
    clocks = [(c - CLOCKS_MEAN) / CLOCKS_STD for c in clocks]
    clocks = torch.tensor(clocks, dtype=torch.float)

    positions = positions.unsqueeze(0)  # (1, seq, 12, 8, 8)
    clocks = clocks.unsqueeze(0)  # (1, seq)
    lengths = torch.tensor([positions.size(1)], dtype=torch.int)
    return positions, clocks, lengths, game_info["Moves"], game_info["SAN"], game.headers


def _stable_cache_key(*parts: str) -> str:
    """Stable cache key across Python process restarts."""
    digest = hashlib.md5("|".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest


def _cache_put(cache_key: str, response: dict[str, Any]) -> None:
    if len(RESULT_CACHE) >= CACHE_MAX_SIZE:
        RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
    RESULT_CACHE[cache_key] = response


def _run_inference(
    pgn_text: str,
    white_baseline_request: float | None,
    black_baseline_request: float | None,
    top_k: int,
    min_ply: int = DEFAULT_MIN_PLY,
) -> dict[str, Any]:
    assert MODEL is not None and DEVICE is not None

    positions, clocks, lengths, moves, san_moves, headers = _pgn_to_tensor_inputs(pgn_text)
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
    raw_attention = outputs["attention_weights"]
    attention_weights = raw_attention.squeeze(0).cpu().tolist() if raw_attention is not None else []

    # De-standardize ratings back to original Elo scale.
    per_move_preds_orig = per_move_preds * RATINGS_STD + RATINGS_MEAN

    white_resolution = resolve_baseline(
        "White", white_baseline_request, headers.get("WhiteElo"),
        lambda: per_move_preds_orig[-1, 0].item(),
    )
    black_resolution = resolve_baseline(
        "Black", black_baseline_request, headers.get("BlackElo"),
        lambda: per_move_preds_orig[-1, 1].item(),
    )
    baseline = torch.tensor([[white_resolution.value, black_resolution.value]], dtype=torch.float)

    # Independent of the baseline (which a caller can override): the actual
    # rating is read only from the PGN header, or null if the source game
    # did not record one.
    white_actual_rating = resolve_actual_rating(headers.get("WhiteElo"))
    black_actual_rating = resolve_actual_rating(headers.get("BlackElo"))

    warnings: list[str] = [w for w in (white_resolution.warning, black_resolution.warning) if w]

    # Anomaly scoring is only available when the served checkpoint has an anomaly
    # branch. Omori's baseline model_55.pth does not, so emit neutral scores there.
    seq_len = per_move_preds.size(0)
    anomaly_available = MODEL.anomaly_detector is not None
    if anomaly_available:
        # Compute anomaly scores using the attention-weighted deviation formula.
        # Predictions must be de-standardized so deviations are on the same Elo
        # scale as the baseline ratings.
        anomaly = MODEL.anomaly_detector(
            predictions=per_move_preds_orig.unsqueeze(0),
            baseline=baseline,
            attention_weights=raw_attention.squeeze(0).unsqueeze(0) if raw_attention is not None else None,
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
        warnings.append(
            "This checkpoint has no attention/anomaly branch; suspicion scores "
            "are neutral zeros, not a real assessment."
        )

    move_records = []
    for i in range(seq_len):
        move_records.append(
            {
                "ply": i + 1,
                "move": san_moves[i] if i < len(san_moves) else None,
                "uci": moves[i] if i < len(moves) else None,
                "white_rating": round(per_move_preds_orig[i, 0].item(), 2),
                "black_rating": round(per_move_preds_orig[i, 1].item(), 2),
                "attention_weight": round(attention_weights[i], 6) if i < len(attention_weights) else None,
                "white_deviation": round(white_deviation[i].item(), 4),
                "black_deviation": round(black_deviation[i].item(), 4),
            }
        )

    critical_moves, critical_move_warnings = compute_critical_moves(
        move_records, anomaly_available, top_k=top_k, min_ply=min_ply
    )
    warnings.extend(critical_move_warnings)

    # Suspicion labels compare this game's score with ordinary held-out test
    # games (see suspicion_labels.py and src/static/suspicion_cutoffs.json's own
    # "note"), never a cheat-detection verdict. Omitted (None) when the checkpoint
    # has no anomaly branch or no cutoffs file has been generated yet.
    white_suspicion_label = None
    black_suspicion_label = None
    suspicion_cutoffs_used = None
    if anomaly_available and SUSPICION_CUTOFFS is not None:
        tc_bucket = time_control_bucket(headers.get("TimeControl"))
        resolved = resolve_cutoffs(SUSPICION_CUTOFFS, tc_bucket)
        white_suspicion_label = label_for_score(white_score, resolved)
        black_suspicion_label = label_for_score(black_score, resolved)
        suspicion_cutoffs_used = {
            "p75": resolved.p75,
            "p95": resolved.p95,
            "source": resolved.source,
            "time_control": tc_bucket,
        }

    result_header = (headers.get("Result") or "*").strip()
    ongoing = result_header == "*"

    return {
        "headers": dict(headers),
        "white_baseline": round(white_resolution.value, 2),
        "black_baseline": round(black_resolution.value, 2),
        "white_baseline_source": white_resolution.source,
        "black_baseline_source": black_resolution.source,
        "white_actual_rating": round(white_actual_rating, 2) if white_actual_rating is not None else None,
        "black_actual_rating": round(black_actual_rating, 2) if black_actual_rating is not None else None,
        "warnings": warnings,
        "white_final_rating": round(per_move_preds_orig[-1, 0].item(), 2),
        "black_final_rating": round(per_move_preds_orig[-1, 1].item(), 2),
        "white_suspicion_score": round(white_score, 4),
        "black_suspicion_score": round(black_score, 4),
        "combined_suspicion_score": round(combined_score, 4),
        "white_suspicion_label": white_suspicion_label,
        "black_suspicion_label": black_suspicion_label,
        "suspicion_cutoffs_used": suspicion_cutoffs_used,
        "per_move": move_records,
        "critical_moves": critical_moves,
        "critical_moves_min_ply": min_ply,
        "ongoing": ongoing,
        "provisional": ongoing,
    }


def _run_inference_or_422(
    pgn_text: str,
    white_baseline: float | None,
    black_baseline: float | None,
    top_k: int,
    min_ply: int = DEFAULT_MIN_PLY,
) -> dict[str, Any]:
    """Run inference, translating expected PGN/model-input problems into HTTP 422."""
    try:
        return _run_inference(pgn_text, white_baseline, black_baseline, top_k, min_ply)
    except ValueError as exc:
        logging.info("Rejecting request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint_path": str(CHECKPOINT_PATH) if CHECKPOINT_PATH is not None else None,
    }


@app.post("/predict/pgn")
async def predict_pgn_text(
    request: PGNTextRequest,
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Critical moves to return per side"),
    min_ply: int = Query(
        DEFAULT_MIN_PLY, ge=0, le=100, description="Exclude plies before this from critical-move ranking"
    ),
):
    """Submit PGN text and receive per-move ratings, suspicion scores, and critical moves."""
    cache_key = _stable_cache_key(
        "pgn_text", request.pgn, str(request.white_baseline), str(request.black_baseline), str(top_k), str(min_ply)
    )
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    result = _run_inference_or_422(request.pgn, request.white_baseline, request.black_baseline, top_k, min_ply)

    _log_prediction({"source": "pgn_text", "cache_key": cache_key, **result})
    response = {"status": "ok", **result}
    _cache_put(cache_key, response)
    return response


@app.post("/predict/upload")
async def predict_pgn_upload(
    file: UploadFile = File(...),
    white_baseline: float | None = Form(None),
    black_baseline: float | None = Form(None),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Critical moves to return per side"),
    min_ply: int = Query(
        DEFAULT_MIN_PLY, ge=0, le=100, description="Exclude plies before this from critical-move ranking"
    ),
):
    """Upload a .pgn file and receive per-move ratings, suspicion scores, and critical moves."""
    content = await file.read()
    pgn_text = content.decode("utf-8", errors="replace")
    cache_key = _stable_cache_key(
        "pgn_upload", file.filename, pgn_text, str(white_baseline), str(black_baseline), str(top_k), str(min_ply)
    )
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    result = _run_inference_or_422(pgn_text, white_baseline, black_baseline, top_k, min_ply)

    _log_prediction({"source": "pgn_upload", "filename": file.filename, "cache_key": cache_key, **result})
    response = {"status": "ok", **result}
    _cache_put(cache_key, response)
    return response


@app.post("/predict/lichess")
async def predict_lichess_game(
    request: LichessGameRequest,
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Critical moves to return per side"),
    min_ply: int = Query(
        DEFAULT_MIN_PLY, ge=0, le=100, description="Exclude plies before this from critical-move ranking"
    ),
):
    """Fetch a Lichess game by ID or URL and receive per-move ratings, suspicion scores, and critical moves."""
    try:
        game_id = parse_game_id(request.game_id)
    except LichessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    cache_key = _stable_cache_key(
        "lichess", game_id, str(request.white_baseline), str(request.black_baseline), str(top_k), str(min_ply)
    )
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]

    try:
        pgn_text = await fetch_game_pgn(game_id)
    except LichessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    result = _run_inference_or_422(pgn_text, request.white_baseline, request.black_baseline, top_k, min_ply)
    result["lichess_game_id"] = game_id

    _log_prediction({"source": "lichess", "game_id": game_id, "cache_key": cache_key, **result})
    response = {"status": "ok", **result}
    _cache_put(cache_key, response)
    return response


@app.get("/live/stream/{game_id}")
async def live_stream_game(
    game_id: str,
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
    min_ply: int = Query(DEFAULT_MIN_PLY, ge=0, le=100),
    white_baseline: float | None = Query(None),
    black_baseline: float | None = Query(None),
):
    """Follow one ongoing (or just-finished) Lichess game, streaming prefix-based
    per-move analysis over Server-Sent Events. See README "Live mode"."""
    try:
        resolved_id = parse_game_id(game_id)
    except LichessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    try:
        pgn_text = await fetch_game_pgn(resolved_id)
    except LichessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    prefix = LiveGamePrefix(headers={}, max_plies=MAX_PLIES)
    try:
        prefix.seed_from_pgn(pgn_text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    generator = stream_game(resolved_id, prefix, _run_inference, top_k, min_ply, white_baseline, black_baseline)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/live/tv")
async def live_tv(
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
    min_ply: int = Query(DEFAULT_MIN_PLY, ge=0, le=100),
):
    """Follow Lichess TV's currently featured game over Server-Sent Events."""
    generator = stream_tv(fetch_game_pgn, _run_inference, top_k, min_ply, MAX_PLIES)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api")
async def api_index():
    return {
        "message": "RatingNet Prototype API",
        "endpoints": {
            "health": "GET /health",
            "predict_text": "POST /predict/pgn",
            "predict_upload": "POST /predict/upload",
            "predict_lichess": "POST /predict/lichess",
            "live_stream_game": "GET /live/stream/{game_id} (Server-Sent Events)",
            "live_tv": "GET /live/tv (Server-Sent Events)",
        },
    }


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {
            "message": "RatingNet Prototype API (no web page built yet; see GET /api)",
        }
    return FileResponse(index_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

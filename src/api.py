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

Every response carries four suspicion scores under ``suspicion_methods``, all
computed from one rating-model pass: the parameter-free attention-weighted
score S_att (``anomaly.py``), the LightGBM detector (``lgbm_detector.py``,
thesis arm A0g), the per-move detector (``detector.py``, arm A3g, the default
and the one mirrored into the top-level ``*_suspicion_*`` fields) and the full
CNN-BiLSTM detector (``cnn_bilstm_detector.py``, arm A4). The web page switches
between them without another request.

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
import cnn_bilstm_detector as cnn_bilstm_module
import detector as detector_module
import lgbm_detector as lgbm_module
from format_data import (
    board_to_array,
    compute_time_spent,
    game_setup_error,
    parse_game,
    parse_time_control,
    time_control_bucket,
    time_to_seconds,
)
from lichess_client import LichessError, fetch_game_pgn, parse_game_id
from live import LiveGamePrefix, sse_event, stream_game, stream_tv
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

# The trained detector (main suspicion score), loaded once at startup. None when
# its weights are missing, in which case the computed score S_att is the main
# score instead and a warning says so.
DETECTOR: detector_module.Detector | None = None

# The other two trained detectors, selectable in the page. None when their
# weights are missing, in which case that method is reported as unavailable.
LGBM_DETECTOR: lgbm_module.LgbmDetector | None = None
CNN_BILSTM_DETECTOR: cnn_bilstm_module.CnnBilstmDetector | None = None

SCORE_KIND_DETECTOR = "detector"
SCORE_KIND_COMPUTED = "computed"
COMPUTED_SCORE_ID = "s_att"

# Selectable suspicion-score methods, in the order the page lists them. Each id
# is also the "score" its cutoffs entry must name.
METHOD_IDS = (COMPUTED_SCORE_ID, lgbm_module.SCORE_ID, detector_module.SCORE_ID, cnn_bilstm_module.SCORE_ID)
DEFAULT_METHOD = detector_module.SCORE_ID

# Percentile cutoffs for every method, loaded once at startup from one file keyed
# by method id. Each entry names its own score in its "score" field and is
# ignored when that does not match its key, so cutoffs can never be applied to
# another method's score. A method with no valid entry gets no labels rather
# than guessed ones. See "Suspicion labels" in docs/web-prototype.md.
METHOD_CUTOFFS: dict[str, dict[str, Any]] = {}
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


def _load_method_cutoffs(path: Path) -> dict[str, dict[str, Any]]:
    """{method_id: cutoffs} for every entry in the combined cutoffs file whose
    own "score" field matches its key; others are dropped with a warning."""
    if not path.exists():
        logging.warning("No cutoffs file at %s; suspicion labels will be omitted from responses.", path)
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    loaded = {}
    for method_id, cutoffs in (data.get("methods") or {}).items():
        if cutoffs.get("score") != method_id:
            logging.warning(
                "Cutoffs entry '%s' in %s is for score '%s'; ignoring it so labels are never "
                "computed against another score's distribution.",
                method_id,
                path,
                cutoffs.get("score"),
            )
            continue
        loaded[method_id] = cutoffs
    for method_id in METHOD_IDS:
        if method_id not in loaded:
            logging.warning("No cutoffs for suspicion method '%s'; its labels will be omitted.", method_id)
    return loaded


def _load_detector() -> detector_module.Detector | None:
    if not detector_module.DEFAULT_WEIGHTS_PATH.exists():
        logging.warning(
            "Detector weights not found at %s; serving the computed score as the main suspicion score.",
            detector_module.DEFAULT_WEIGHTS_PATH,
        )
        return None
    return detector_module.load_detector(device=DEVICE)


def _load_lgbm_detector() -> lgbm_module.LgbmDetector | None:
    if not lgbm_module.DEFAULT_MODEL_PATH.exists():
        logging.warning("LightGBM detector model not found at %s; that method is unavailable.",
                        lgbm_module.DEFAULT_MODEL_PATH)
        return None
    return lgbm_module.load_lgbm_detector()


def _load_cnn_bilstm_detector() -> cnn_bilstm_module.CnnBilstmDetector | None:
    if not cnn_bilstm_module.DEFAULT_WEIGHTS_PATH.exists():
        logging.warning("CNN-BiLSTM detector weights not found at %s; that method is unavailable.",
                        cnn_bilstm_module.DEFAULT_WEIGHTS_PATH)
        return None
    return cnn_bilstm_module.load_cnn_bilstm_detector(device=DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup hook: load the model once and keep it in memory."""
    global MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH, METHOD_CUTOFFS
    global DETECTOR, LGBM_DETECTOR, CNN_BILSTM_DETECTOR
    _setup_logging()
    MODEL, MODEL_PARAMS, DEVICE, CHECKPOINT_PATH = _load_model()
    logging.info("Serving checkpoint: %s", CHECKPOINT_PATH)
    DETECTOR = _load_detector()
    LGBM_DETECTOR = _load_lgbm_detector()
    CNN_BILSTM_DETECTOR = _load_cnn_bilstm_detector()
    METHOD_CUTOFFS = _load_method_cutoffs(SUSPICION_CUTOFFS_PATH)
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

    setup_error = game_setup_error(game.headers)
    if setup_error:
        raise ValueError(setup_error)

    game_info = parse_game(game, max_plies=MAX_PLIES)
    if game_info is None:
        raise ValueError(
            "PGN has no usable clock annotations. This model requires per-move "
            "[%clk ...] comments; export the game with clock times included."
        )

    positions = torch.stack(game_info["Positions"])  # (seq, 12, 8, 8)
    raw_clock_seconds = [time_to_seconds(c) for c in game_info["Clocks"]]
    clocks = [(c - CLOCKS_MEAN) / CLOCKS_STD for c in raw_clock_seconds]
    clocks = torch.tensor(clocks, dtype=torch.float)

    positions = positions.unsqueeze(0)  # (1, seq, 12, 8, 8)
    clocks = clocks.unsqueeze(0)  # (1, seq)
    lengths = torch.tensor([positions.size(1)], dtype=torch.int)
    return (
        positions,
        clocks,
        lengths,
        game_info["Moves"],
        game_info["SAN"],
        game.headers,
        raw_clock_seconds,
    )


def _stable_cache_key(*parts: str) -> str:
    """Stable cache key across Python process restarts."""
    digest = hashlib.md5("|".join(parts).encode("utf-8"), usedforsecurity=False).hexdigest()
    return digest


def _cache_put(cache_key: str, response: dict[str, Any]) -> None:
    if len(RESULT_CACHE) >= CACHE_MAX_SIZE:
        RESULT_CACHE.pop(next(iter(RESULT_CACHE)))
    RESULT_CACHE[cache_key] = response


def _labels_for_scores(
    cutoffs: dict[str, Any] | None,
    white_score: float,
    black_score: float,
    time_control: str | None,
) -> tuple[str | None, str | None, dict[str, Any] | None]:
    """(white_label, black_label, cutoffs_used) for one score, or all None when
    that score has no cutoffs file."""
    if cutoffs is None:
        return None, None, None
    resolved = resolve_cutoffs(cutoffs, time_control)
    overall_n_sides = (cutoffs.get("overall") or {}).get("n_sides")
    used = {
        "score": cutoffs.get("score"),
        "p75": resolved.p75,
        "p95": resolved.p95,
        "source": resolved.source,
        "time_control": time_control,
        "provisional": cutoffs.get("provisional", False),
        "provisional_note": cutoffs.get("provisional_note"),
        "provisional_games": overall_n_sides // 2 if isinstance(overall_n_sides, int) else None,
    }
    return label_for_score(white_score, resolved), label_for_score(black_score, resolved), used


def _suspicion_methods(
    method_scores: dict[str, tuple[float, float] | None], time_control: str | None
) -> dict[str, dict[str, Any]]:
    """Per-method scores and labels, in METHOD_IDS order, for the page's method
    selector. A method whose model is not loaded is ``available: false`` with
    null fields; S_att scores are rating points, the detectors' are 0 to 1."""
    methods = {}
    for method_id in METHOD_IDS:
        scores = method_scores.get(method_id)
        entry: dict[str, Any] = {
            "available": scores is not None,
            "scale": "rating_points" if method_id == COMPUTED_SCORE_ID else "unit",
            "white_score": None,
            "black_score": None,
            "white_label": None,
            "black_label": None,
            "cutoffs_used": None,
        }
        if scores is not None:
            white, black = scores
            digits = 4 if method_id == COMPUTED_SCORE_ID else 6
            labels = _labels_for_scores(METHOD_CUTOFFS.get(method_id), white, black, time_control)
            entry.update(
                white_score=round(white, digits),
                black_score=round(black, digits),
                white_label=labels[0],
                black_label=labels[1],
                cutoffs_used=labels[2],
            )
        methods[method_id] = entry
    return methods


def _run_inference(
    pgn_text: str,
    white_baseline_request: float | None,
    black_baseline_request: float | None,
    top_k: int,
    min_ply: int = DEFAULT_MIN_PLY,
) -> dict[str, Any]:
    assert MODEL is not None and DEVICE is not None

    positions, clocks, lengths, moves, san_moves, headers, raw_clock_seconds = _pgn_to_tensor_inputs(pgn_text)
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
        white_computed = anomaly["white_score"].item()
        black_computed = anomaly["black_score"].item()
        combined_computed = anomaly["combined_score"].item()
    else:
        white_deviation = torch.zeros(seq_len)
        black_deviation = torch.zeros(seq_len)
        white_computed = 0.0
        black_computed = 0.0
        combined_computed = 0.0
        warnings.append(
            "This checkpoint has no attention/anomaly branch; suspicion scores "
            "are neutral zeros, not a real assessment."
        )

    # The trained detectors read the rating model's own outputs (per-move
    # estimates and attention, or for A4 its CNN trunk), so they need the same
    # attention-enabled checkpoint. All of them reuse this one rating-model pass.
    method_scores: dict[str, tuple[float, float] | None] = {method_id: None for method_id in METHOD_IDS}
    if anomaly_available:
        method_scores[COMPUTED_SCORE_ID] = (white_computed, black_computed)
    if anomaly_available and raw_attention is not None:
        detector_args = (
            per_move_preds_orig.numpy(),
            raw_attention.squeeze(0).cpu().numpy(),
            moves,
            raw_clock_seconds,
            white_resolution.value,
            black_resolution.value,
        )
        if DETECTOR is not None:
            method_scores[detector_module.SCORE_ID] = detector_module.score_game(DETECTOR, *detector_args)
        if LGBM_DETECTOR is not None:
            method_scores[lgbm_module.SCORE_ID] = lgbm_module.score_game(LGBM_DETECTOR, *detector_args)
        if CNN_BILSTM_DETECTOR is not None:
            method_scores[cnn_bilstm_module.SCORE_ID] = cnn_bilstm_module.score_game(
                CNN_BILSTM_DETECTOR, MODEL, positions, clocks
            )
    white_detector, black_detector = method_scores[detector_module.SCORE_ID] or (None, None)

    # Clock remaining is already parsed from the PGN's [%clk ...] comments (one of
    # the model's own inputs; see the "clockTime" METRIC_INFO entry). Time spent
    # per move is derived from it, not a separate model output.
    tc_base, tc_increment = parse_time_control(headers.get("TimeControl"))
    time_spent_seconds = compute_time_spent(raw_clock_seconds, tc_base, tc_increment)

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
                "clock_seconds": raw_clock_seconds[i] if i < len(raw_clock_seconds) else None,
                "time_spent_seconds": time_spent_seconds[i] if i < len(time_spent_seconds) else None,
            }
        )

    critical_moves, critical_move_warnings = compute_critical_moves(
        move_records, anomaly_available, top_k=top_k, min_ply=min_ply
    )
    warnings.extend(critical_move_warnings)

    # Suspicion labels compare each score with ordinary held-out test games, using
    # that score's own cutoffs file (see suspicion_labels.py and each file's
    # "note"), never a cheat-detection verdict. Omitted (None) when the checkpoint
    # has no anomaly branch or the score has no cutoffs file.
    tc_bucket = time_control_bucket(headers.get("TimeControl"))
    suspicion_methods = _suspicion_methods(method_scores, tc_bucket)
    computed_entry = suspicion_methods[COMPUTED_SCORE_ID]
    computed_labels = (computed_entry["white_label"], computed_entry["black_label"], computed_entry["cutoffs_used"])

    if white_detector is not None and black_detector is not None:
        score_kind = SCORE_KIND_DETECTOR
        white_score, black_score = white_detector, black_detector
        detector_entry = suspicion_methods[detector_module.SCORE_ID]
        main_labels = (detector_entry["white_label"], detector_entry["black_label"], detector_entry["cutoffs_used"])
    else:
        score_kind = SCORE_KIND_COMPUTED
        white_score, black_score = white_computed, black_computed
        main_labels = computed_labels
        if anomaly_available:
            warnings.append(
                "The trained detector is not available on this server, so the main suspicion "
                "score shown is the computed attention-weighted score (S_att)."
            )

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
        "suspicion_score_kind": score_kind,
        "white_suspicion_score": round(white_score, 6),
        "black_suspicion_score": round(black_score, 6),
        "white_suspicion_label": main_labels[0],
        "black_suspicion_label": main_labels[1],
        "suspicion_cutoffs_used": main_labels[2],
        "white_computed_score": round(white_computed, 4),
        "black_computed_score": round(black_computed, 4),
        "combined_computed_score": round(combined_computed, 4),
        "white_computed_label": computed_labels[0],
        "black_computed_label": computed_labels[1],
        "computed_cutoffs_used": computed_labels[2],
        "default_suspicion_method": DEFAULT_METHOD,
        "suspicion_methods": suspicion_methods,
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


async def _live_stream_game_events(
    game_id: str,
    top_k: int,
    min_ply: int,
    white_baseline: float | None,
    black_baseline: float | None,
):
    """Resolve, fetch, and seed a live game, yielding setup failures as SSE
    error events instead of letting them raise before the stream starts."""
    try:
        resolved_id = parse_game_id(game_id)
    except LichessError as exc:
        yield sse_event({"type": "error", "detail": exc.detail})
        return

    try:
        pgn_text = await fetch_game_pgn(resolved_id)
    except LichessError as exc:
        yield sse_event({"type": "error", "detail": exc.detail})
        return

    prefix = LiveGamePrefix(headers={}, max_plies=MAX_PLIES)
    try:
        prefix.seed_from_pgn(pgn_text)
    except ValueError as exc:
        yield sse_event({"type": "error", "detail": str(exc)})
        return

    async for event in stream_game(
        resolved_id, prefix, _run_inference, top_k, min_ply, white_baseline, black_baseline
    ):
        yield event


@app.get("/live/stream/{game_id}")
async def live_stream_game(
    game_id: str,
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20),
    min_ply: int = Query(DEFAULT_MIN_PLY, ge=0, le=100),
    white_baseline: float | None = Query(None),
    black_baseline: float | None = Query(None),
):
    """Follow one ongoing (or just-finished) Lichess game, streaming prefix-based
    per-move analysis over Server-Sent Events. See README "Live mode".

    Setup problems (bad game ID, Lichess 404/429, a non-standard game, missing
    clocks) are sent as an SSE {"type": "error"} event and the stream then
    closes, never as a bare HTTP error status: EventSource cannot read an
    error response's body, so a raised HTTPException here would only ever
    reach the browser as a generic "connection lost" with no reason.
    """
    generator = _live_stream_game_events(game_id, top_k, min_ply, white_baseline, black_baseline)
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

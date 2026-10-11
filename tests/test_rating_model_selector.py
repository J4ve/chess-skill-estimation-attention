"""Tests for the "Rating model" selector (api.py's MODEL vs GAP_WEIGHTED_MODEL).

Uses tiny, randomly initialized ChessEloPredictor instances in place of the
real checkpoints, so these run with no RATINGNET_CHECKPOINT set, like the
rest of tests/ (see CLAUDE.md "Tests need no model"). The two tiny models are
independently seeded, so their forward passes produce different numbers;
that difference is what proves _run_inference actually used the model it was
told to, not always the same one.
"""

import pytest
import torch
from fastapi import HTTPException

import api
from chess_rating_net import ChessEloPredictor

WITH_CLOCK_PGN = (
    '[Event "Demo"]\n[White "A"]\n[Black "B"]\n[WhiteElo "1500"]\n'
    '[BlackElo "1500"]\n[TimeControl "300+0"]\n\n'
    "1. e4 {[%clk 0:05:00]} e5 {[%clk 0:05:00]} "
    "2. Nf3 {[%clk 0:04:58]} Nc6 {[%clk 0:04:58]} *"
)

TINY_PARAMS = dict(conv_filters=2, lstm_layers=1, lstm_h=4, fc1_h=4, use_attention=True, use_anomaly=True)


def _tiny_model(seed):
    torch.manual_seed(seed)
    return ChessEloPredictor(**TINY_PARAMS).eval()


@pytest.fixture
def two_models(monkeypatch):
    """Patch api's loaded-model globals with two distinct tiny models, and
    disable the trained detectors (which expect the real architecture's
    feature shapes, not this test's shrunken one)."""
    main_model = _tiny_model(0)
    gap_model = _tiny_model(1)
    monkeypatch.setattr(api, "MODEL", main_model)
    monkeypatch.setattr(api, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(api, "GAP_WEIGHTED_MODEL", gap_model)
    monkeypatch.setattr(api, "DETECTOR", None)
    monkeypatch.setattr(api, "LGBM_DETECTOR", None)
    monkeypatch.setattr(api, "CNN_BILSTM_DETECTOR", None)
    return main_model, gap_model


def test_resolve_rating_model_main_is_the_main_global(two_models):
    main_model, _ = two_models
    model, device = api._resolve_rating_model(api.MAIN_MODEL_ID)
    assert model is main_model
    assert device == torch.device("cpu")


def test_resolve_rating_model_gap_weighted_is_the_gap_weighted_global(two_models):
    _, gap_model = two_models
    model, _ = api._resolve_rating_model(api.GAP_WEIGHTED_MODEL_ID)
    assert model is gap_model


def test_resolve_rating_model_unknown_id_raises_value_error(two_models):
    with pytest.raises(ValueError, match="Unknown rating model"):
        api._resolve_rating_model("not_a_real_model")


def test_resolve_rating_model_gap_weighted_unavailable_raises_value_error(monkeypatch, two_models):
    monkeypatch.setattr(api, "GAP_WEIGHTED_MODEL", None)
    with pytest.raises(ValueError, match="not available"):
        api._resolve_rating_model(api.GAP_WEIGHTED_MODEL_ID)


def test_run_inference_or_422_translates_unknown_model_to_422(two_models):
    with pytest.raises(HTTPException) as exc_info:
        api._run_inference_or_422(WITH_CLOCK_PGN, None, None, 5, 10, "not_a_real_model")
    assert exc_info.value.status_code == 422


def test_run_inference_or_422_translates_unavailable_gap_weighted_to_422(monkeypatch, two_models):
    monkeypatch.setattr(api, "GAP_WEIGHTED_MODEL", None)
    with pytest.raises(HTTPException) as exc_info:
        api._run_inference_or_422(WITH_CLOCK_PGN, None, None, 5, 10, api.GAP_WEIGHTED_MODEL_ID)
    assert exc_info.value.status_code == 422


def test_run_inference_default_model_id_is_main(two_models):
    # No model_id argument at all (matches live.py's 5-positional-arg calls,
    # and every predict/* request before this selector existed).
    result_implicit = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10)
    result_explicit_main = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10, api.MAIN_MODEL_ID)
    assert result_implicit == result_explicit_main


def test_run_inference_main_omits_rating_model_key(two_models):
    result = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10, api.MAIN_MODEL_ID)
    assert "rating_model" not in result


def test_run_inference_gap_weighted_includes_rating_model_key(two_models):
    result = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10, api.GAP_WEIGHTED_MODEL_ID)
    assert result["rating_model"] == api.GAP_WEIGHTED_MODEL_ID


def test_run_inference_actually_uses_the_selected_model(two_models):
    # The two tiny models are independently (randomly) initialized, so their
    # predictions differ; this is what proves the selection is wired through
    # to the forward pass, anomaly_detector, and de-standardization, not
    # just threaded through as an unused parameter.
    result_main = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10, api.MAIN_MODEL_ID)
    result_gap = api._run_inference(WITH_CLOCK_PGN, None, None, 5, 10, api.GAP_WEIGHTED_MODEL_ID)
    assert result_main["white_final_rating"] != result_gap["white_final_rating"]
    assert result_main["black_final_rating"] != result_gap["black_final_rating"]


# --- Loading the gap-weighted checkpoint -------------------------------------


def _write_tiny_checkpoint(path, seed, ratings_mean=None, ratings_std=None):
    torch.manual_seed(seed)
    model = ChessEloPredictor(**TINY_PARAMS)
    params = {
        **TINY_PARAMS,
        "ratings_mean": api.RATINGS_MEAN if ratings_mean is None else ratings_mean,
        "ratings_std": api.RATINGS_STD if ratings_std is None else ratings_std,
    }
    torch.save({"model_state_dict": model.state_dict(), "params": params}, path)


def test_load_gap_weighted_model_strict_load_succeeds(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "best_model.pth"
    _write_tiny_checkpoint(ckpt_path, seed=2)
    monkeypatch.setenv("RATINGNET_CHECKPOINT_GAP_WEIGHTED", str(ckpt_path))

    loaded = api._load_gap_weighted_model(torch.device("cpu"))

    assert loaded is not None
    model, path = loaded
    assert isinstance(model, ChessEloPredictor)
    assert path == ckpt_path.resolve()


def test_load_gap_weighted_model_mismatched_ratings_mean_raises(tmp_path, monkeypatch):
    ckpt_path = tmp_path / "best_model.pth"
    _write_tiny_checkpoint(ckpt_path, seed=2, ratings_mean=api.RATINGS_MEAN + 1)
    monkeypatch.setenv("RATINGNET_CHECKPOINT_GAP_WEIGHTED", str(ckpt_path))

    with pytest.raises(ValueError, match="ratings_mean"):
        api._load_gap_weighted_model(torch.device("cpu"))


def test_load_gap_weighted_model_missing_checkpoint_returns_none(monkeypatch):
    monkeypatch.delenv("RATINGNET_CHECKPOINT_GAP_WEIGHTED", raising=False)
    monkeypatch.setattr(api, "GAP_WEIGHTED_CHECKPOINT_REL", api.Path("definitely") / "not" / "there.pth")

    assert api._load_gap_weighted_model(torch.device("cpu")) is None


def test_discover_gap_weighted_checkpoint_bad_override_raises(monkeypatch):
    monkeypatch.setenv("RATINGNET_CHECKPOINT_GAP_WEIGHTED", "/no/such/file.pth")

    with pytest.raises(FileNotFoundError):
        api._discover_gap_weighted_checkpoint()

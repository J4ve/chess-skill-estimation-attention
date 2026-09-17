import json
import math
from pathlib import Path

import numpy as np
import pytest

import api
import detector
from suspicion_labels import HIGHLY_UNUSUAL, TYPICAL, UNUSUAL

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "detector_parity_synthetic_caught.json"


@pytest.fixture(scope="module")
def det():
    return detector.load_detector()


@pytest.fixture(scope="module")
def parity_case():
    return json.loads(FIXTURE.read_text())


def test_provenance_matches_feature_order(det):
    assert det.provenance["features"] == detector.FEATURES
    assert det.provenance["score_id"] == detector.SCORE_ID
    assert det.feat_mu.shape == (17,)
    assert det.feat_sd.shape == (17,)


def test_score_matches_thesis_hpc_code_path(det, parity_case):
    white, black = detector.score_game(
        det,
        np.asarray(parity_case["per_move_preds"], dtype=np.float32),
        np.asarray(parity_case["attention"], dtype=np.float32),
        parity_case["moves_uci"],
        parity_case["clock_seconds"],
        parity_case["white_baseline"],
        parity_case["black_baseline"],
    )
    assert white == pytest.approx(parity_case["expected_white_score"], abs=1e-4)
    assert black == pytest.approx(parity_case["expected_black_score"], abs=1e-4)


def test_build_features_shape_and_side_marking(parity_case):
    preds = np.asarray(parity_case["per_move_preds"], dtype=np.float32)
    raw = detector.build_features(
        preds, None, parity_case["moves_uci"], parity_case["clock_seconds"], detector.BLACK, 1500.0
    )
    n = min(len(preds), detector.MAX_PLIES)
    assert raw.shape == (n, len(detector.FEATURES))
    col = {f: raw[:, i] for i, f in enumerate(detector.FEATURES)}
    assert list(col["is_suspect_move"][:4]) == [0.0, 1.0, 0.0, 1.0]
    assert col["r_hat_suspect"][0] == pytest.approx(preds[0, 1])
    assert col["dev_signed"][0] == pytest.approx(preds[0, 1] - 1500.0, abs=1e-3)
    # clock_delta needs a same-side clock two plies back
    assert np.isnan(col["clock_delta"][:2]).all()
    assert list(col["clock_delta_missing"][:2]) == [1.0, 1.0]
    assert col["ply_pos"][3] == pytest.approx(0.03)
    # uniform attention when none is given: alpha * n == 1
    assert np.allclose(col["alpha_n"], 1.0)


def test_cheap_features_capture_check_and_material():
    # 1. e4 d5 2. exd5 (capture) ... 3. Bb5+ (check)
    moves = ["e2e4", "d7d5", "e4d5", "g8f6", "f1b5"]
    cap, chk, mat = detector._cheap_features(moves, detector.WHITE)
    assert list(cap) == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert list(chk) == [0.0, 0.0, 0.0, 0.0, 1.0]
    # material before the move, White's point of view: +1 once the pawn is taken
    assert list(mat) == [0.0, 0.0, 0.0, 1.0, 1.0]
    _, _, mat_black = detector._cheap_features(moves, detector.BLACK)
    assert list(mat_black) == [0.0, 0.0, 0.0, -1.0, -1.0]


def test_score_is_a_probability_and_only_first_100_plies_count(det, parity_case):
    preds = np.asarray(parity_case["per_move_preds"], dtype=np.float32)
    raw = detector.build_features(
        preds, np.asarray(parity_case["attention"]), parity_case["moves_uci"],
        parity_case["clock_seconds"], detector.WHITE, 1100.0,
    )
    score = detector.score_side(det, raw)
    assert 0.0 <= score <= 1.0 and math.isfinite(score)
    assert raw.shape[0] <= detector.MAX_PLIES


DETECTOR_CUTOFFS = {
    "score": detector.SCORE_ID,
    "overall": {"n_sides": 6000, "p75": 0.6, "p95": 0.8},
    "by_time_control": {"bullet": {"p75": 0.55, "p95": 0.75, "use_overall": False}},
}


def test_labels_for_detector_scores_use_its_cutoffs():
    white, black, used = api._labels_for_scores(DETECTOR_CUTOFFS, 0.2, 0.9, "bullet")
    assert (white, black) == (TYPICAL, HIGHLY_UNUSUAL)
    assert used["score"] == detector.SCORE_ID
    assert (used["p75"], used["p95"], used["source"]) == (0.55, 0.75, "time_control")
    assert used["provisional_games"] == 3000

    white, black, used = api._labels_for_scores(DETECTOR_CUTOFFS, 0.6, 0.8, None)
    assert (white, black) == (UNUSUAL, UNUSUAL)
    assert used["source"] == "overall"


def test_labels_omitted_without_cutoffs():
    assert api._labels_for_scores(None, 0.5, 0.5, "blitz") == (None, None, None)


def test_cutoffs_file_for_another_score_is_rejected(tmp_path):
    path = tmp_path / "cutoffs.json"
    path.write_text(json.dumps({**DETECTOR_CUTOFFS, "score": "s_att"}))
    assert api._load_cutoffs(path, detector.SCORE_ID) is None
    assert api._load_cutoffs(path, "s_att") is not None
    assert api._load_cutoffs(tmp_path / "missing.json", "s_att") is None


def test_shipped_cutoffs_files_name_their_scores():
    for path, expected in (
        (api.SUSPICION_CUTOFFS_PATH, detector.SCORE_ID),
        (api.COMPUTED_CUTOFFS_PATH, api.COMPUTED_SCORE_ID),
    ):
        if path.exists():
            assert json.loads(path.read_text())["score"] == expected

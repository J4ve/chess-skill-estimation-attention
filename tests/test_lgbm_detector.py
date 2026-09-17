import json
import math
from pathlib import Path

import numpy as np
import pytest

import detector
import lgbm_detector

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def model():
    return lgbm_detector.load_lgbm_detector()


@pytest.fixture(scope="module")
def parity_case():
    case = json.loads((FIXTURES / "detector_parity_synthetic_caught.json").read_text())
    expected = np.load(FIXTURES / "method_parity_synthetic_caught.npz")["expected_lgbm"]
    return case, expected


def test_model_matches_provenance(model):
    assert model.provenance["score_id"] == lgbm_detector.SCORE_ID
    assert len(model.feature_names) == 97
    assert model.feature_names == sorted(model.feature_names)
    assert len(model.trees) == model.provenance["refit"]["best_iteration"]
    assert model.sigmoid == 1.0


def test_pooled_features_are_exactly_the_trained_columns(model, parity_case):
    case, _ = parity_case
    pp = detector.per_ply_arrays(
        np.asarray(case["per_move_preds"], dtype=np.float32), np.asarray(case["attention"]),
        case["moves_uci"], case["clock_seconds"], detector.WHITE, case["white_baseline"],
    )
    pooled = lgbm_detector.pool_game(pp, len(pp["ply_idx"]))
    assert sorted(pooled) == model.feature_names


def test_score_matches_thesis_hpc_code_path(model, parity_case):
    case, expected = parity_case
    white, black = lgbm_detector.score_game(
        model,
        np.asarray(case["per_move_preds"], dtype=np.float32),
        np.asarray(case["attention"], dtype=np.float32),
        case["moves_uci"],
        case["clock_seconds"],
        case["white_baseline"],
        case["black_baseline"],
    )
    assert white == pytest.approx(expected[0], abs=1e-4)
    assert black == pytest.approx(expected[1], abs=1e-4)
    assert 0.0 <= white <= 1.0 and math.isfinite(white)


# A two-tree model: tree 0 splits feature 0 at 0.5 with NaN going left
# (decision_type 2 | 8 = 10: default_left, missing type NaN); tree 1 splits
# feature 1 at 1.5 with missing type None (NaN treated as 0.0, decision_type 0).
TINY_MODEL = """tree
version=v4
num_class=1
num_tree_per_iteration=1
max_feature_idx=1
objective=binary sigmoid:2
feature_names=a b

Tree=0
num_leaves=2
num_cat=0
split_feature=0
threshold=0.5
decision_type=10
left_child=-1
right_child=-2
leaf_value=-1 1
shrinkage=1

Tree=1
num_leaves=2
num_cat=0
split_feature=1
threshold=1.5
decision_type=0
left_child=-1
right_child=-2
leaf_value=0.25 -0.25
shrinkage=1

end of trees
"""


def _prob(raw, sigmoid=2.0):
    return 1.0 / (1.0 + math.exp(-sigmoid * raw))


def test_tree_walk_follows_lightgbm_missing_value_rules():
    trees, names, sigmoid = lgbm_detector.load_model_text(TINY_MODEL)
    model = lgbm_detector.LgbmDetector(trees=trees, feature_names=names, sigmoid=sigmoid, provenance={})
    assert names == ["a", "b"] and sigmoid == 2.0
    assert model.predict([0.0, 0.0]) == pytest.approx(_prob(-1 + 0.25))
    assert model.predict([0.5, 1.5]) == pytest.approx(_prob(-1 + 0.25))  # <= goes left
    assert model.predict([0.6, 2.0]) == pytest.approx(_prob(1 - 0.25))
    # NaN with missing type NaN takes the default (left) branch.
    assert model.predict([float("nan"), 2.0]) == pytest.approx(_prob(-1 - 0.25))
    # NaN with missing type None is compared as 0.0 (left of 1.5).
    assert model.predict([1.0, float("nan")]) == pytest.approx(_prob(1 + 0.25))


def test_categorical_models_are_rejected():
    with pytest.raises(ValueError):
        lgbm_detector.load_model_text(TINY_MODEL.replace("decision_type=0", "decision_type=1"))

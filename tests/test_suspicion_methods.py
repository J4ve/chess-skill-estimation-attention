import json

import api
import cnn_bilstm_detector
import detector
import lgbm_detector
from suspicion_labels import HIGHLY_UNUSUAL, TYPICAL, UNUSUAL


def _cutoffs(score, p75, p95):
    return {"score": score, "overall": {"n_sides": 1000, "p75": p75, "p95": p95}, "by_time_control": {}}


def test_method_ids_are_the_four_scores_with_the_per_move_detector_default():
    assert api.METHOD_IDS == ("s_att", lgbm_detector.SCORE_ID, detector.SCORE_ID, cnn_bilstm_detector.SCORE_ID)
    assert api.DEFAULT_METHOD == detector.SCORE_ID


def test_combined_cutoffs_file_drops_entries_naming_another_score(tmp_path):
    path = tmp_path / "cutoffs.json"
    path.write_text(json.dumps({"methods": {
        "s_att": _cutoffs("s_att", 100.0, 200.0),
        "lgbm_a0g": _cutoffs(detector.SCORE_ID, 0.5, 0.8),  # crossed: must be ignored
        detector.SCORE_ID: _cutoffs(detector.SCORE_ID, 0.48, 0.82),
    }}))
    loaded = api._load_method_cutoffs(path)
    assert set(loaded) == {"s_att", detector.SCORE_ID}
    assert api._load_method_cutoffs(tmp_path / "missing.json") == {}


def test_shipped_cutoffs_cover_every_method_and_name_their_own_score():
    data = json.loads(api.SUSPICION_CUTOFFS_PATH.read_text())
    assert set(data["methods"]) == set(api.METHOD_IDS)
    for method_id, cutoffs in data["methods"].items():
        assert cutoffs["score"] == method_id
        assert cutoffs["overall"]["p75"] < cutoffs["overall"]["p95"]
    assert len(api._load_method_cutoffs(api.SUSPICION_CUTOFFS_PATH)) == len(api.METHOD_IDS)


def test_suspicion_methods_label_each_score_against_its_own_cutoffs(monkeypatch):
    monkeypatch.setattr(api, "METHOD_CUTOFFS", {
        "s_att": _cutoffs("s_att", 100.0, 200.0),
        "lgbm_a0g": _cutoffs("lgbm_a0g", 0.5, 0.8),
        detector.SCORE_ID: _cutoffs(detector.SCORE_ID, 0.3, 0.6),
    })
    scores = {
        "s_att": (150.0, 250.0),
        "lgbm_a0g": (0.4, 0.9),
        detector.SCORE_ID: (0.4, 0.9),
        cnn_bilstm_detector.SCORE_ID: None,  # model not loaded
    }
    methods = api._suspicion_methods(scores, "blitz")
    assert list(methods) == list(api.METHOD_IDS)

    s_att = methods["s_att"]
    assert s_att["available"] and s_att["scale"] == "rating_points"
    assert (s_att["white_label"], s_att["black_label"]) == (UNUSUAL, HIGHLY_UNUSUAL)
    assert s_att["cutoffs_used"]["score"] == "s_att"

    # Same score, different cutoffs, different label.
    assert methods["lgbm_a0g"]["white_label"] == TYPICAL
    assert methods[detector.SCORE_ID]["white_label"] == UNUSUAL
    assert methods[detector.SCORE_ID]["scale"] == "unit"

    missing = methods[cnn_bilstm_detector.SCORE_ID]
    assert missing == {
        "available": False, "scale": "unit", "white_score": None, "black_score": None,
        "white_label": None, "black_label": None, "cutoffs_used": None,
    }


def test_method_without_cutoffs_still_reports_its_score(monkeypatch):
    monkeypatch.setattr(api, "METHOD_CUTOFFS", {})
    methods = api._suspicion_methods({m: (0.1234567, 0.7654321) for m in api.METHOD_IDS}, None)
    entry = methods[lgbm_detector.SCORE_ID]
    assert entry["available"] and entry["white_score"] == 0.123457
    assert entry["white_label"] is None and entry["cutoffs_used"] is None

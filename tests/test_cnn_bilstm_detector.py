from pathlib import Path

import numpy as np
import pytest
import torch

import cnn_bilstm_detector

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "method_parity_synthetic_caught.npz"


@pytest.fixture(scope="module")
def det():
    return cnn_bilstm_detector.load_cnn_bilstm_detector()


@pytest.fixture(scope="module")
def case():
    return np.load(FIXTURE)


def test_provenance_names_the_score(det):
    assert det.provenance["score_id"] == cnn_bilstm_detector.SCORE_ID
    assert det.provenance["split"] == "v2"


def test_score_matches_thesis_hpc_code_path(det, case):
    # The fixture holds the frozen CNN trunk's float16 embeddings for the
    # synthetic_caught sample, so this needs no rating checkpoint.
    white, black = cnn_bilstm_detector.score_embeddings(
        det, torch.from_numpy(case["emb"].astype(np.float32)), torch.from_numpy(case["clocks_z"])
    )
    assert white == pytest.approx(float(case["expected_cnn_bilstm"][0]), abs=1e-4)
    assert black == pytest.approx(float(case["expected_cnn_bilstm"][1]), abs=1e-4)


def test_padding_does_not_leak_into_the_score(det, case):
    emb = torch.from_numpy(case["emb"].astype(np.float32))[:40]
    clk = torch.from_numpy(case["clocks_z"])[:40]
    # Extra clock entries beyond the embeddings must not change anything.
    a = cnn_bilstm_detector.score_embeddings(det, emb, clk)
    b = cnn_bilstm_detector.score_embeddings(det, emb, torch.cat([clk, torch.full((10,), 9.0)]))
    assert a == pytest.approx(b, abs=1e-7)
    assert all(0.0 <= s <= 1.0 for s in a)


def test_single_ply_game_pools_all_plies_for_black(det, case):
    # Black has no ply in a one-ply game; the batcher then pools every valid ply.
    emb = torch.from_numpy(case["emb"].astype(np.float32))[:1]
    white, black = cnn_bilstm_detector.score_embeddings(det, emb, torch.from_numpy(case["clocks_z"])[:1])
    assert white == pytest.approx(black, abs=1e-7)


def test_embed_positions_uses_the_float16_round_trip():
    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i, (cin, cout) in enumerate(((12, 32), (32, 64), (64, 128), (128, 256)), start=1):
                setattr(self, f"conv{i}", torch.nn.Conv2d(cin, cout, 3, padding=1))
                setattr(self, f"bn{i}", torch.nn.BatchNorm2d(cout))
            self.pool = torch.nn.AvgPool2d(2, 2)

    torch.manual_seed(0)
    trunk = Trunk().eval()
    boards = torch.rand(1, 120, 12, 8, 8)
    emb = cnn_bilstm_detector.embed_positions(trunk, boards)
    assert emb.shape == (100, 256) and emb.dtype == torch.float32
    assert torch.equal(emb, emb.half().float())

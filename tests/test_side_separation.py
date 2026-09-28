"""Tests for the White/Black separation arms of chess_rating_net.

Two things matter here. First, that the new loss terms and the new head compute
what the pre-registration says they compute. Second, and more important for the
study, that leaving the flags off changes nothing: the served model and every
published number were produced by the unflagged path, so the unflagged path has
to stay parameter-for-parameter and output-for-output what it was.
"""
import torch
import torch.nn as nn

from chess_rating_net import (
    GAP_WEIGHT_CAP,
    GAP_WEIGHT_SCALE,
    ChessEloPredictor,
    gap_weights,
    rating_gap_abs_error,
    separation_training_loss,
    side_abs_error,
)


# --- Arm A: the mismatch weighting ----------------------------------------


def test_gap_weight_is_one_for_an_evenly_matched_game():
    w = gap_weights(torch.tensor([1500.0]), torch.tensor([1500.0]))
    assert torch.allclose(w, torch.tensor([1.0]))


def test_gap_weight_is_four_at_the_preregistered_300_point_threshold():
    """The scale is chosen so the wide-gap threshold lands on 4x. Pre-registered."""
    w = gap_weights(torch.tensor([1800.0]), torch.tensor([1500.0]))
    assert torch.allclose(w, torch.tensor([4.0]))


def test_gap_weight_is_symmetric_in_the_two_sides():
    a = gap_weights(torch.tensor([1800.0]), torch.tensor([1500.0]))
    b = gap_weights(torch.tensor([1500.0]), torch.tensor([1800.0]))
    assert torch.allclose(a, b)


def test_gap_weight_is_capped_for_the_rare_extreme_gaps():
    """The widest real gap in the test partition is 1,837 points."""
    w = gap_weights(torch.tensor([3000.0]), torch.tensor([1000.0]))
    assert torch.allclose(w, torch.tensor([GAP_WEIGHT_CAP]))
    assert 1.0 + 2000.0 / GAP_WEIGHT_SCALE > GAP_WEIGHT_CAP  # the cap really bit


def test_gap_weight_grows_with_the_gap():
    gaps = torch.tensor([0.0, 50.0, 100.0, 300.0, 600.0])
    w = gap_weights(torch.full_like(gaps, 1500.0) + gaps, torch.full_like(gaps, 1500.0))
    assert torch.all(w[1:] > w[:-1])


def test_gap_weighted_loss_is_the_weighted_mean_of_the_per_game_errors():
    preds = torch.tensor([[1500.0, 1500.0], [1600.0, 1400.0]])
    targets = torch.tensor([[1510.0, 1490.0], [1900.0, 1300.0]])
    white = targets[:, 0]
    black = targets[:, 1]

    total, side, diff = separation_training_loss(
        preds, targets, white, black, gap_weighting=True
    )
    assert diff is None

    per_game = side_abs_error(preds, targets)          # [10.0, 200.0]
    w = gap_weights(white, black)                       # [1.2, 7.0]
    expected = (w * per_game).sum() / w.sum()
    assert torch.allclose(side, expected)
    assert torch.allclose(total, expected)


def test_gap_weighting_pulls_the_loss_toward_the_mismatched_game():
    """The point of the arm: a wide-gap game must count for more than an even one."""
    preds = torch.tensor([[1500.0, 1500.0], [1500.0, 1500.0]])
    targets = torch.tensor([[1505.0, 1495.0], [1900.0, 1300.0]])
    unweighted, _, _ = separation_training_loss(preds, targets)
    weighted, _, _ = separation_training_loss(
        preds, targets, targets[:, 0], targets[:, 1], gap_weighting=True
    )
    assert weighted > unweighted


def test_gap_weighting_without_elos_is_an_error_rather_than_a_silent_fallback():
    preds = torch.zeros(2, 2)
    try:
        separation_training_loss(preds, preds, gap_weighting=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError when the Elos are missing")


# --- Arm B: the difference term -------------------------------------------


def test_difference_error_equals_the_true_gap_for_a_mirrored_prediction():
    """The mirrored solution this experiment exists to break: both sides equal."""
    preds = torch.tensor([[1500.0, 1500.0]])
    targets = torch.tensor([[1800.0, 1500.0]])
    assert torch.allclose(rating_gap_abs_error(preds, targets), torch.tensor([300.0]))


def test_difference_error_is_zero_when_the_predicted_gap_is_right():
    preds = torch.tensor([[1700.0, 1400.0]])
    targets = torch.tensor([[1800.0, 1500.0]])  # same 300 gap, both sides off by 100
    assert torch.allclose(rating_gap_abs_error(preds, targets), torch.tensor([0.0]))


def test_difference_error_is_signed_so_swapping_the_sides_is_not_free():
    preds = torch.tensor([[1500.0, 1800.0]])
    targets = torch.tensor([[1800.0, 1500.0]])
    assert torch.allclose(rating_gap_abs_error(preds, targets), torch.tensor([600.0]))


def test_difference_term_enters_the_total_with_its_weight():
    preds = torch.tensor([[1500.0, 1500.0]])
    targets = torch.tensor([[1800.0, 1500.0]])
    total, side, diff = separation_training_loss(preds, targets, diff_loss_weight=0.5)
    assert torch.allclose(side, torch.tensor(150.0))   # (300 + 0) / 2
    assert torch.allclose(diff, torch.tensor(300.0))
    assert torch.allclose(total, torch.tensor(150.0 + 0.5 * 300.0))


def test_lambda_one_half_gives_the_two_objectives_equal_per_output_gradient():
    """The pre-registered justification for lambda = 0.5, checked by autograd.

    The side term averages over both outputs of every game, so each output sees
    1/(2B); the difference term averages over games, so each output sees
    lambda/B. At lambda = 0.5 those are equal and neither objective dominates.
    """
    targets = torch.tensor([[1800.0, 1500.0], [1400.0, 1700.0]])

    def grad_of(weight, only):
        preds = torch.tensor([[1500.0, 1600.0], [1550.0, 1500.0]], requires_grad=True)
        total, side, diff = separation_training_loss(preds, targets, diff_loss_weight=weight)
        (diff * weight if only == "diff" else side).backward()
        return preds.grad.abs()

    assert torch.allclose(grad_of(0.5, "side"), grad_of(0.5, "diff"))


def test_difference_term_is_off_at_weight_zero():
    preds = torch.tensor([[1500.0, 1500.0]])
    targets = torch.tensor([[1800.0, 1500.0]])
    total, side, diff = separation_training_loss(preds, targets, diff_loss_weight=0.0)
    assert diff is None
    assert torch.allclose(total, side)


def test_unflagged_side_loss_equals_plain_l1():
    """With both options off the side term is exactly what nn.L1Loss reports."""
    torch.manual_seed(0)
    preds = torch.randn(8, 2) * 300 + 1500
    targets = torch.randn(8, 2) * 300 + 1500
    total, side, diff = separation_training_loss(preds, targets)
    assert diff is None
    assert torch.allclose(total, nn.L1Loss()(preds, targets))


# --- Arm B: the head ------------------------------------------------------


def _forward(model, seed=0):
    torch.manual_seed(seed)
    model.eval()
    positions = torch.randn(3, 5, 12, 8, 8)
    clocks = torch.randn(3, 5)
    lengths = torch.tensor([5, 4, 3], dtype=torch.int)
    with torch.no_grad():
        return model(positions, clocks, lengths)


def test_separate_heads_keeps_the_baseline_output_shape():
    model = ChessEloPredictor(separate_heads=True)
    per_move, last_step = _forward(model)
    assert per_move.shape == (3, 5, 2)
    assert last_step.shape == (3, 2)


def test_separate_heads_gives_each_side_its_own_parameters():
    """A split of the final Linear alone would be a reparameterization, so the
    branch has to start at fc1. Check both layers are per-side and that the
    shared head is gone."""
    model = ChessEloPredictor(separate_heads=True)
    names = dict(model.named_parameters())
    for expected in (
        "fc1_white.weight", "fc1_black.weight", "fc2_white.weight", "fc2_black.weight",
    ):
        assert expected in names, expected
    assert not any(n.startswith("fc1.") or n.startswith("fc2.") for n in names)
    assert names["fc2_white.weight"].shape == (1, 32)
    assert names["fc1_white.weight"].shape != names["fc2_white.weight"].shape


def test_separate_heads_actually_produces_different_numbers_for_the_two_sides():
    model = ChessEloPredictor(separate_heads=True)
    _, last_step = _forward(model)
    assert not torch.allclose(last_step[:, 0], last_step[:, 1])


def test_default_model_is_unchanged_by_the_new_flag():
    """The served architecture: same parameter names and shapes as before."""
    model = ChessEloPredictor()
    names = dict(model.named_parameters())
    assert "fc1.weight" in names and "fc2.weight" in names
    assert names["fc2.weight"].shape == (2, 32)
    assert not any(n.startswith("fc1_") or n.startswith("fc2_") for n in names)
    assert model.separate_heads is False


def test_a_default_checkpoint_still_loads_strictly_into_a_default_model():
    """score_test_split.py loads with strict=True, so the key set must not drift."""
    a = ChessEloPredictor(use_attention=True)
    b = ChessEloPredictor(use_attention=True)
    b.load_state_dict(a.state_dict(), strict=True)


def test_a_separate_head_checkpoint_refuses_to_load_into_the_shared_model():
    """An architecture mix-up has to be loud, not a silently wrong evaluation."""
    shared = ChessEloPredictor()
    split = ChessEloPredictor(separate_heads=True)
    try:
        shared.load_state_dict(split.state_dict(), strict=True)
    except RuntimeError:
        return
    raise AssertionError("expected a strict-load failure across the two head layouts")


def test_separate_heads_leaves_the_shared_trunk_untouched():
    shared = ChessEloPredictor(use_attention=True)
    split = ChessEloPredictor(use_attention=True, separate_heads=True)
    shared_trunk = {
        k: v.shape for k, v in shared.state_dict().items()
        if not k.startswith(("fc1", "fc2"))
    }
    split_trunk = {
        k: v.shape for k, v in split.state_dict().items()
        if not k.startswith(("fc1", "fc2"))
    }
    assert shared_trunk == split_trunk

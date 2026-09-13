"""
Anomaly detection module for the RatingNet extension.

The module computes a per-move deviation signal between the move-by-move
predicted rating curve and an established player baseline, then weights each
deviation by the attention weight produced by the attention module. Both
per-move and aggregated suspicion scores are exposed.

**R_baseline definition (Fix 2).** At inference, ``R_baseline`` is the
player's known pre-game Lichess Glicko-2 rating for the relevant time
control.  It can be supplied explicitly by the caller (e.g. from the PGN
headers or from a ``rating_after_last_game`` field), or — when unavailable —
it falls back to the model's own final-ply predicted rating (see
``api.py``).  Both ``predictions`` and ``baseline`` **must** be on the
original Elo scale (roughly 400–3000+), not the standardized scale used
internally during training (mean ≈ 0, std ≈ 1).  A unit-scale assertion
guards against accidental mixing of the two conventions, which would change
the suspicion score by a factor of ~366.
"""

import torch
import torch.nn as nn


class AnomalyDetector(nn.Module):
    """Per-move and aggregate anomaly scoring.

    Given a per-move rating prediction curve R_t (batch, seq, 2) and a baseline
    rating for each player (batch, 2), the detector computes a deviation
    d_t = |R_t - R_baseline| and an attention-weighted suspicion score
    S = sum_t alpha_t * d_t. The attention weights alpha_t are provided by the
    attention module; when attention is disabled or unavailable a uniform weight
    of 1 / seq_length is used so the aggregate reduces to the mean deviation.

    .. important::

       Both ``predictions`` and ``baseline`` must be on the **original Elo
       scale** (typical range 400–3000+).  Passing standardized values
       (z-scored, roughly −4 to +4) will trigger an ``AssertionError``.  The
       API layer (``api.py``) already de-standardizes predictions before
       calling this module; the guard exists to prevent silent misuse at the
       module level.

    Args:
        normalize: If True, divide the aggregate score by the sum of attention
            weights (softmax already sums to 1, so this is a safety no-op).
    """

    # Minimum plausible |baseline| value on the original Elo scale.
    # Real Elo values are 400–3000+; standardized values are ~±4.
    _ELO_SCALE_FLOOR = 100.0

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        predictions: torch.Tensor,
        baseline: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predictions: (batch, seq, 2) **original-scale** (Elo-point) ratings.
            baseline: (batch, 2) established baseline rating for white and black,
                on the **original Elo scale**.
            attention_weights: Optional (batch, seq) attention weights. If None,
                uniform weights are used.

        Returns:
            Dictionary with keys:
                - "per_move_deviation": (batch, seq, 2)
                - "per_move_weighted": (batch, seq, 2)
                - "white_deviation": (batch, seq)
                - "black_deviation": (batch, seq)
                - "white_score": (batch,) aggregated suspicion for white
                - "black_score": (batch,) aggregated suspicion for black
                - "combined_score": (batch,) sum of white and black scores

        Raises:
            AssertionError: If ``baseline`` values appear to be on the
                standardized scale (all |values| < 100), indicating a
                unit-scale mismatch that would silently produce scores
                ~366× too small.
        """
        # --- Unit-scale guard (Fix 2) ---
        assert baseline.abs().max().item() > self._ELO_SCALE_FLOOR, (
            f"AnomalyDetector expects baseline on the original Elo scale "
            f"(typical range 400–3000), but received max |baseline| = "
            f"{baseline.abs().max().item():.2f}, which looks like standardized "
            f"values.  De-standardize with `baseline * ratings_std + ratings_mean` "
            f"before calling this module."
        )

        batch, seq, _ = predictions.shape
        baseline = baseline.unsqueeze(1)  # (batch, 1, 2)
        per_move_deviation = torch.abs(predictions - baseline)  # (batch, seq, 2)

        if attention_weights is None:
            attention_weights = torch.ones(batch, seq, device=predictions.device, dtype=predictions.dtype) / seq
        else:
            attention_weights = attention_weights.to(predictions.device)
            # Normalize safety check: softmax sums to 1, but if masked we re-normalize.
            if self.normalize:
                weight_sum = attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                attention_weights = attention_weights / weight_sum

        weighted = per_move_deviation * attention_weights.unsqueeze(-1)  # (batch, seq, 2)
        white_score = weighted[:, :, 0].sum(dim=1)  # (batch,)
        black_score = weighted[:, :, 1].sum(dim=1)  # (batch,)

        return {
            "per_move_deviation": per_move_deviation,
            "per_move_weighted": weighted,
            "white_deviation": per_move_deviation[:, :, 0],
            "black_deviation": per_move_deviation[:, :, 1],
            "white_score": white_score,
            "black_score": black_score,
            "combined_score": white_score + black_score,
            "attention_weights": attention_weights,
        }


"""A2 band-normalized features: anchor LightGBM on rating-derived features
z-scored against clean (rate 0) games of the same nominal band.

Plan: analysis/anomaly-extra-arms-plan.md section 5 (A2).
  training bands : stats from rate 0 games in the train split of that band
  A2_own         : withheld band stats from its own rate 0 games, cross-fitted
                   by game-index parity (never a game's own or its twin's values)
  A2_interp      : withheld band stats = mean of the neighbouring training
                   bands' stats (1300 <- 1200, 1400; 1700 <- 1600, 1800)
One model; the variants differ only on withheld-band rows.

  python extra_arm_a2.py [--split v2|grouped] [--smoke]
"""
import argparse

import numpy as np

import extra_arms_common as C
from extra_arm_a0 import lgbm_arm

RATING_PREFIXES = ("d_t_", "alphad_", "r_hat_suspect_", "run_std_", "abs_first_diff_", "abs_second_diff_")
RATING_SCALARS = ("r_hat_final", "run_std_final", "d_t_lastq_mean")
NEIGHBOURS = {1300: (1200, 1400), 1700: (1600, 1800)}


def rating_columns(keys):
    return [j for j, k in enumerate(keys) if k.startswith(RATING_PREFIXES) or k in RATING_SCALARS]


def _stats(Xc):
    mu = np.nanmean(Xc, axis=0)
    sd = np.nanstd(Xc, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu, sd


def make_transform(prefix):
    def transform(X, keys, data, split):
        cols = rating_columns(keys)
        R = X[:, cols]
        clean = data.rate == 0
        seen_bands = sorted(set(data.band[~np.isin(data.band, C.HELDOUT_BANDS)].tolist()))
        stats = {}
        for b in seen_bands:
            m = clean & (data.band == b) & (split == "train")
            stats[b] = _stats(R[m])
        own = R.copy(); interp = R.copy()
        for b in seen_bands:
            m = data.band == b
            mu, sd = stats[b]
            own[m] = (R[m] - mu) / sd
            interp[m] = own[m]
        parity = data.game_index % 2
        for b in C.HELDOUT_BANDS:
            m = data.band == b
            for p in (0, 1):
                src = clean & m & (parity != p)
                mu, sd = _stats(R[src])
                tgt = m & (parity == p)
                own[tgt] = (R[tgt] - mu) / sd
            lo, hi = NEIGHBOURS[b]
            mu = (stats[lo][0] + stats[hi][0]) / 2.0
            sd = (stats[lo][1] + stats[hi][1]) / 2.0
            interp[m] = (R[m] - mu) / sd
        Xo = X.copy(); Xo[:, cols] = own
        Xi = X.copy(); Xi[:, cols] = interp
        # Xo and Xi are identical on every non-withheld-band row, so training
        # on either gives the same model.
        return {f"{prefix}_own": Xo, f"{prefix}_interp": Xi}
    return transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="v2", choices=["v2", "grouped"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    C.require_gate(a.smoke)
    arm = "A2" if a.split == "v2" else "A2g"
    lgbm_arm(arm, a.split, a.smoke, transform=make_transform(arm))


if __name__ == "__main__":
    main()

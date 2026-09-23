#!/usr/bin/env python3
"""Synthesize a Lichess-style remaining-time countdown for the synthetic
anomaly-validation corpus (90,000 bot-vs-bot games).

Why this exists
---------------
`prototype/src/generate_anomaly_corpus.py` writes each ply's *engine compute
time* into the `Clocks` field (`seconds_to_clock(elapsed)`, line 118, where
`elapsed` is wall-clock time for one engine call at line 105-110). The trained
model's clock feature is a *remaining-time countdown* harvested from Lichess
`[%clk ...]` comments (`prototype/src/format_data.py:67-69`) and z-scored with
mean=273 / std=380 (`prototype/src/chess_rating_net.py:63-64,79-81`). The two
quantities are not the same thing, so the corpus is not inference-ready. This
script produces the missing field WITHOUT regenerating any game.

It also fixes a second, separate schema gap: `ChessGamesDataset.__getitem__`
reads `game_info["Time"]` (`chess_rating_net.py:93`) and the anomaly corpus
never writes that key (`generate_anomaly_corpus.py:122-128,177-182`), so
loading a corpus game through the existing Dataset raises KeyError before the
clock feature is ever reached.

Nothing in `prototype/src/` is modified. This script only reads the corpus and
writes new artifacts.

Design summary (full rationale in analysis/anomaly-clock-synthesis-plan.md)
--------------------------------------------------------------------------
1. Each game is assigned a concrete Lichess time control `base+inc`, drawn from
   a mix that reproduces the training corpus's time-control mix.
2. Per-move think times come from a *share-of-remaining-time* model:
       d_k = phi_k * (c / n_k) * X,   X ~ LogNormal(mu=-sigma^2/2, sigma)
   where c is the mover's remaining time, n_k is the expected number of moves
   still to make, phi_k is a game-phase profile (fast opening, slow
   middlegame), and X is a unit-mean multiplicative "this position was
   hard/easy" factor. The phase profile phi_k and the dispersion sigma both
   vary with the game's Maia rating band, using Sigman et al. (2010)'s own
   published band-varying coefficients (see the "Rating-band dependence"
   section below). The parametric form is the fallback; `--params` swaps in
   an empirical per-ply-bin share curve fitted from real Lichess games by
   `--fit-from` (see METHOD_EMPIRICAL).
3. Real clock arithmetic: two independent countdowns (White, Black) that
   alternate into one flat per-ply list, spend time, floor at zero, then gain
   the increment. Increment games can therefore go UP, exactly as real ones do.
4. The clock stream is label-blind by default: it never reads
   `move_is_substituted`, so no timing artifact is planted for the anomaly
   detector to find. `--cheater-timing` opts into a labelled variant for a
   clearly-separate sensitivity experiment.
5. Time-control assignment and the think-time RNG stream are seeded from
   (band, game index) only, never from engine or substitution rate, so game i
   of `1500_stockfish16_r060` gets the same time control and the same random
   draws as game i of `1500_stockfish16_r00`. The band-dependent phase/sigma in
   point 2 is also a function of band alone, so it too is identical across the
   five rate arms of a band. Clocks therefore cannot confound the across-rate
   comparison Chapter 4 reports; they do vary across the nine band groups, which
   is deliberate and is the axis Chapter 4 breaks ROC-AUC down by.

Usage
-----
Self-test (no corpus needed, pure stdlib, runs in seconds):
    python3 synthesize_anomaly_clocks.py --self-test

Sanity report on the default parameters (no corpus needed):
    python3 synthesize_anomaly_clocks.py --simulate 5000 --report-json out.json

Retrofit the real corpus (HPC; needs torch importable to unpickle Positions):
    python3 synthesize_anomaly_clocks.py \
        --corpus ~/Bacsain/thesis2/data/anomaly_corpus --mode sidecar \
        --out ~/Bacsain/thesis2/data/anomaly_clocks

Fit the empirical think-time model from real Lichess pickles, then use it:
    python3 synthesize_anomaly_clocks.py --fit-from /tmp/ratingnet_data_flat \
        --fit-out clock_params.json
    python3 synthesize_anomaly_clocks.py --corpus ... --params clock_params.json

Compare synthesized clocks against real ones (two-sample KS per bucket):
    python3 synthesize_anomaly_clocks.py --compare-real /tmp/ratingnet_data_flat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import statistics
import sys
from pathlib import Path

VERSION = "1.0.0"

METHOD_PARAMETRIC = "parametric-share"
METHOD_EMPIRICAL = "empirical-share"
METHOD_CONSTANT = "constant"

# Normalization constants the trained model uses, for diagnostics only.
# chess_rating_net.py:63-64.
CLOCKS_MEAN = 273.0
CLOCKS_STD = 380.0

# ---------------------------------------------------------------------------
# Verbatim-behaviour copies of the pipeline's own helpers. These are copies on
# purpose: this script must stay importable without torch/chess so it can be
# self-tested anywhere. Any divergence here is a bug.
# ---------------------------------------------------------------------------


def seconds_to_clock(seconds: float) -> str:
    """Same as generate_anomaly_corpus.py:63-67 and Lichess's H:MM:SS form."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def time_to_seconds(time_str: str) -> int:
    """Same as format_data.py:33-36. Integer-only on purpose: the training
    pipeline's version raises on a fractional-second clock, so anything this
    script emits must survive int()."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def categorize_time_control(estimated_duration: int) -> str:
    """Same thresholds as format_data.py:98-109 (29/179/479/1499)."""
    if estimated_duration < 29:
        return "ultrabullet"
    elif estimated_duration < 179:
        return "bullet"
    elif estimated_duration < 479:
        return "blitz"
    elif estimated_duration < 1499:
        return "rapid"
    else:
        return "classical"


def bucket_of(base: int, inc: int) -> str:
    """The pipeline buckets by base + 40*inc (chess_rating_net.py:93-95)."""
    return categorize_time_control(base + 40 * inc)


# ---------------------------------------------------------------------------
# Time-control mix
# ---------------------------------------------------------------------------

# Bucket weights reproduce the training corpus's realized mix as reported in
# analysis/170k-verification.md:683 (8,320 blitz / 6,343 bullet / 2,183 rapid /
# 125 classical / 40 ultrabullet in the 17,014-game test set). That document
# tags the figure [derived] (scaled from a 6,000-game profile), not directly
# measured, so treat it as the best available estimate and re-measure from
# .sampling_summary.json when the real corpus lands.
#
# The split of each bucket into concrete base+inc controls is this script's own
# choice of common Lichess controls; it is an assumption, flagged as such in
# the plan document. Weights are normalized at load time.
DEFAULT_TC_MIX: list[tuple[int, int, float]] = [
    # base, increment, weight        bucket check: base + 40*inc
    (20, 0, 0.0024),   # 20     -> ultrabullet
    (60, 0, 0.2000),   # 60     -> bullet
    (120, 1, 0.1730),  # 160    -> bullet
    (180, 0, 0.2000),  # 180    -> blitz
    (180, 2, 0.1000),  # 260    -> blitz
    (300, 0, 0.1400),  # 300    -> blitz
    (300, 3, 0.0490),  # 420    -> blitz
    (600, 0, 0.0800),  # 600    -> rapid
    (600, 5, 0.0300),  # 800    -> rapid
    (900, 10, 0.0180), # 1300   -> rapid
    (1800, 0, 0.0050), # 1800   -> classical
    (1800, 20, 0.0023),# 2600   -> classical
]


def normalized_mix(mix: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    total = sum(w for _, _, w in mix)
    if total <= 0:
        raise ValueError("time-control mix weights must sum to a positive number")
    return [(b, i, w / total) for b, i, w in mix]


def draw_time_control(rng: random.Random, mix: list[tuple[int, int, float]]) -> tuple[int, int]:
    u = rng.random()
    acc = 0.0
    for base, inc, w in mix:
        acc += w
        if u <= acc:
            return base, inc
    return mix[-1][0], mix[-1][1]


# ---------------------------------------------------------------------------
# Think-time model
# ---------------------------------------------------------------------------

# Per-player move index k (0-based). Piecewise-linear phase profile: the
# opening is fast (book / pattern recall), the early middlegame is the slowest
# stretch, and play speeds up again later. That shape is the qualitative
# finding of Sigman, Etchemendy, Fernandez Slezak and Cecchi (2010),
# "Response Time Distributions in Rapid Chess", Front. Neurosci., who report
# fast opening and endgame moves and significantly longer middlegame moves.
# Their 2.8M-game FICS database is the raw download across 1 to 15 minute
# controls; the pacing result is measured on 3-minute games without increment,
# of which the paper lists 91,340 analyzed. The paper also calls this trend
# "expected" rather than headline. The exact knot values here are this script's
# calibration, not theirs, and are the first thing --fit-from should replace.
#
# Recalibration 2026-08-28 (analysis/anomaly-clock-recalibrate.md, review 1.7).
# The previous knots (0.15, 0.45, 1.00, 1.10, 0.95, 0.85) drained about 88.3%
# of a 180 s budget by White's 40th move at 180+0; Sigman et al. report 74.7%
# (low-rated) to 77.9% (high-rated) at exactly that point in exactly that
# control. The knots below are the old shape scaled down by ~0.68, which lands
# the mid band at about 76.7% consumed after move 40 (see self-test group 10).
# The fast-open / slow-middlegame / late-speed-up shape is preserved.
DEFAULT_PHASE_KNOTS: list[tuple[float, float]] = [
    (0.0, 0.10),
    (3.0, 0.30),
    (7.0, 0.68),
    (20.0, 0.75),
    (35.0, 0.65),
    (50.0, 0.58),
]

# ---------------------------------------------------------------------------
# Rating-band (Maia skill) dependence of the think-time model
# ---------------------------------------------------------------------------
#
# The anomaly corpus's primary design axis is the 9 Maia bands, 1100 to 1900,
# 10,000 games each (analysis/anomaly-corpus-generated.md). Sigman et al. (2010)
# publish band-varying pacing coefficients, so a single phase profile and a
# single dispersion for every band would throw away exactly the axis Chapter 4
# breaks ROC-AUC down by, and exactly the axis the cited paper measured. The
# band therefore threads into the parametric think-time layer through two of
# the paper's own published relations, not invented values. Full derivation and
# before/after probes: analysis/anomaly-clock-recalibrate.md, deliverable 2.
#
# Normalized skill: s = clamp((band - 1100) / 800, 0, 1). s = 0 at band 1100,
# s = 1 at band 1900, s = 0.5 at the 1500 corpus centre. s is a function of
# band alone, so it is identical across the 5 substitution-rate arms of a band
# (game i of 1500_..._r00 and 1500_..._r060 get the same s); it varies only
# across the 9 band groups.
BAND_LO = 1100
BAND_HI = 1900

# (1) Dispersion. Sigman's Results section gives the within-player SD of
# response time as a function of its mean: "for low rated players, regression:
# SD = 0.1 s + 0.91 <RT>, for high rated players SD = 0.6 s + 1.36 <RT>". The
# slope is the dimensionless, dominant term: it is the linear-space coefficient
# of variation of RT once the mean RT is more than a second or two. We read the
# slope directly as the coefficient of variation of the unit-mean multiplier X,
# interpolated linearly in skill:
#     CV_X(s) = 0.91 + (1.36 - 0.91) * s
# The sub-second intercepts (0.1 s, 0.6 s) are dropped on purpose: at the mean
# think times in this corpus's buckets (roughly 1 to 24 s) they move CV_X by at
# most ~0.2, and folding them in requires assuming one reference RT, i.e.
# re-introducing an invented constant. A lognormal whose log-scale sigma
# reproduces that CV is sigma_log(s) = sqrt(ln(1 + CV_X(s)^2)), which runs from
# about 0.78 at band 1100 through 0.91 at band 1500 to 1.02 at band 1900.
# ClockParams.sigma_log (default 0.91) is used only when no band is supplied;
# it equals sigma_log(0.5) so the band and band-agnostic paths agree at the
# corpus centre.
SIGMAN_CV_LO = 0.91
SIGMAN_CV_HI = 1.36

# (2) Phase shape. Sigman: "High rated players amplify the variations of RTs
# during the game: they play faster than lower rated players during the opening
# games, and slower during the middle game." The band adjustment is a small,
# monotone-in-band see-saw on the phase profile: as the band rises the opening
# knots (k < PHASE_OPEN_KMAX) are scaled down and the middlegame knots
# (PHASE_OPEN_KMAX <= k <= PHASE_MID_KMAX) scaled up, by at most
# PHASE_OPEN_GAIN / PHASE_MID_GAIN at the 1900 end, with the mirror image at
# the 1100 end and no change at the centre. Gains calibrated so the fraction of
# a 180 s budget consumed after White's 40th move runs from about 0.747 at band
# 1100 to about 0.783 at band 1900, matching Sigman's own "at move 40" figures
# (low-rated 74.7 +/- 0.4%, high-rated 77.9 +/- 0.3%) to within a point.
PHASE_OPEN_GAIN = 0.11
PHASE_MID_GAIN = 0.06
PHASE_OPEN_KMAX = 7
PHASE_MID_KMAX = 45


def band_skill(band: "int | None") -> float:
    """Normalized skill scalar in [0, 1]; 0.5 when the band is unknown."""
    if band is None:
        return 0.5
    return min(1.0, max(0.0, (band - BAND_LO) / float(BAND_HI - BAND_LO)))


# Per-player move-index bin lower edges, used by the empirical fit.
PLY_BIN_EDGES: list[int] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def bin_index(k: int) -> int:
    idx = 0
    for i, edge in enumerate(PLY_BIN_EDGES):
        if k >= edge:
            idx = i
    return idx


class ClockParams:
    """All tunables in one place, JSON-serializable, overridable from a file."""

    def __init__(
        self,
        method: str = METHOD_PARAMETRIC,
        sigma_log: float = 0.91,
        n_expected: float = 40.0,
        n_floor: float = 12.0,
        kappa: float = 0.45,
        x_lo: float = 0.08,
        x_hi: float = 8.0,
        min_move_s: float = 0.1,
        floor_s: float = 0.1,
        report_after_increment: bool = True,
        phase_knots: list[tuple[float, float]] | None = None,
        tc_mix: list[tuple[int, int, float]] | None = None,
        empirical: dict | None = None,
        cheater_timing: dict | None = None,
    ):
        self.method = method
        # Band-agnostic log-dispersion of the per-move multiplier. Used only
        # when synthesize_game_clocks is called without a band; the default
        # equals the Sigman mid-band (1500) value, see sigma_log_for_band.
        self.sigma_log = sigma_log
        self.n_expected = n_expected
        self.n_floor = n_floor
        self.kappa = kappa
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.min_move_s = min_move_s
        self.floor_s = floor_s
        # Lichess credits the increment after a move is played. Whether the
        # [%clk] value printed for that move already includes it is a one-line
        # convention that must be confirmed against a real PGN; the difference
        # is a constant offset of at most one increment (3s / 380 = 0.008 in
        # normalized units for a typical blitz increment).
        self.report_after_increment = report_after_increment
        self.phase_knots = list(phase_knots) if phase_knots else list(DEFAULT_PHASE_KNOTS)
        self.tc_mix = normalized_mix(list(tc_mix) if tc_mix else list(DEFAULT_TC_MIX))
        # empirical: {bucket: {"share_median": [...], "share_sigma": [...]}}
        self.empirical = empirical or {}
        # Optional, non-default: a labelled cheater timing profile.
        self.cheater_timing = cheater_timing or {"share_scale": 0.45, "sigma_scale": 0.35}

    def phase(self, k: int) -> float:
        knots = self.phase_knots
        if k <= knots[0][0]:
            return knots[0][1]
        if k >= knots[-1][0]:
            return knots[-1][1]
        for (k0, v0), (k1, v1) in zip(knots, knots[1:]):
            if k0 <= k <= k1:
                if k1 == k0:
                    return v1
                t = (k - k0) / (k1 - k0)
                return v0 + t * (v1 - v0)
        return knots[-1][1]

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "method": self.method,
            "sigma_log": self.sigma_log,
            "n_expected": self.n_expected,
            "n_floor": self.n_floor,
            "kappa": self.kappa,
            "x_lo": self.x_lo,
            "x_hi": self.x_hi,
            "min_move_s": self.min_move_s,
            "floor_s": self.floor_s,
            "report_after_increment": self.report_after_increment,
            "phase_knots": [list(p) for p in self.phase_knots],
            "tc_mix": [list(t) for t in self.tc_mix],
            "empirical": self.empirical,
            "cheater_timing": self.cheater_timing,
        }

    @staticmethod
    def from_dict(d: dict) -> "ClockParams":
        return ClockParams(
            method=d.get("method", METHOD_PARAMETRIC),
            sigma_log=d.get("sigma_log", 0.91),
            n_expected=d.get("n_expected", 40.0),
            n_floor=d.get("n_floor", 12.0),
            kappa=d.get("kappa", 0.45),
            x_lo=d.get("x_lo", 0.08),
            x_hi=d.get("x_hi", 8.0),
            min_move_s=d.get("min_move_s", 0.1),
            floor_s=d.get("floor_s", 0.1),
            report_after_increment=d.get("report_after_increment", True),
            phase_knots=[tuple(p) for p in d["phase_knots"]] if d.get("phase_knots") else None,
            tc_mix=[tuple(t) for t in d["tc_mix"]] if d.get("tc_mix") else None,
            empirical=d.get("empirical"),
            cheater_timing=d.get("cheater_timing"),
        )


def sigma_log_for_band(params: ClockParams, band: "int | None") -> float:
    """Log-scale dispersion of the per-move multiplier for this band.

    Interpolates Sigman et al.'s low/high-rated RT coefficient of variation
    (slopes 0.91 and 1.36) linearly in skill and converts it to a lognormal
    sigma. Falls back to params.sigma_log (the band-agnostic knob) when the
    band is unknown; params.sigma_log defaults to this function's own s = 0.5
    value, so the two paths agree at the corpus centre.
    """
    if band is None:
        return params.sigma_log
    s = band_skill(band)
    cv = SIGMAN_CV_LO + (SIGMAN_CV_HI - SIGMAN_CV_LO) * s
    return math.sqrt(math.log(1.0 + cv * cv))


def phase_for_band(params: ClockParams, k: int, band: "int | None") -> float:
    """phi_k with Sigman's rating-dependent see-saw applied.

    High band (s -> 1): opening scaled down, middlegame scaled up (the paper's
    "faster during the opening ... slower during the middle game"). Low band
    (s -> 0): the mirror image. Centre band: unchanged. Monotone in band.
    """
    phi = params.phase(k)
    if band is None:
        return phi
    tilt = 2.0 * band_skill(band) - 1.0  # -1 at band 1100, +1 at band 1900
    if k < PHASE_OPEN_KMAX:
        return phi * (1.0 - PHASE_OPEN_GAIN * tilt)
    if k <= PHASE_MID_KMAX:
        return phi * (1.0 + PHASE_MID_GAIN * tilt)
    return phi


def _share_for_move(
    params: ClockParams, bucket: str, k: int, rng: random.Random, band: "int | None" = None
) -> float:
    """Fraction of the mover's remaining time to spend on this move.

    `band` is the game's Maia rating band (1100..1900) or None. It modulates the
    parametric layer only: the empirical layer carries its own per-bucket,
    per-bin dispersion fitted from real games, which has no band coordinate.
    """
    if params.method == METHOD_EMPIRICAL and bucket in params.empirical:
        table = params.empirical[bucket]
        b = min(bin_index(k), len(table["share_median"]) - 1)
        med = table["share_median"][b]
        sig = table["share_sigma"][b]
        x = math.exp(rng.gauss(0.0, sig)) if sig > 0 else 1.0
        x = min(max(x, params.x_lo), params.x_hi)
        return med * x
    # Parametric fallback. mu = -sigma^2/2 makes E[X] = 1, so the median share
    # is slightly below phi/n and the mean share is exactly phi/n. Both phi and
    # sigma depend on the band (Sigman et al. 2010's published coefficients).
    sigma = sigma_log_for_band(params, band)
    mu = -0.5 * sigma * sigma
    x = math.exp(rng.gauss(mu, sigma))
    x = min(max(x, params.x_lo), params.x_hi)
    n_k = max(params.n_floor, params.n_expected - k)
    return phase_for_band(params, k, band) * x / n_k


def synthesize_game_clocks(
    n_plies: int,
    base: int,
    inc: int,
    params: ClockParams,
    rng: random.Random,
    substituted: list[bool] | None = None,
    band: "int | None" = None,
) -> dict:
    """Return the per-ply clock strings plus diagnostics for one game.

    Ply i (0-based) is White's when i is even, matching
    generate_anomaly_corpus.play_one_game: the board starts from
    chess.Board() and Moves[0] is always White's (generate_anomaly_corpus.py
    :96-120).

    `band` is the game's Maia rating band (1100..1900); it feeds the
    band-dependent phase profile and dispersion (Sigman et al. 2010). None
    keeps the band-agnostic mid-band behaviour.
    """
    bucket = bucket_of(base, inc)
    clocks: list[str] = []
    remaining: list[float] = []
    think: list[float] = []
    c = {True: float(base), False: float(base)}  # True = White

    for i in range(n_plies):
        is_white = (i % 2 == 0)
        k = i // 2  # this player's 0-based move index
        cur = c[is_white]

        if params.method == METHOD_CONSTANT:
            d = 0.0
        else:
            share = _share_for_move(params, bucket, k, rng, band)
            if substituted is not None and i < len(substituted) and substituted[i]:
                share *= params.cheater_timing.get("share_scale", 1.0)
            d = share * cur
            d = max(d, params.min_move_s)
            d = min(d, params.kappa * cur)

        after_spend = max(params.floor_s, cur - d)
        actual_spend = cur - after_spend
        # The constant arm is the no-signal control, so it must not accrue the
        # increment either. Crediting it (spend 0, then add inc every move)
        # turned each of the six nonzero-increment entries in DEFAULT_TC_MIX
        # into a monotone base + k*inc ramp: a trivially learnable encoding of
        # ply index, which is the opposite of an ablation.
        eff_inc = 0 if params.method == METHOD_CONSTANT else inc
        c_new = after_spend + eff_inc
        c[is_white] = c_new

        reported = c_new if params.report_after_increment else after_spend
        clocks.append(seconds_to_clock(reported))
        remaining.append(reported)
        think.append(actual_spend)

    return {
        "clocks": clocks,
        "remaining": remaining,
        "think": think,
        "bucket": bucket,
        "time": f"{base}+{inc}",
    }


# ---------------------------------------------------------------------------
# Deterministic per-game seeding
# ---------------------------------------------------------------------------


def stable_seed(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def game_streams(global_seed: int, band: int, game_index: int) -> tuple[random.Random, random.Random]:
    """Two RNGs seeded from (band, game index) only.

    Engine and substitution rate are deliberately excluded so that game i of
    every case directory in a band shares one time control and one think-time
    stream. Clocks then cannot explain any across-rate difference in Chapter 4.
    """
    tc_rng = random.Random(stable_seed(global_seed, "tc", band, game_index))
    tt_rng = random.Random(stable_seed(global_seed, "think", band, game_index))
    return tc_rng, tt_rng


# ---------------------------------------------------------------------------
# Corpus walking
# ---------------------------------------------------------------------------


def parse_case_dir(name: str) -> tuple[int, str, str] | None:
    """'1500_stockfish16_r015' -> (1500, 'stockfish16', 'r015')."""
    parts = name.split("_")
    if len(parts) < 3:
        return None
    if not parts[0].isdigit():
        return None
    return int(parts[0]), "_".join(parts[1:-1]), parts[-1]


def game_index_of(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("game_"):
        return None
    tail = stem[len("game_"):]
    return int(tail) if tail.isdigit() else None


def retrofit_corpus(
    corpus: Path,
    out: Path | None,
    params: ClockParams,
    global_seed: int,
    mode: str,
    limit: int | None,
    label_aware: bool,
) -> dict:
    """Walk case dirs and produce clocks. mode is sidecar | rewrite | inplace."""
    case_dirs = sorted(d for d in corpus.iterdir() if d.is_dir())
    stats = {"cases": 0, "games": 0, "skipped": 0, "buckets": {}, "remaining_sample": []}
    for case in case_dirs:
        meta = parse_case_dir(case.name)
        if meta is None:
            print(f"skip (unparsable case dir): {case.name}", file=sys.stderr)
            continue
        band, engine, rate = meta
        files = sorted(case.glob("game_*.pkl"))
        if limit:
            files = files[:limit]
        if not files:
            continue
        sidecar: dict[str, dict] = {}
        for path in files:
            gi = game_index_of(path)
            if gi is None:
                stats["skipped"] += 1
                continue
            try:
                with open(path, "rb") as f:
                    game = pickle.load(f)
            except ModuleNotFoundError as exc:
                raise SystemExit(
                    f"cannot unpickle {path}: {exc}. The corpus pickles hold torch "
                    f"tensors in 'Positions', so this step must run in an environment "
                    f"where torch imports (conda env ratingnet2 on the HPC)."
                ) from exc
            n_plies = len(game.get("Moves", []))
            if n_plies == 0:
                stats["skipped"] += 1
                continue
            tc_rng, tt_rng = game_streams(global_seed, band, gi)
            base, inc = draw_time_control(tc_rng, params.tc_mix)
            subs = game.get("move_is_substituted") if label_aware else None
            # `band` comes from the case directory name ({band}_{engine}_r{rate})
            # via parse_case_dir; it is constant across a band's five rate arms.
            res = synthesize_game_clocks(n_plies, base, inc, params, tt_rng, subs, band=band)

            record = {
                "Clocks": res["clocks"],
                "Time": res["time"],
                "clock_synthesis": {
                    "script": Path(__file__).name,
                    "version": VERSION,
                    "method": params.method,
                    "seed": global_seed,
                    "band": band,
                    "engine": engine,
                    "rate": rate,
                    "game_index": gi,
                    "base": base,
                    "increment": inc,
                    "bucket": res["bucket"],
                    "label_aware": bool(label_aware),
                },
            }
            stats["buckets"][res["bucket"]] = stats["buckets"].get(res["bucket"], 0) + 1
            stats["games"] += 1
            if len(stats["remaining_sample"]) < 200000:
                stats["remaining_sample"].extend(res["remaining"])

            if mode == "sidecar":
                sidecar[path.name] = record
            else:
                # Never destroy the original field: the engine think-times are
                # the only record of how long each engine call actually took.
                if "Clocks_engine_seconds" not in game:
                    game["Clocks_engine_seconds"] = game.get("Clocks")
                game["Clocks"] = record["Clocks"]
                game["Time"] = record["Time"]
                game["clock_synthesis"] = record["clock_synthesis"]
                dest = path if mode == "inplace" else (out / case.name / path.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(game, f)
                tmp.replace(dest)

        if mode == "sidecar":
            dest_dir = out / case.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            with open(dest_dir / "clocks.json", "w") as f:
                json.dump({"version": VERSION, "params": params.to_dict(), "games": sidecar}, f)
        stats["cases"] += 1
        print(f"case={case.name} games={len(files)} mode={mode}", flush=True)
    return stats


# ---------------------------------------------------------------------------
# Statistics: two-sample KS, quantiles
# ---------------------------------------------------------------------------


def ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov D and asymptotic p-value.

    p uses the standard Q_KS series; with n in the 10^5 range every real
    difference is 'significant', so read D (the effect size), not p.
    """
    if not a or not b:
        return float("nan"), float("nan")
    xa = sorted(a)
    xb = sorted(b)
    na, nb = len(xa), len(xb)
    ia = ib = 0
    d = 0.0
    while ia < na or ib < nb:
        # Advance both samples past every observation equal to the current
        # smallest value, so ties never create a spurious step (an ECDF is
        # evaluated at a value, not at an index).
        if ia < na and ib < nb:
            v = xa[ia] if xa[ia] <= xb[ib] else xb[ib]
        elif ia < na:
            v = xa[ia]
        else:
            v = xb[ib]
        while ia < na and xa[ia] == v:
            ia += 1
        while ib < nb and xb[ib] == v:
            ib += 1
        d = max(d, abs(ia / na - ib / nb))
    en = math.sqrt(na * nb / (na + nb))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.04:
        # The alternating series does not converge for tiny lambda; the limit
        # as lambda -> 0 is p = 1 (the two ECDFs are indistinguishable).
        return d, 1.0
    p = 0.0
    for j in range(1, 101):
        term = 2 * ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        p += term
        if abs(term) < 1e-12:
            break
    return d, max(0.0, min(1.0, p))


def describe(xs: list[float]) -> dict:
    if not xs:
        return {}
    ys = sorted(xs)
    def q(p: float) -> float:
        idx = min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))
        return ys[idx]
    return {
        "n": len(ys),
        "mean": round(statistics.fmean(ys), 3),
        "std": round(statistics.pstdev(ys), 3) if len(ys) > 1 else 0.0,
        "min": round(ys[0], 3),
        "p05": round(q(0.05), 3),
        "p25": round(q(0.25), 3),
        "median": round(q(0.50), 3),
        "p75": round(q(0.75), 3),
        "p95": round(q(0.95), 3),
        "max": round(ys[-1], 3),
    }


# ---------------------------------------------------------------------------
# Reading real Lichess pickles (for --fit-from and --compare-real)
# ---------------------------------------------------------------------------


def read_real_games(directory: Path, limit: int | None = None):
    """Yield (base, inc, [remaining seconds per ply]) from preprocessed games.

    Matches the schema written by preprocess_lichess.py via format_data.parse_game
    (format_data.py:85-95): Clocks are 'H:MM:SS' strings, Time is 'base+inc'.
    """
    files = sorted(directory.glob("*.pkl"))
    if limit:
        files = files[:limit]
    for path in files:
        try:
            with open(path, "rb") as f:
                game = pickle.load(f)
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"cannot unpickle {path}: {exc}. Real corpus pickles hold torch "
                f"tensors; run this in an env where torch imports."
            ) from exc
        tc = game.get("Time") or ""
        if "+" not in tc:
            continue
        try:
            base, inc = (int(x) for x in tc.split("+"))
        except ValueError:
            continue
        try:
            secs = [time_to_seconds(c) for c in game.get("Clocks", [])]
        except (ValueError, IndexError):
            continue
        if not secs:
            continue
        yield base, inc, secs


def think_times_from_remaining(base: int, inc: int, secs: list[float], report_after_increment: bool) -> list[tuple[int, float, float]]:
    """Recover (per-player move index k, remaining-before, think time).

    Under the 'reported value already includes the increment' convention:
        reported_k = max(0, before_k - d_k) + inc
    so d_k = before_k + inc - reported_k, with before_0 = base and
    before_k = reported_{k-1}.
    """
    out = []
    prev = {True: float(base), False: float(base)}
    for i, r in enumerate(secs):
        is_white = (i % 2 == 0)
        k = i // 2
        before = prev[is_white]
        d = before + (inc if report_after_increment else 0) - r
        if d < 0:
            d = 0.0
        out.append((k, before, d))
        prev[is_white] = r if report_after_increment else r + inc
    return out


def fit_params_from_real(directory: Path, limit: int | None, base_params: ClockParams) -> ClockParams:
    """Estimate a per-bucket, per-ply-bin share curve from real games.

    share = think time / remaining time before the move. Stored as the median
    share per bin plus the sd of log(share / median), which is exactly what
    _share_for_move consumes when method == empirical-share.
    """
    acc: dict[str, list[list[float]]] = {}
    n_games = 0
    for base, inc, secs in read_real_games(directory, limit):
        bucket = bucket_of(base, inc)
        acc.setdefault(bucket, [[] for _ in PLY_BIN_EDGES])
        for k, before, d in think_times_from_remaining(base, inc, secs, base_params.report_after_increment):
            if before <= 0:
                continue
            share = d / before
            if share <= 0 or share >= 1:
                continue
            acc[bucket][min(bin_index(k), len(PLY_BIN_EDGES) - 1)].append(share)
        n_games += 1

    empirical: dict[str, dict] = {}
    for bucket, bins in acc.items():
        med, sig = [], []
        for vals in bins:
            if len(vals) < 30:
                med.append(float("nan"))
                sig.append(float("nan"))
                continue
            m = statistics.median(vals)
            logs = [math.log(v / m) for v in vals if v > 0]
            med.append(m)
            sig.append(statistics.pstdev(logs) if len(logs) > 1 else 0.0)
        # Fill unestimated bins by carrying the nearest estimated neighbour.
        for i in range(len(med)):
            if math.isnan(med[i]):
                left = next((med[j] for j in range(i, -1, -1) if not math.isnan(med[j])), None)
                right = next((med[j] for j in range(i, len(med)) if not math.isnan(med[j])), None)
                med[i] = left if left is not None else (right if right is not None else 0.03)
                sig[i] = 0.8
        empirical[bucket] = {"share_median": med, "share_sigma": sig, "n_bins": len(med)}

    fitted = ClockParams.from_dict(base_params.to_dict())
    fitted.method = METHOD_EMPIRICAL
    fitted.empirical = empirical
    print(f"fitted from {n_games} real games; buckets={sorted(empirical)}", flush=True)
    fitted.empirical["_source"] = {"dir": str(directory), "n_games": n_games}
    return fitted


# ---------------------------------------------------------------------------
# Simulation / sanity reporting (no corpus required)
# ---------------------------------------------------------------------------


def simulate(n_games: int, params: ClockParams, global_seed: int, plies: int = 100) -> dict:
    """Synthesize n_games clock traces without touching any corpus file."""
    per_bucket_remaining: dict[str, list[float]] = {}
    per_bucket_think: dict[str, list[float]] = {}
    all_remaining: list[float] = []
    bands = list(range(1100, 2000, 100))
    for i in range(n_games):
        band = bands[i % len(bands)]
        tc_rng, tt_rng = game_streams(global_seed, band, i)
        base, inc = draw_time_control(tc_rng, params.tc_mix)
        res = synthesize_game_clocks(plies, base, inc, params, tt_rng, band=band)
        per_bucket_remaining.setdefault(res["bucket"], []).extend(res["remaining"])
        per_bucket_think.setdefault(res["bucket"], []).extend(res["think"])
        all_remaining.extend(res["remaining"])
    normalized = [(x - CLOCKS_MEAN) / CLOCKS_STD for x in all_remaining]
    return {
        "n_games": n_games,
        "plies_per_game": plies,
        "overall_remaining_seconds": describe(all_remaining),
        "normalized_feature": describe(normalized),
        "model_norm_constants": {"mean": CLOCKS_MEAN, "std": CLOCKS_STD},
        "per_bucket_remaining_seconds": {b: describe(v) for b, v in sorted(per_bucket_remaining.items())},
        "per_bucket_think_seconds": {b: describe(v) for b, v in sorted(per_bucket_think.items())},
        "bucket_share_of_games": {
            b: round(len(v) / plies / n_games, 4) for b, v in sorted(per_bucket_remaining.items())
        },
        "_samples": {b: v for b, v in per_bucket_remaining.items()},
    }


def compare_with_real(real_dir: Path, params: ClockParams, global_seed: int, limit: int | None) -> dict:
    """Two-sample KS between synthesized and real remaining-time distributions,
    per time-control bucket. This is the check that turns 'plausible' from an
    assertion into a measurement. It needs real preprocessed pickles."""
    real_by_bucket: dict[str, list[float]] = {}
    real_lengths: list[int] = []
    n = 0
    for base, inc, secs in read_real_games(real_dir, limit):
        real_by_bucket.setdefault(bucket_of(base, inc), []).extend(secs)
        real_lengths.append(len(secs))
        n += 1
    if n == 0:
        raise SystemExit(f"no usable real games found under {real_dir}")
    mean_plies = int(round(statistics.fmean(real_lengths)))
    sim = simulate(max(200, n), params, global_seed, plies=min(100, mean_plies))
    out = {"real_games": n, "real_mean_plies": mean_plies, "buckets": {}}
    for bucket, real_vals in sorted(real_by_bucket.items()):
        syn_vals = sim["_samples"].get(bucket, [])
        d, p = ks_two_sample(syn_vals, real_vals)
        out["buckets"][bucket] = {
            "ks_D": round(d, 4) if d == d else None,
            "ks_p": p if p == p else None,
            "real": describe(real_vals),
            "synthetic": describe(syn_vals),
        }
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> int:
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    params = ClockParams()

    # 1. Bucket mapping of the shipped mix is what the comment claims.
    expect = {
        (20, 0): "ultrabullet", (60, 0): "bullet", (120, 1): "bullet",
        (180, 0): "blitz", (180, 2): "blitz", (300, 0): "blitz", (300, 3): "blitz",
        (600, 0): "rapid", (600, 5): "rapid", (900, 10): "rapid",
        (1800, 0): "classical", (1800, 20): "classical",
    }
    for (b, i), want in expect.items():
        check(bucket_of(b, i) == want, f"bucket_of({b},{i}) = {bucket_of(b, i)}, expected {want}")

    # 2. Clock arithmetic invariants over many controls and lengths.
    for base, inc, _ in params.tc_mix:
        for n_plies in (1, 2, 7, 40, 100):
            rng = random.Random(stable_seed("t", base, inc, n_plies))
            res = synthesize_game_clocks(n_plies, base, inc, params, rng)
            check(len(res["clocks"]) == n_plies, f"length mismatch {base}+{inc} n={n_plies}")
            secs = [time_to_seconds(c) for c in res["clocks"]]  # raises if not H:MM:SS ints
            check(all(s >= 0 for s in secs), f"negative clock at {base}+{inc}")
            check(all(s > 0 for s in secs),
                  f"synthesized flag (clock hit 0) at {base}+{inc} n={n_plies}")
            # Per-player countdown: each player's own series must not rise by
            # more than one increment, and must fall when inc == 0.
            for parity in (0, 1):
                own = secs[parity::2]
                for a, b in zip(own, own[1:]):
                    check(b <= a + inc + 1, f"clock rose by more than the increment ({a}->{b}) at {base}+{inc}")
                    if inc == 0:
                        check(b <= a, f"clock rose with zero increment ({a}->{b}) at {base}+{inc}")
            # First reported value can never exceed base + increment.
            check(secs[0] <= base + inc + 1, f"first clock {secs[0]} > base+inc at {base}+{inc}")

    # 3. Determinism and rate-independence of the assignment.
    a1 = game_streams(42, 1500, 7)
    a2 = game_streams(42, 1500, 7)
    tc1 = draw_time_control(a1[0], params.tc_mix)
    tc2 = draw_time_control(a2[0], params.tc_mix)
    check(tc1 == tc2, "time-control draw is not reproducible")
    r1 = synthesize_game_clocks(60, *tc1, params, a1[1])
    r2 = synthesize_game_clocks(60, *tc2, params, a2[1])
    check(r1["clocks"] == r2["clocks"], "clock synthesis is not reproducible")

    # 4. Increment games really can go up (not merely monotone).
    rng = random.Random(1)
    res = synthesize_game_clocks(100, 900, 10, params, rng)
    secs = [time_to_seconds(c) for c in res["clocks"]]
    white = secs[0::2]
    check(any(b > a for a, b in zip(white, white[1:])),
          "no upward move in a 900+10 game; increment arithmetic looks wrong")

    # 5. Label-blindness: the default path must ignore move_is_substituted.
    subs = [i % 3 == 0 for i in range(80)]
    rng_a = random.Random(5)
    rng_b = random.Random(5)
    plain = synthesize_game_clocks(80, 300, 0, params, rng_a)
    with_labels_but_blind = synthesize_game_clocks(80, 300, 0, params, rng_b, substituted=None)
    check(plain["clocks"] == with_labels_but_blind["clocks"], "default path is not label-blind")
    rng_c = random.Random(5)
    labelled = synthesize_game_clocks(80, 300, 0, params, rng_c, substituted=subs)
    check(labelled["clocks"] != plain["clocks"], "--cheater-timing had no effect")

    # 6. Constant method produces a flat clock (the ablation arm).
    flat = synthesize_game_clocks(50, 300, 0, ClockParams(method=METHOD_CONSTANT), random.Random(0))
    check(len(set(flat["clocks"])) == 1, "constant method is not constant")
    # Increment controls are half of DEFAULT_TC_MIX by count and 0.3724 by
    # weight, and they are the case the zero-increment check above cannot see:
    # crediting the increment emits a base + k*inc ramp, not a control.
    for tc_base, tc_inc in ((900, 10), (120, 1), (300, 3), (1800, 20)):
        ramp = synthesize_game_clocks(
            50, tc_base, tc_inc, ClockParams(method=METHOD_CONSTANT), random.Random(0))
        check(len(set(ramp["clocks"])) == 1,
              f"constant method is not constant at {tc_base}+{tc_inc}: "
              f"{len(set(ramp['clocks']))} distinct values, "
              f"{ramp['clocks'][0]} -> {ramp['clocks'][-1]}")
        check(ramp["remaining"][0] == float(tc_base),
              f"constant method at {tc_base}+{tc_inc} does not sit at the base value")
        check(max(ramp["think"]) == 0.0,
              f"constant method at {tc_base}+{tc_inc} reports nonzero think time")

    # 7. KS on identical samples is 0 (tie handling), on shifted samples large.
    rr = random.Random(2)
    s2 = [rr.gauss(0, 1) for _ in range(500)]
    s3 = [rr.gauss(6, 1) for _ in range(500)]
    d_same, p_same = ks_two_sample(s2, s2)
    d_diff, p_diff = ks_two_sample(s2, s3)
    check(d_same == 0.0, f"KS of a sample against itself = {d_same}")
    check(p_same > 0.99, f"KS p of a sample against itself = {p_same}")
    check(d_diff > 0.9, f"KS of a 6-sigma shift = {d_diff}, expected > 0.9")
    check(p_diff < 1e-6, f"KS p of a 6-sigma shift = {p_diff}, expected ~0")
    # Ties must not inflate D: a heavily-rounded pair of identical samples.
    t1 = [round(rr.gauss(0, 1)) for _ in range(400)]
    check(ks_two_sample(t1, t1)[0] == 0.0, "tie handling inflates D")

    # 8. End-to-end retrofit on a constructed fixture corpus (no torch needed:
    #    fixture Positions are plain lists, not tensors).
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        corpus = root / "corpus"
        for case, n_games, n_plies in (("1500_stockfish16_r00", 4, 60), ("1500_stockfish16_r060", 4, 44)):
            d = corpus / case
            d.mkdir(parents=True)
            for gi in range(n_games):
                game = {
                    "Positions": [[0] * 4 for _ in range(n_plies)],
                    "Moves": ["e2e4"] * n_plies,
                    "Clocks": ["0:00:00"] * n_plies,
                    "Result": "1/2-1/2",
                    "WhiteElo": 1500, "BlackElo": 1500,
                    "move_is_substituted": [False] * n_plies,
                }
                with open(d / f"game_{gi:05d}.pkl", "wb") as f:
                    pickle.dump(game, f)
        out = root / "sidecar"
        retrofit_corpus(corpus, out, params, 42, "sidecar", None, False)
        j0 = json.loads((out / "1500_stockfish16_r00" / "clocks.json").read_text())
        j1 = json.loads((out / "1500_stockfish16_r060" / "clocks.json").read_text())
        for gi in range(4):
            name = f"game_{gi:05d}.pkl"
            check(j0["games"][name]["Time"] == j1["games"][name]["Time"],
                  f"time control differs across substitution rates for {name}")
            short = j1["games"][name]["Clocks"]
            long = j0["games"][name]["Clocks"]
            check(len(short) == 44 and len(long) == 60, "sidecar clock length does not match ply count")
            check(short == long[:44], "shared RNG stream did not produce a shared prefix")
        # rewrite mode preserves the original engine think-times and adds Time
        out2 = root / "rewritten"
        retrofit_corpus(corpus, out2, params, 42, "rewrite", None, False)
        with open(out2 / "1500_stockfish16_r00" / "game_00000.pkl", "rb") as f:
            g = pickle.load(f)
        check(g["Clocks_engine_seconds"] == ["0:00:00"] * 60, "original Clocks not preserved")
        check(g["Clocks"] != g["Clocks_engine_seconds"], "Clocks not replaced")
        check(g["Time"] == j0["games"]["game_00000.pkl"]["Time"], "Time key missing or inconsistent")
        check(len(g["Clocks"]) == len(g["Positions"]) == len(g["Moves"]),
              "clock/position length guard would reject this game (format_data.py:78-80)")

        # 9. The fitter round-trips: fit on synthesized 'pretend-real' games and
        #    check the recovered share curve is in the right ballpark.
        realish = root / "realish"
        realish.mkdir()
        for gi in range(120):
            rng = random.Random(stable_seed("realish", gi))
            base, inc = draw_time_control(random.Random(stable_seed("realish-tc", gi)), params.tc_mix)
            res = synthesize_game_clocks(100, base, inc, params, rng)
            with open(realish / f"game_{gi:05d}.pkl", "wb") as f:
                pickle.dump({"Clocks": res["clocks"], "Time": res["time"],
                             "Positions": [], "Moves": []}, f)
        fitted = fit_params_from_real(realish, None, params)
        check(fitted.method == METHOD_EMPIRICAL, "fit did not set the empirical method")
        check("blitz" in fitted.empirical, "fit produced no blitz bucket")
        med = fitted.empirical["blitz"]["share_median"]
        check(all(0.0 < m < 0.5 for m in med), f"implausible fitted share medians: {med}")
        check(med[0] < med[-1] or med[3] > med[0],
              "fitted share curve lost the fast-opening shape entirely")
        # And the fitted params must actually drive synthesis.
        e = synthesize_game_clocks(60, 300, 0, fitted, random.Random(3))
        check(all(time_to_seconds(c) > 0 for c in e["clocks"]), "empirical method produced a flag")

    # 10. First-order pacing calibration (deliverable 1, review section 1.7).
    #     Sigman et al. (2010) report that at White's 40th move in 3-minute
    #     no-increment games, real players have consumed 74.7% (low-rated) to
    #     77.9% (high-rated) of their clock budget. Synthesize >= 2000 games at
    #     180+0 to 80 plies and measure the fraction of the 180 s budget
    #     consumed after ply index 78 (White's 40th move). The mid band must
    #     land inside [0.74, 0.79]. This fails loudly if DEFAULT_PHASE_KNOTS or
    #     sigma_log is reverted to the pre-recalibration values (which gave
    #     ~0.883).
    cal = ClockParams()
    cal_fracs = []
    for gi in range(2500):
        rng = random.Random(stable_seed("cal40", gi))
        res = synthesize_game_clocks(80, 180, 0, cal, rng, band=1500)
        cal_fracs.append((180.0 - res["remaining"][78]) / 180.0)
    cal_mean = statistics.fmean(cal_fracs)
    check(0.74 <= cal_mean <= 0.79,
          f"40th-move budget fraction at 180+0, mid band = {cal_mean:.4f}, "
          f"expected in [0.74, 0.79] (Sigman et al. 2010: 0.747 to 0.779)")

    # 11. Rating-band dependence (deliverable 2, review section 1.6). The band
    #     must reach the model through a real parameter and change the emitted
    #     distribution materially, in Sigman's reported direction: high-rated
    #     players consume more of their budget by move 40 and have larger RT
    #     dispersion. The mid band must still pass deliverable 1's [0.74, 0.79].
    def _band_probe(band: int) -> tuple[float, float]:
        fr: list[float] = []
        think: list[float] = []
        for gj in range(2000):
            rng = random.Random(stable_seed("bandprobe", band, gj))
            res = synthesize_game_clocks(80, 180, 0, ClockParams(), rng, band=band)
            fr.append((180.0 - res["remaining"][78]) / 180.0)
            think.extend(res["think"])
        tmean = statistics.fmean(think)
        tcv = statistics.pstdev(think) / tmean if tmean else 0.0
        return statistics.fmean(fr), tcv

    f_lo, cv_lo = _band_probe(1100)
    f_mid, cv_mid = _band_probe(1500)
    f_hi, cv_hi = _band_probe(1900)
    check(f_hi > f_lo + 0.01,
          f"40th-move budget fraction not materially higher for band 1900 than "
          f"band 1100 (1100={f_lo:.4f}, 1900={f_hi:.4f}); Sigman reports "
          f"high-rated higher (74.7% vs 77.9%)")
    check(f_lo < f_mid < f_hi,
          f"40th-move budget fraction not monotone in band "
          f"(1100={f_lo:.4f}, 1500={f_mid:.4f}, 1900={f_hi:.4f})")
    check(0.74 <= f_mid <= 0.79,
          f"mid band 40th-move fraction {f_mid:.4f} left [0.74, 0.79] under the "
          f"band model (deliverable 1 must still pass at the mix / mid band)")
    check(cv_hi > cv_lo + 0.05,
          f"per-move think-time dispersion not materially larger for band 1900 "
          f"(CV: 1100={cv_lo:.3f}, 1900={cv_hi:.3f}); Sigman SD slopes are "
          f"0.91 (low) vs 1.36 (high)")
    check(cv_lo < cv_mid < cv_hi,
          f"think-time dispersion not monotone in band "
          f"(CV: 1100={cv_lo:.3f}, 1500={cv_mid:.3f}, 1900={cv_hi:.3f})")
    # Band must enter through a parameter, not merely the RNG seed: same seed,
    # different band, different clocks.
    rb_lo = synthesize_game_clocks(60, 180, 0, ClockParams(), random.Random(99), band=1100)
    rb_hi = synthesize_game_clocks(60, 180, 0, ClockParams(), random.Random(99), band=1900)
    check(rb_lo["clocks"] != rb_hi["clocks"],
          "band did not change the clocks at a fixed RNG seed (still seed-only)")

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  - " + f)
        return 1
    print("SELF-TEST PASSED (11 groups)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, help="Root of the anomaly corpus (dir of {band}_{engine}_r{rate}/)")
    ap.add_argument("--out", type=Path, help="Output root for sidecar/rewrite modes")
    ap.add_argument("--mode", choices=["sidecar", "rewrite", "inplace"], default="sidecar",
                    help="sidecar (default, ~72 MB) writes clocks.json per case and leaves the "
                         "23 GiB of pickles untouched; rewrite copies the pickles with the new "
                         "field; inplace edits them (destructive to the file, though the original "
                         "think-times are kept under Clocks_engine_seconds).")
    ap.add_argument("--params", type=Path, help="JSON parameter file (from --fit-out) to plug in")
    ap.add_argument("--method", choices=[METHOD_PARAMETRIC, METHOD_EMPIRICAL, METHOD_CONSTANT],
                    help="Override the method. 'constant' is the clock-ablation control arm.")
    ap.add_argument("--fixed-tc", help="Force one time control, e.g. 300+0 (single-control robustness arm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="Only process the first N games per case")
    ap.add_argument("--cheater-timing", action="store_true",
                    help="NON-DEFAULT: make substituted plies systematically faster. This plants a "
                         "timing signal correlated with the ground-truth label, so it is only valid "
                         "as a separately-reported sensitivity arm, never as the primary corpus.")
    ap.add_argument("--fit-from", type=Path, help="Directory of real preprocessed .pkl games to fit from")
    ap.add_argument("--fit-out", type=Path, help="Where to write the fitted parameter JSON")
    ap.add_argument("--compare-real", type=Path, help="Directory of real .pkl games to KS-compare against")
    ap.add_argument("--simulate", type=int, help="Simulate N games and print a distribution report")
    ap.add_argument("--report-json", type=Path, help="Write the report to this path as JSON")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.params:
        if not args.params.exists():
            print(f"parameter file not found: {args.params}", file=sys.stderr)
            return 2
        params = ClockParams.from_dict(json.loads(args.params.read_text()))
    else:
        params = ClockParams()
    if args.method:
        params.method = args.method
    if args.fixed_tc:
        base, inc = (int(x) for x in args.fixed_tc.split("+"))
        params.tc_mix = normalized_mix([(base, inc, 1.0)])

    report: dict = {}

    if args.fit_from:
        if not args.fit_from.exists():
            print(f"fit source not found: {args.fit_from} (nothing to fit; this is the "
                  f"plug-in step that needs the real corpus)", file=sys.stderr)
            return 3
        params = fit_params_from_real(args.fit_from, args.limit, params)
        if args.fit_out:
            args.fit_out.write_text(json.dumps(params.to_dict(), indent=2))
            print(f"wrote {args.fit_out}")

    if args.compare_real:
        if not args.compare_real.exists():
            print(f"comparison corpus not found: {args.compare_real}. The real-vs-synthetic "
                  f"KS check cannot run until real preprocessed games are available.",
                  file=sys.stderr)
            return 3
        report["comparison"] = compare_with_real(args.compare_real, params, args.seed, args.limit)

    if args.simulate:
        sim = simulate(args.simulate, params, args.seed)
        sim.pop("_samples", None)
        report["simulation"] = sim

    if args.corpus:
        if not args.corpus.exists():
            print(f"corpus not found: {args.corpus}. This clone does not contain the 90,000-game "
                  f"corpus (it lives on the HPC at ~/Bacsain/thesis2/data/anomaly_corpus). "
                  f"Run --self-test or --simulate here instead.", file=sys.stderr)
            return 3
        if args.mode != "inplace" and not args.out:
            print("--out is required unless --mode inplace", file=sys.stderr)
            return 2
        stats = retrofit_corpus(args.corpus, args.out, params, args.seed, args.mode,
                                args.limit, args.cheater_timing)
        sample = stats.pop("remaining_sample")
        stats["remaining_seconds"] = describe(sample)
        stats["normalized_feature"] = describe([(x - CLOCKS_MEAN) / CLOCKS_STD for x in sample])
        report["retrofit"] = stats

    if not report:
        report["simulation"] = {k: v for k, v in simulate(2000, params, args.seed).items()
                                if k != "_samples"}

    text = json.dumps(report, indent=2)
    print(text)
    if args.report_json:
        args.report_json.write_text(text)
        print(f"wrote {args.report_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

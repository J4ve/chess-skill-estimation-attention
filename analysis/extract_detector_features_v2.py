"""Feature extraction for the supervised anomaly (cheat) detector - anomaly corpus v2.

v2 variant of ``extract_detector_features.py``.  The v2 synthetic corpus
(``data/anomaly_corpus_v2/``, 134 cells, 268k games) changed the pickle schema
and the cell grid, so this script adapts the two extraction stages accordingly
while keeping the ``.npz`` layout compatible with ``train_anomaly_detector.py``.

What changed from v1 (see analysis/anomaly-v2-feature-extraction.md):

  * cell grid: ``{band}_{engine}_r{rate}`` with rate in {00,02,05,10,20,40,60}
    (v1 had 00/005/015/030/060), plus 8 ``{band}_hardneg`` strength-mismatch
    clean cells.  Cells are discovered by globbing ``--corpus_dir`` so the order
    is identical between the two stages (needed for the index join in training).
  * clocks: v2 bakes a real remaining-time countdown into ``Clocks`` (list of
    "H:MM:SS" strings, one per ply).  No sidecar - read ``raw["Clocks"]``.
  * R_base: v1 estimated it as the empirical mean final predicted rating over
    clean games.  v2 uses ``R_base = maia_band`` directly (per task spec), so the
    first clean-only GPU pass is gone.
  * new per-ply labels: ``engine_differs_from_maia`` (True / False / None) and
    ``maia_argmax_move`` (UCI or None).
  * new per-game fields: ``opponent_band``, ``suspect_band``, ``engine_setting``
    ({"axis": "depth"|"nodes", "value": int} or None), ``opening_line`` length,
    ``hard_negative``, and the generator's own ``nominal_vs_effective_rate``.

Two independently runnable stages, both write into ``--out_dir``:

  --stage rating_cheap
      Frozen tuned-attention rating model forward (return_attention=True) per
      game -> per-ply R_hat (both colours), attention alpha_t, deviation
      d_t = |R_hat_t - maia_band|, running mean/std, first/second differences;
      cheap board features (python-chess replay): is_capture, is_check,
      material_balance; clock features from the real countdown.  Writes
      ``v2_features/rating_cheap.npz`` (+ ``rating_cheap_meta.json``).
      GPU 0 or 2 only (GPU 1 is the parallel baseline-lr3e4 training arm).

  --stage engine
      Stockfish 16 at a fixed depth (default 20, design range 18-20), multipv 3,
      over the substituted plies AND a matched sample of clean suspect-side plies
      per game (not every ply - the corpus is 268k games).  Per analysed ply:
      centipawn loss, top-1 / top-3 engine-move match, |eval|, eval swing.
      Per-ply arrays are written full length (n_plies per game) with NaN at plies
      that were not analysed, so the training script's index join still works.
      Parallel across ``--sf_workers`` processes.  Writes ``v2_features/engine.npz``.

Both stages are resumable: results are written as one ``.npz`` shard per cell
under ``<out_dir>/<stage>_shards/``; with ``--resume`` a cell whose shard already
exists is skipped.  The final merged ``.npz`` is rebuilt from all shards present
at the end of every run (a partial merge is written even if some cells are still
missing, with a warning).

Ragged per-ply arrays are stored flat with a per-game offset index
(``game_start``, length n_games+1).  game_id = "{cell}/{basename}".
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# On the HPC this script is dropped into ~/Bacsain/thesis2/analysis/scripts/;
# the rating-net module lives in ~/Bacsain/thesis2/src/.
for cand in (HERE.parent.parent / "src",
             HERE.parent.parent / "prototype" / "src",
             Path(os.path.expanduser("~/Bacsain/thesis2/src"))):
    if cand.is_dir():
        sys.path.insert(0, str(cand))

MAX_MOVES = 100
CLOCKS_MEAN, CLOCKS_STD = 273.0, 380.0       # matches ChessGamesDataset defaults
RATINGS_MEAN, RATINGS_STD = 1514.0, 366.0    # matches ChessGamesDataset defaults

PIECE_VAL = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}  # chess.PAWN..KING

# Stockfish score handling.  mate_score keeps forced mates distinct from a huge
# material edge; _clamp_cp then bounds every recorded eval so a mate line does
# not swamp the mean/percentile features.  cp_loss is additionally capped - past
# ~2000 cp the exact value is noise ("catastrophic blunder").
MATE_SCORE = 100000
MATE_CP_CLAMP = 10000.0
CP_LOSS_CLAMP = 2000.0


def _clamp_cp(x):
    return max(-MATE_CP_CLAMP, min(MATE_CP_CLAMP, float(x)))


# ----------------------------------------------------------------------------
# cell discovery / parsing
# ----------------------------------------------------------------------------
def discover_cells(corpus_dir, bands=None, engines=None, rates=None):
    """Return a sorted list of (cell_name, band, engine, rate, is_hardneg).

    engine is one of {"stockfish16", "lc0", "none"}; rate is an int
    (-1 for hard-negative cells).  Optional filters keep only matching cells.
    """
    out = []
    for d in sorted(glob.glob(os.path.join(corpus_dir, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        parts = name.split("_")
        try:
            band = int(parts[0])
        except (ValueError, IndexError):
            continue
        if name.endswith("_hardneg"):
            eng, rate, hn = "none", -1, True
        elif len(parts) == 3 and parts[2].startswith("r"):
            eng, hn = parts[1], False
            try:
                rate = int(parts[2][1:])
            except ValueError:
                continue
        else:
            continue
        if bands and band not in bands:
            continue
        if engines and eng not in engines and not hn:
            continue
        if rates is not None and not hn and rate not in rates:
            continue
        out.append((name, band, eng, rate, hn))
    return out


# ----------------------------------------------------------------------------
# rating model
# ----------------------------------------------------------------------------
def _load_model(checkpoint_path, device):
    import torch
    from chess_rating_net import ChessEloPredictor

    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # The frozen architecture is the tuned attention arm: 4-layer CNN +
    # Bahdanau attention (attention_dim 64).  deeper_cnn stays False.
    model = ChessEloPredictor(use_attention=True, attention_type="bahdanau", attention_dim=64)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys ignored (e.g. {list(unexpected)[:3]})")
    if missing:
        print(f"  [load] {len(missing)} missing keys (e.g. {list(missing)[:3]})")
    model.to(device)
    model.eval()
    return model


def _clocks_z_from_strings(clock_strings, n_real):
    import torch
    from chess_rating_net import time_to_seconds

    secs = [time_to_seconds(c) for c in clock_strings]
    z = [(c - CLOCKS_MEAN) / CLOCKS_STD for c in secs]
    t = torch.tensor(z, dtype=torch.float)[:MAX_MOVES]
    if t.shape[0] < n_real:
        pad = torch.full((n_real - t.shape[0],), (0.0 - CLOCKS_MEAN) / CLOCKS_STD)
        t = torch.cat([t, pad])
    return t[:n_real], np.asarray(secs[:n_real], dtype=np.float64)


def _run_game(model, device, positions, clocks_z):
    import torch

    positions = positions.unsqueeze(0).to(device)
    clocks_z = clocks_z.unsqueeze(0).to(device)
    lengths = torch.tensor([positions.shape[1]])
    with torch.no_grad():
        out = model(positions, clocks_z, lengths, return_attention=True)
    preds = out["per_move_preds"].squeeze(0).cpu().numpy() * RATINGS_STD + RATINGS_MEAN  # (seq,2)
    attn = out.get("attention_weights")
    attn = attn.squeeze(0).cpu().numpy() if attn is not None else None  # (seq,)
    return preds, attn


# ----------------------------------------------------------------------------
# cheap board features (python-chess replay)
# ----------------------------------------------------------------------------
def _cheap_features(moves, suspect_idx):
    """Per-ply arrays: is_capture, is_check, material_balance (suspect POV,
    pre-move), plus the list of FENs (position BEFORE each move).
    suspect_idx 0=white, 1=black."""
    import chess

    board = chess.Board()
    n = len(moves)
    is_capture = np.zeros(n, dtype=np.float32)
    is_check = np.zeros(n, dtype=np.float32)
    material = np.zeros(n, dtype=np.float32)
    fens = []
    ok = True
    for t, uci in enumerate(moves):
        fens.append(board.fen())
        bal = 0.0
        for pt, val in PIECE_VAL.items():
            bal += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        material[t] = bal if suspect_idx == 0 else -bal
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            mv = None
        if mv is None or mv not in board.legal_moves:
            ok = False
            break
        is_capture[t] = 1.0 if board.is_capture(mv) else 0.0
        is_check[t] = 1.0 if board.gives_check(mv) else 0.0
        board.push(mv)
    while len(fens) < n:
        fens.append(board.fen())
    return is_capture, is_check, material, fens, ok


def _traj_features(series):
    """series: (n,) suspect-side predicted rating per ply.
    Returns run_mean, run_std, first_diff, second_diff (all (n,))."""
    n = len(series)
    csum = np.cumsum(series)
    idx = np.arange(1, n + 1, dtype=np.float64)
    run_mean = csum / idx
    csum2 = np.cumsum(series ** 2)
    run_var = np.maximum(csum2 / idx - run_mean ** 2, 0.0)
    run_std = np.sqrt(run_var)
    first_diff = np.zeros(n, dtype=np.float64)
    first_diff[1:] = np.diff(series)
    second_diff = np.zeros(n, dtype=np.float64)
    second_diff[1:] = np.diff(first_diff)
    return run_mean, run_std, first_diff, second_diff


# ----------------------------------------------------------------------------
# per-ply label helpers
# ----------------------------------------------------------------------------
def _as_bool_arr(seq, n_real):
    a = np.zeros(n_real, dtype=bool)
    for t in range(min(n_real, len(seq or []))):
        a[t] = bool(seq[t])
    return a


def _engine_differs_arr(seq, n_real):
    """Map the tri-state engine_differs_from_maia list to float:
    True -> 1.0, False -> 0.0, None -> NaN.  Also return the eligibility mask
    (entry is not None)."""
    v = np.full(n_real, np.nan, dtype=np.float64)
    elig = np.zeros(n_real, dtype=bool)
    for t in range(min(n_real, len(seq or []))):
        x = seq[t]
        if x is None:
            continue
        elig[t] = True
        v[t] = 1.0 if x is True else 0.0
    return v, elig


# ----------------------------------------------------------------------------
# stage: rating_cheap
# ----------------------------------------------------------------------------
def _shard_path(out_dir, stage, cell):
    return os.path.join(out_dir, f"{stage}_shards", f"{cell}.npz")


def stage_rating_cheap(args):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device)
    shard_dir = os.path.join(args.out_dir, "rating_cheap_shards")
    os.makedirs(shard_dir, exist_ok=True)
    print(f"device={device}  games_per_cell={args.games_per_cell or 'all'}  "
          f"checkpoint={args.checkpoint}", flush=True)

    bands = _int_set(args.bands)
    engines = _str_set(args.engines)
    cells = discover_cells(args.corpus_dir, bands, engines)
    print(f"{len(cells)} cells", flush=True)

    rbase_meta = {}
    t0 = time.time()
    n_games_total = 0
    for ci, (cell, band, eng, rate, hn) in enumerate(cells):
        sp = _shard_path(args.out_dir, "rating_cheap", cell)
        if args.resume and os.path.exists(sp):
            print(f"[{ci+1}/{len(cells)}] {cell}: shard exists, skip", flush=True)
            continue
        files = sorted(glob.glob(os.path.join(args.corpus_dir, cell, "game_*.pkl")))
        if args.games_per_cell:
            files = files[: args.games_per_cell]
        if not files:
            print(f"[{ci+1}/{len(cells)}] {cell}: no games", flush=True)
            continue

        acc = _RatingAcc()
        finals = []
        for fp in files:
            raw = pickle.load(open(fp, "rb"))
            pos = torch.stack(raw["Positions"])[:MAX_MOVES]
            n_real = pos.shape[0]
            moves = list(raw["Moves"])[:n_real]
            sidx = 0 if raw.get("suspect_color") == "white" else 1
            r_base = float(raw.get("maia_band", band))

            cs = raw.get("Clocks", []) or []
            clk_z, clk_sec = _clocks_z_from_strings(cs, n_real)
            preds, attn = _run_game(model, device, pos, clk_z)
            if attn is None:
                attn = np.full(n_real, 1.0 / max(n_real, 1))
            attn = np.asarray(attn[:n_real], dtype=np.float64)

            rhat_s = preds[:n_real, sidx].astype(np.float64)
            rhat_o = preds[:n_real, 1 - sidx].astype(np.float64)
            d_t = np.abs(rhat_s - r_base)
            rmean, rstd, d1, d2 = _traj_features(rhat_s)
            cap, chk, mat, _fens, ok = _cheap_features(moves, sidx)

            sub = _as_bool_arr(raw.get("move_is_substituted", []), n_real)
            edm, elig = _engine_differs_arr(raw.get("engine_differs_from_maia", []), n_real)
            argmax_present = np.zeros(n_real, dtype=np.float32)
            am = raw.get("maia_argmax_move", []) or []
            for t in range(min(n_real, len(am))):
                argmax_present[t] = 1.0 if am[t] is not None else 0.0
            is_sus = np.array([(t % 2) == sidx for t in range(n_real)], dtype=np.float32)

            # clock remaining + per-side think time (same-side delta, 2 plies back)
            clkrem = np.zeros(n_real, dtype=np.float64)
            m = min(n_real, clk_sec.shape[0])
            clkrem[:m] = clk_sec[:m]
            clkdelta = np.full(n_real, np.nan, dtype=np.float64)
            for t in range(2, n_real):
                if clkrem[t - 2] > 0 and clkrem[t] > 0:
                    clkdelta[t] = clkrem[t - 2] - clkrem[t]

            rate_info = raw.get("nominal_vs_effective_rate", {}) or {}
            # denominator = suspect plies eligible for substitution (ply >= 17);
            # the generator records these in n_eligible_plies, and every such ply
            # also carries a maia_argmax_move, so argmax_present.sum() cross-checks.
            n_elig_suspect = int(rate_info.get("n_eligible_plies", int(argmax_present.sum())))
            n_dif = int(rate_info.get("n_differs", int(np.nansum(edm))))
            n_substd = int(rate_info.get("n_substituted", int(sub.sum())))
            eff_rate = float(rate_info.get("effective", 0.0))
            eff_rate_check = float(n_dif / max(int(argmax_present.sum()), 1))
            es = raw.get("engine_setting") or {}
            gid = f"{cell}/{os.path.basename(fp)}"

            acc.add_game(
                game_id=gid, band=band, engine=eng, rate=rate, is_hardneg=int(hn),
                suspect_idx=sidx, n_plies=n_real,
                white_elo=int(raw.get("WhiteElo", 0)), black_elo=int(raw.get("BlackElo", 0)),
                maia_band=int(raw.get("maia_band", band)),
                suspect_band=int(raw.get("suspect_band", raw.get("maia_band", band))),
                opponent_band=int(raw.get("opponent_band", raw.get("maia_band", band))),
                r_base=r_base,
                game_label=int(sub.any()),
                game_label_differs=int(n_dif > 0),
                n_sub=n_substd, n_differs=n_dif, n_eligible=n_elig_suspect,
                n_sub_attempts=int(elig.sum()),
                n_suspect=int(is_sus.sum()),
                nominal_rate=float(raw.get("substitution_rate", 0.0)),
                eff_rate=eff_rate,
                eff_rate_check=eff_rate_check,
                nominal_rate_realized=float(rate_info.get("nominal", 0.0)),
                opening_len=int(raw.get("n_book_plies", len(raw.get("opening_line", []) or []))),
                hard_negative=int(bool(raw.get("hard_negative", hn))),
                engine_setting_axis={"depth": 1, "nodes": 2}.get(es.get("axis"), 0),
                engine_setting_value=int(es.get("value", 0)),
                replay_ok=int(ok),
                # per ply
                ply_idx=np.arange(n_real, dtype=np.int32),
                is_suspect_move=is_sus,
                move_is_substituted=sub.astype(np.float32),
                engine_differs=edm.astype(np.float32),
                is_eligible=elig.astype(np.float32),
                maia_argmax_present=argmax_present,
                r_hat_suspect=rhat_s, r_hat_other=rhat_o,
                alpha=attn, d_t=d_t,
                run_mean=rmean, run_std=rstd, first_diff=d1, second_diff=d2,
                is_capture=cap, is_check=chk, material_balance=mat,
                clock_remaining=clkrem, clock_delta=clkdelta,
            )
            finals.append(float(rhat_s[n_real - 1]) if n_real else float(band))
            n_games_total += 1
            if n_games_total % 2000 == 0:
                dt = time.time() - t0
                print(f"  ...{n_games_total} games  {dt:.0f}s  ({n_games_total/dt:.1f} g/s)", flush=True)

        acc.save(sp)
        rbase_meta[cell] = {
            "r_base": float(band), "nominal": float(band), "n_games": len(files),
            "empirical_final_mean": float(np.mean(finals)) if finals else None,
        }
        print(f"[{ci+1}/{len(cells)}] {cell}: wrote {len(files)} games -> {sp}", flush=True)

    _merge_shards(args.out_dir, "rating_cheap", cells,
                  os.path.join(args.out_dir, "rating_cheap.npz"))
    with open(os.path.join(args.out_dir, "rating_cheap_meta.json"), "w") as f:
        json.dump({"r_base": rbase_meta, "checkpoint": args.checkpoint,
                   "games_per_cell": args.games_per_cell,
                   "r_base_definition": "maia_band (per task spec)",
                   "corpus_dir": args.corpus_dir}, f, indent=2)
    print(f"\ndone rating_cheap  ({n_games_total} games this run)", flush=True)


class _RatingAcc:
    """Accumulates per-game scalars and per-ply flat arrays for one cell,
    then writes a single compressed .npz shard."""

    _SCALARS = ("game_id", "band", "engine", "rate", "is_hardneg", "suspect_idx",
                "n_plies", "white_elo", "black_elo", "maia_band", "suspect_band",
                "opponent_band", "r_base", "game_label", "game_label_differs",
                "n_sub", "n_differs", "n_eligible", "n_sub_attempts", "n_suspect",
                "nominal_rate", "eff_rate", "eff_rate_check", "nominal_rate_realized",
                "opening_len", "hard_negative", "engine_setting_axis",
                "engine_setting_value", "replay_ok")
    _PLY = ("ply_idx", "is_suspect_move", "move_is_substituted", "engine_differs",
            "is_eligible", "maia_argmax_present", "r_hat_suspect", "r_hat_other",
            "alpha", "d_t", "run_mean", "run_std", "first_diff", "second_diff",
            "is_capture", "is_check", "material_balance", "clock_remaining",
            "clock_delta")

    def __init__(self):
        self.s = {k: [] for k in self._SCALARS}
        self.p = {k: [] for k in self._PLY}
        self.game_start = [0]

    def add_game(self, **kw):
        for k in self._SCALARS:
            self.s[k].append(kw[k])
        for k in self._PLY:
            self.p[k].append(np.asarray(kw[k]))
        self.game_start.append(self.game_start[-1] + int(kw["n_plies"]))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = {"game_start": np.asarray(self.game_start, dtype=np.int64)}
        for k in self._SCALARS:
            if k in ("game_id", "engine"):
                out[k] = np.asarray(self.s[k], dtype=object if k == "game_id" else "<U16")
            elif k in ("r_base", "nominal_rate", "eff_rate", "eff_rate_check",
                       "nominal_rate_realized"):
                out[k] = np.asarray(self.s[k], dtype=np.float32)
            else:
                out[k] = np.asarray(self.s[k], dtype=np.int32)
        for k in self._PLY:
            dt = np.int32 if k == "ply_idx" else np.float32
            out[k] = (np.concatenate(self.p[k]).astype(dt) if self.p[k]
                      else np.zeros(0, dtype=dt))
        _atomic_savez(path, out)


# ----------------------------------------------------------------------------
# stage: engine (Stockfish, parallel, subset of plies)
# ----------------------------------------------------------------------------
def _select_plies(raw, n_real, sidx, n_book, clean_per_game, seed):
    """Return a sorted list of ply indices to analyse: every substituted ply
    plus a matched sample of clean suspect-side post-book plies from the same
    game.  For games with no substitution, sample ``clean_per_game`` clean
    suspect-side post-book plies."""
    sub = raw.get("move_is_substituted", []) or []
    sub_plies = [t for t in range(n_real) if t < len(sub) and sub[t]]
    clean_pool = [t for t in range(max(n_book, 0), n_real)
                  if (t % 2) == sidx and not (t < len(sub) and sub[t])]
    rng = random.Random(seed)
    k = max(len(sub_plies), clean_per_game)
    if k >= len(clean_pool):
        clean_sel = clean_pool
    else:
        clean_sel = rng.sample(clean_pool, k)
    return sorted(set(sub_plies) | set(clean_sel))


def _sf_one_game(a):
    """Worker: analyse a subset of one game's plies with Stockfish.  Never
    raises - on any failure the game's per-ply arrays come back all-NaN with a
    zero ``analyzed`` mask, so one bad game cannot abort a whole cell (and, since
    resume is cell-granular, cannot wedge a resumed run in a crash loop)."""
    (gid, fp, sf_bin, depth, time_cap_s, clean_per_game, seed) = a
    try:
        raw = pickle.load(open(fp, "rb"))
        n_real = min(len(raw["Moves"]), MAX_MOVES)
    except Exception as e:
        print(f"  [engine] {gid}: unreadable ({e!r})", flush=True)
        z = np.zeros(0, dtype=np.float32)
        return gid, 0, z, z, z, z, z, z
    try:
        return _sf_one_game_inner(gid, fp, raw, n_real, sf_bin, depth,
                                  time_cap_s, clean_per_game, seed)
    except Exception as e:
        print(f"  [engine] {gid}: analysis failed ({e!r}) - NaN arrays", flush=True)
        nan = np.full(n_real, np.nan, dtype=np.float32)
        return (gid, n_real, nan.copy(), nan.copy(), nan.copy(), nan.copy(),
                nan.copy(), np.zeros(n_real, dtype=np.float32))


def _sf_one_game_inner(gid, fp, raw, n_real, sf_bin, depth, time_cap_s,
                       clean_per_game, seed):
    import chess
    import chess.engine

    moves = list(raw["Moves"])[:n_real]
    sidx = 0 if raw.get("suspect_color") == "white" else 1
    n_book = int(raw.get("n_book_plies", 0))
    want = set(_select_plies(raw, n_real, sidx, n_book, clean_per_game, seed))

    cp_loss = np.full(n_real, np.nan, dtype=np.float32)
    top1 = np.full(n_real, np.nan, dtype=np.float32)
    top3 = np.full(n_real, np.nan, dtype=np.float32)
    abs_eval = np.full(n_real, np.nan, dtype=np.float32)
    eval_swing = np.full(n_real, np.nan, dtype=np.float32)
    analyzed = np.zeros(n_real, dtype=np.float32)

    if not want:
        return gid, n_real, cp_loss, top1, top3, abs_eval, eval_swing, analyzed

    lim = chess.engine.Limit(depth=depth)
    if time_cap_s and time_cap_s > 0:
        lim = chess.engine.Limit(depth=depth, time=time_cap_s)

    eng = chess.engine.SimpleEngine.popen_uci(sf_bin)
    try:
        eng.configure({"Threads": 1, "Hash": 64})
        board = chess.Board()
        prev_white_cp = None       # eval (White POV, clamped) at the previous analysed ply
        for t, uci in enumerate(moves):
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                break
            if mv not in board.legal_moves:
                break
            if t not in want:
                board.push(mv)
                continue
            try:
                infos = eng.analyse(board, lim, multipv=3)
            except Exception:
                board.push(mv)
                continue
            if isinstance(infos, dict):
                infos = [infos]
            best_moves, best_score_cp = [], None
            for i, inf in enumerate(infos):
                pv = inf.get("pv")
                if pv:
                    best_moves.append(pv[0])
                sc = inf.get("score")
                if i == 0 and sc is not None:
                    best_score_cp = sc.pov(board.turn).score(mate_score=MATE_SCORE)
            top1[t] = 1.0 if (best_moves and mv == best_moves[0]) else 0.0
            top3[t] = 1.0 if mv in best_moves else 0.0
            white_cp = None
            if best_score_cp is not None:
                bc = _clamp_cp(best_score_cp)
                abs_eval[t] = abs(bc)
                white_cp = bc if board.turn == chess.WHITE else -bc
                if prev_white_cp is not None:
                    # position volatility between successive analysed checkpoints
                    eval_swing[t] = white_cp - prev_white_cp

            played_cp = None
            for inf in infos:
                pv = inf.get("pv")
                if pv and pv[0] == mv:
                    sc = inf.get("score")
                    if sc is not None:
                        played_cp = sc.pov(board.turn).score(mate_score=MATE_SCORE)
                    break
            if played_cp is None:
                try:
                    inf2 = eng.analyse(board, lim, root_moves=[mv])
                    if isinstance(inf2, list):
                        inf2 = inf2[0]
                    sc = inf2.get("score")
                    if sc is not None:
                        played_cp = sc.pov(board.turn).score(mate_score=MATE_SCORE)
                except Exception:
                    played_cp = None
            if best_score_cp is not None and played_cp is not None:
                cp_loss[t] = min(CP_LOSS_CLAMP,
                                 max(0.0, _clamp_cp(best_score_cp) - _clamp_cp(played_cp)))
            analyzed[t] = 1.0
            if white_cp is not None:
                prev_white_cp = white_cp
            board.push(mv)
    finally:
        try:
            eng.quit()
        except Exception:
            pass
    return gid, n_real, cp_loss, top1, top3, abs_eval, eval_swing, analyzed


def stage_engine(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    shard_dir = os.path.join(args.out_dir, "engine_shards")
    os.makedirs(shard_dir, exist_ok=True)
    bands = _int_set(args.bands)
    engines = _str_set(args.engines)
    cells = discover_cells(args.corpus_dir, bands, engines)
    print(f"engine stage: {len(cells)} cells  workers={args.sf_workers}  "
          f"depth={args.sf_depth}  time_cap={args.sf_time_cap_s}s  "
          f"clean_plies_per_game={args.clean_plies_per_game}  "
          f"sf_games_per_cell={args.sf_games_per_cell or 'all'}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.sf_workers) as ex:
        for ci, (cell, band, eng, rate, hn) in enumerate(cells):
            sp = _shard_path(args.out_dir, "engine", cell)
            if args.resume and os.path.exists(sp):
                print(f"[{ci+1}/{len(cells)}] {cell}: shard exists, skip", flush=True)
                continue
            files = sorted(glob.glob(os.path.join(args.corpus_dir, cell, "game_*.pkl")))
            if args.sf_games_per_cell:
                files = files[: args.sf_games_per_cell]
            if not files:
                continue
            tasks = []
            for fp in files:
                gid = f"{cell}/{os.path.basename(fp)}"
                seed = zlib.crc32(gid.encode())      # stable across runs
                tasks.append((gid, fp, args.sf_bin, args.sf_depth, args.sf_time_cap_s,
                              args.clean_plies_per_game, seed))
            g_id, g_n = [], []
            p_cp, p_t1, p_t3, p_ae, p_sw, p_an = [], [], [], [], [], []
            game_start = [0]
            futs = {ex.submit(_sf_one_game, tk): tk[0] for tk in tasks}
            res = {}
            for fut in as_completed(futs):
                gid, n_real, cp, t1, t3, ae, sw, an = fut.result()
                res[gid] = (n_real, cp, t1, t3, ae, sw, an)
            for tk in tasks:                       # deterministic shard order
                gid = tk[0]
                n_real, cp, t1, t3, ae, sw, an = res[gid]
                g_id.append(gid); g_n.append(n_real)
                p_cp.append(cp); p_t1.append(t1); p_t3.append(t3)
                p_ae.append(ae); p_sw.append(sw); p_an.append(an)
                game_start.append(game_start[-1] + n_real)
            _atomic_savez(sp, dict(
                game_id=np.asarray(g_id, dtype=object),
                game_start=np.asarray(game_start, dtype=np.int64),
                n_plies=np.asarray(g_n, dtype=np.int32),
                cp_loss=_cc(p_cp), top1_match=_cc(p_t1), top3_match=_cc(p_t3),
                abs_eval=_cc(p_ae), eval_swing=_cc(p_sw), analyzed=_cc(p_an),
            ))
            dt = time.time() - t0
            print(f"[{ci+1}/{len(cells)}] {cell}: {len(tasks)} games -> {sp}  "
                  f"({dt:.0f}s elapsed)", flush=True)

    _merge_shards(args.out_dir, "engine", cells,
                  os.path.join(args.out_dir, "engine.npz"))
    with open(os.path.join(args.out_dir, "engine_meta.json"), "w") as f:
        json.dump({"sf_depth": args.sf_depth, "sf_time_cap_s": args.sf_time_cap_s,
                   "multipv": 3, "sf_bin": args.sf_bin,
                   "clean_plies_per_game": args.clean_plies_per_game,
                   "sf_games_per_cell": args.sf_games_per_cell,
                   "ply_selection": "substituted plies + matched clean suspect-side "
                                    "post-book plies; NaN elsewhere; analyzed mask included",
                   "corpus_dir": args.corpus_dir}, f, indent=2)
    print(f"\ndone engine  {time.time()-t0:.0f}s", flush=True)


def _cc(lst):
    return np.concatenate(lst).astype(np.float32) if lst else np.zeros(0, dtype=np.float32)


def _atomic_savez(path, arrays):
    """np.savez_compressed to a temp file in the same dir, then rename into
    place.  Writes through an explicit file handle so numpy does not append a
    second '.npz' to the temp name."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# shard merge
# ----------------------------------------------------------------------------
def _merge_shards(out_dir, stage, cells, out_path):
    shard_dir = os.path.join(out_dir, f"{stage}_shards")
    present = [(c[0], _shard_path(out_dir, stage, c[0])) for c in cells]
    present = [(name, p) for name, p in present if os.path.exists(p)]
    missing = len(cells) - len(present)
    if not present:
        print(f"[merge] no {stage} shards yet, nothing to merge", flush=True)
        return
    per_game_keys, per_ply_keys = None, None
    merged = {}
    game_start = [0]
    for name, p in present:
        z = np.load(p, allow_pickle=True)
        gs = z["game_start"]
        n = len(z["game_id"])
        if per_game_keys is None:
            per_game_keys, per_ply_keys = [], []
            for k in z.files:
                if k == "game_start":
                    continue
                (per_game_keys if len(z[k]) == n else per_ply_keys).append(k)
            for k in per_game_keys + per_ply_keys:
                merged[k] = []
        for k in per_game_keys:
            merged[k].append(z[k])
        for k in per_ply_keys:
            merged[k].append(z[k])
        for j in range(n):
            game_start.append(game_start[-1] + int(gs[j + 1] - gs[j]))
    final = {"game_start": np.asarray(game_start, dtype=np.int64)}
    for k in per_game_keys + per_ply_keys:
        final[k] = np.concatenate(merged[k])
    _atomic_savez(out_path, final)
    n_games = len(final["game_id"])
    tag = f" ({missing} cells still missing)" if missing else ""
    print(f"[merge] wrote {out_path}  ({n_games} games, {len(present)}/{len(cells)} cells){tag}",
          flush=True)


# ----------------------------------------------------------------------------
def _int_set(s):
    return {int(x) for x in s.split(",")} if s else None


def _str_set(s):
    return {x.strip() for x in s.split(",")} if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["rating_cheap", "engine"])
    ap.add_argument("--corpus_dir", required=True,
                    help="e.g. data/anomaly_corpus_v2")
    ap.add_argument("--checkpoint",
                    default="models/preflight_check_2m/best_model.pth",
                    help="frozen tuned-attention rating checkpoint (best val)")
    ap.add_argument("--out_dir", required=True, help="e.g. analysis/v2_features")
    ap.add_argument("--bands", default=None, help="optional filter, e.g. 1100,1500")
    ap.add_argument("--engines", default=None, help="optional filter, e.g. lc0,stockfish16")
    ap.add_argument("--games_per_cell", type=int, default=0,
                    help="rating_cheap: 0 = all games in the cell")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose shard already exists")
    # engine stage
    ap.add_argument("--sf_games_per_cell", type=int, default=0,
                    help="engine: 0 = all games in the cell")
    ap.add_argument("--sf_bin", default=os.path.expanduser(
        "~/Bacsain/thesis2/engines/stockfish/stockfish-ubuntu-x86-64-avx2"))
    ap.add_argument("--sf_depth", type=int, default=20,
                    help="fixed search depth (design range 18-20)")
    ap.add_argument("--sf_time_cap_s", type=float, default=5.0,
                    help="wall-clock safety cap per position (0 = pure depth)")
    ap.add_argument("--sf_workers", type=int, default=10)
    ap.add_argument("--clean_plies_per_game", type=int, default=10,
                    help="engine: matched clean suspect-side plies sampled per game")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.stage == "rating_cheap":
        stage_rating_cheap(args)
    else:
        stage_engine(args)


if __name__ == "__main__":
    main()

"""Stockfish pass over every ply of a sample of TEST games: one engine.analyse
per position (T+1 positions per T-ply game), not per move. The move's
centipawn loss and the before/after evaluation are then derived from
consecutive positions' evals, so no second ("root_moves") search is needed:

    white_pov(t)   = engine eval of the position before move t, White POV
    white_pov(t+1) = engine eval of the position after move t (== before
                     move t+1), White POV
    mover sign s   = +1 if White played move t, else -1
    cp_loss(t)     = max(0, s * (white_pov(t) - white_pov(t+1)))

This is the standard before/after construction (what Lichess-style blunder
checks compute), just computed from a single sequential pass of searches
instead of a best-move search plus a root_moves-constrained search per ply.

Depth, Threads=1, Hash are fixed for determinism; the Stockfish binary path
and its reported version are recorded in the output for the write-up.

Never raises on a single bad game: that game's record carries an "error"
field instead, so one bad game cannot abort the whole run (mirrors the
"never raises" discipline in analysis/extract_detector_features_v2.py's
_sf_one_game).

Usage:
    python move_explanation_engine.py \
        --store-dir /tmp/ratingnet_store \
        --games-file sample_games.json \
        --sf-bin ~/thesis2/engines/stockfish/stockfish-ubuntu-x86-64-avx2 \
        --sf-depth 17 --sf-time-cap-s 1.0 \
        --workers 10 \
        --max-plies 100 \
        --out-jsonl engine.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import zlib
from pathlib import Path

MATE_SCORE = 100000
MATE_CP_CLAMP = 10000.0
CP_LOSS_CLAMP = 2000.0

# Non-pawn, non-king material (points), summed over both sides, used only to
# bucket game phase. Opening is ply-based (first 10 plies, usually still
# book); past that, middlegame/endgame is this material threshold, not ply,
# so it tracks actual simplification rather than move count.
PIECE_VAL = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}  # chess.PAWN..KING
ENDGAME_MATERIAL_THRESHOLD = 13.0  # roughly a queen+rook or less, each side
OPENING_PLY_CUTOFF = 10


def _clamp_cp(x: float) -> float:
    return max(-MATE_CP_CLAMP, min(MATE_CP_CLAMP, float(x)))


def game_phase(board, ply: int) -> str:
    import chess

    if ply < OPENING_PLY_CUTOFF:
        return "opening"
    material = 0.0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            material += PIECE_VAL[piece_type] * len(board.pieces(piece_type, color))
    return "endgame" if material <= ENDGAME_MATERIAL_THRESHOLD else "middlegame"


def analyse_game(gid, moves_uci, sf_bin, depth, time_cap_s, max_plies):
    """Returns a dict with per-ply arrays, or {"game_id": gid, "error": str}."""
    import chess
    import chess.engine

    n = min(len(moves_uci), max_plies)
    white_pov = [None] * (n + 1)  # white_pov[p] = eval of position after p plies
    depth_reached = [None] * (n + 1)  # the depth/time cap races; record which position hit which
    mover = [None] * n
    phase = [None] * n
    san = [None] * n

    lim = chess.engine.Limit(depth=depth)
    if time_cap_s and time_cap_s > 0:
        lim = chess.engine.Limit(depth=depth, time=time_cap_s)

    try:
        eng = chess.engine.SimpleEngine.popen_uci(sf_bin)
    except Exception as e:
        return {"game_id": gid, "error": f"engine launch failed: {e!r}"}

    try:
        eng.configure({"Threads": 1, "Hash": 64})
        board = chess.Board()

        def eval_white_pov(b):
            try:
                info = eng.analyse(b, lim)
                sc = info.get("score")
                reached = info.get("depth")
                if sc is None:
                    return None, reached
                return _clamp_cp(sc.white().score(mate_score=MATE_SCORE)), reached
            except Exception:
                return None, None

        white_pov[0], depth_reached[0] = eval_white_pov(board)
        for t in range(n):
            try:
                mv = chess.Move.from_uci(moves_uci[t])
            except Exception as e:
                return {"game_id": gid, "error": f"bad uci at ply {t}: {e!r}", "n_plies_done": t}
            if mv not in board.legal_moves:
                return {"game_id": gid, "error": f"illegal move at ply {t}: {moves_uci[t]}", "n_plies_done": t}
            san[t] = board.san(mv)
            mover[t] = "white" if board.turn == chess.WHITE else "black"
            phase[t] = game_phase(board, t)
            board.push(mv)
            white_pov[t + 1], depth_reached[t + 1] = eval_white_pov(board)
    finally:
        try:
            eng.quit()
        except Exception:
            pass

    cp_loss = [None] * n
    eval_before = [None] * n
    eval_after = [None] * n
    depth_before = [None] * n
    depth_after = [None] * n
    for t in range(n):
        eb, ea = white_pov[t], white_pov[t + 1]
        eval_before[t] = eb
        eval_after[t] = ea
        depth_before[t] = depth_reached[t]
        depth_after[t] = depth_reached[t + 1]
        if eb is None or ea is None:
            continue
        s = 1.0 if mover[t] == "white" else -1.0
        cp_loss[t] = min(CP_LOSS_CLAMP, max(0.0, s * (eb - ea)))

    return {
        "game_id": gid,
        "n_plies": n,
        "mover": mover,
        "san": san,
        "phase": phase,
        "eval_before_white_pov": eval_before,
        "eval_after_white_pov": eval_after,
        "depth_before": depth_before,
        "depth_after": depth_after,
        "cp_loss": cp_loss,
    }


def _worker(args_tuple):
    gid, moves_uci, sf_bin, depth, time_cap_s, max_plies = args_tuple
    try:
        return analyse_game(gid, moves_uci, sf_bin, depth, time_cap_s, max_plies)
    except Exception as e:
        return {"game_id": gid, "error": f"unhandled: {e!r}"}


def load_moves_from_store(store_dir: str, gid: str) -> list[str]:
    """Reads Moves (UCI) for one game directly from the GameBlobStore sqlite,
    without importing the prototype's torch-dependent GameBlobStore class, so
    this script has no torch dependency and can run alongside the GPU job."""
    con = sqlite3.connect(f"file:{Path(store_dir) / 'corpus.sqlite'}?mode=ro&immutable=1", uri=True)
    try:
        row = con.execute("SELECT blob FROM games WHERE name = ?", (gid,)).fetchone()
        if row is None:
            raise KeyError(gid)
        data = pickle.loads(zlib.decompress(row[0]))
        return list(data["Moves"])
    finally:
        con.close()


def _probe_stockfish_id(sf_bin: str) -> dict:
    """UCI "id name"/"id author" via a raw pipe, not python-chess (see main())."""
    import subprocess

    proc = subprocess.Popen(
        [sf_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    out = {"name": None, "author": None}
    try:
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("id name "):
                out["name"] = line[len("id name "):]
            elif line.startswith("id author "):
                out["author"] = line[len("id author "):]
            elif line == "uciok":
                break
        proc.stdin.write("quit\n")
        proc.stdin.flush()
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--games-file", required=True)
    ap.add_argument("--sf-bin", required=True)
    ap.add_argument("--sf-depth", type=int, default=17)
    ap.add_argument("--sf-time-cap-s", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=100)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    games_meta = json.loads(Path(args.games_file).read_text())
    game_ids: list[str] = games_meta["game_ids"]
    if args.limit:
        game_ids = game_ids[: args.limit]

    # Get the engine's UCI "id" via a raw subprocess, NOT python-chess's
    # SimpleEngine, in this (soon to be forked) parent process: starting a
    # python-chess engine here sets up asyncio/subprocess-watcher state that
    # is not fork-safe, and every worker below inherits a broken copy of it
    # ("'_Local' object has no attribute 'watcher'"). python-chess engines
    # are only ever started inside a worker process, after the fork.
    sf_id = _probe_stockfish_id(args.sf_bin)
    print(f"Stockfish id: {sf_id}", flush=True)

    tasks = []
    for gid in game_ids:
        moves_uci = load_moves_from_store(args.store_dir, gid)
        tasks.append((gid, moves_uci, args.sf_bin, args.sf_depth, args.sf_time_cap_s, args.max_plies))

    n_done = 0
    n_error = 0
    mp_ctx = multiprocessing.get_context("spawn")
    with open(args.out_jsonl, "w") as out_f, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp_ctx
    ) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            rec = fut.result()
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_done += 1
            if "error" in rec:
                n_error += 1
                print(f"[error] {rec['game_id']}: {rec['error']}", flush=True)
            if n_done % 50 == 0:
                print(f"{n_done}/{len(tasks)} games analysed ({n_error} errors)", flush=True)

    print(f"Done: {n_done} games, {n_error} errors. Stockfish: {sf_id}", flush=True)
    Path(args.out_jsonl + ".meta.json").write_text(
        json.dumps(
            {
                "stockfish_id": sf_id,
                "sf_depth": args.sf_depth,
                "sf_time_cap_s": args.sf_time_cap_s,
                "n_games": n_done,
                "n_errors": n_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

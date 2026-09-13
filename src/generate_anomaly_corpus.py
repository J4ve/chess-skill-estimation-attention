"""
Generate the synthetic bot-vs-bot anomaly-validation corpus, version 2
(Chapter 3's Anomaly Validation Protocol, Part 1: synthetic generation).

Full design and rationale: data/anomaly-corpus-v2/design.md. v2 supersedes the
90k v1 corpus (analysis/anomaly-corpus-generated.md) and closes its five holes:

  1. Real-opening seeding. Every game is seeded with a real opening line drawn
     from the project's own 2.55M-game Lichess corpus (per-band frequency
     tables built by build_opening_tables.py), replayed 6-10 plies, then Maia
     plays on. No external book.
  2. Maia policy sampling. For plies 1..16 (opening + early middlegame) a Maia
     move is sampled from the Maia policy head at temperature --maia-temp
     (default 0.9); from ply 17 the move is Maia's argmax. This spreads games
     within a cell without letting sampling noise distort the clean baseline
     (honest argmax Maia from the middlegame on). Because lc0 v0.31 exposes no
     search-time temperature option, the policy is read directly from
     `VerboseMoveStats` and sampled here (see MaiaPolicyEngine).
  3. Out-of-book substitution only. A move is eligible for engine substitution
     only from ply 17 (0-based index 16), i.e. always past the seeded book.
     Placement within the eligible plies stays uniform-at-random. Optional
     resample-until-differs: if the drawn engine move equals Maia's own argmax
     the substitution is deferred to a later eligible ply (up to a cap), so the
     effective substitution rate tracks the nominal one by construction. Both
     rates are recorded per game.
  4. Per-game engine strength. The substitution engine's strength is sampled
     per game and recorded: Stockfish fixed search depth from {14,16,18,20}
     (with a wall-time safety cap), Lc0 node budget from {200,400,800,1600}.
  5. Simulated remaining-time clocks. A real time control (base+increment) is
     drawn per game from the corpus time-control mix and a Lichess-style
     remaining-time countdown is written into `Clocks`, using the shared,
     label-blind think-time model in
     analysis/scripts/synthesize_anomaly_clocks.py. No sidecar file. A
     constant-clock leak-check variant is available via --clock-mode constant.

Hard-negative strength-mismatch cells (design decision 6): with --hard-negative
the suspect plays a *stronger* Maia band than its recorded rating baseline
(band + 200..600, sampled per game) against a clean same-band opponent, with no
substitution at all, labelled clean. This forces a detector to key on move
provenance rather than "one side plays above the nominal band".

Output schema mirrors prototype/src/format_data.py's parse_game (12-plane board
tensors, UCI move strings, "H:MM:SS" clock strings, "base+inc" Time) so the
pickles load straight through ChessGamesDataset, with the label fields listed
in design decision 9 appended. See _game_metadata() below for the exact keys.

Usage (one cell = one Maia band x one engine x one substitution rate):

    python generate_anomaly_corpus.py \
        --maia-band 1500 --engine stockfish16 --substitution-rate 0.10 \
        --num-games 2000 --output-dir out/1500_stockfish16_r010 \
        --opening-table data/opening_tables/openings_band1500.json \
        --maia-weights-dir engines/maia_weights \
        --lc0-path engines/lc0/build/release/lc0 \
        --stockfish-path engines/stockfish/stockfish-ubuntu-x86-64-avx2 \
        --lc0-net-path engines/lc0_networks/small.pb.gz

The v2 driver run_anomaly_corpus_v2.sh loops this over the full grid. One JSON
SUMMARY line is printed at the end for cost/coverage extrapolation.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import pickle
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine

_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
from format_data import board_to_array  # noqa: E402


MAX_PLIES_DEFAULT = 100
SF_DEPTHS = (14, 16, 18, 20)
LC0_NODES = (200, 400, 800, 1600)
HARD_NEG_DELTAS = (200, 400, 600)
BAND_LO, BAND_HI = 1100, 1900

_P_LINE = re.compile(r"^info string (\S+)\s+\(\s*\d+\s*\)\s+N:.*\(P:\s*([-\d.]+)%\)")


def stable_seed(*parts) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# --------------------------------------------------------------------------- #
# Shared clock model (analysis/scripts/synthesize_anomaly_clocks.py), imported
# not copied so the countdown model stays single-sourced. Pure stdlib.
# --------------------------------------------------------------------------- #
def _import_clock_module(extra_dir: str | None):
    candidates = []
    if extra_dir:
        candidates.append(Path(extra_dir))
    candidates += [
        _SRC_DIR.parents[1] / "analysis" / "scripts",   # repo layout
        _SRC_DIR.parent.parent / "analysis" / "scripts",
        Path.home() / "Bacsain" / "thesis2" / "analysis" / "scripts",
    ]
    for c in candidates:
        if (c / "synthesize_anomaly_clocks.py").exists():
            sys.path.insert(0, str(c))
            import synthesize_anomaly_clocks as m  # type: ignore

            return m, str(c)
    raise ImportError(
        "synthesize_anomaly_clocks.py not found; pass --clock-module-dir. "
        f"looked in: {[str(c) for c in candidates]}"
    )


# --------------------------------------------------------------------------- #
# Maia via raw UCI: policy read + temperature sampling
# --------------------------------------------------------------------------- #
class MaiaPolicyEngine:
    """A single lc0 process running one Maia weight file, driven over raw UCI.

    lc0 v0.31 has no search-time temperature option, so temperature sampling of
    Maia's move is done here: `go nodes 1` with VerboseMoveStats on prints the
    root policy prior P for every legal move; `sample()` draws a move with
    weight ``P ** (1 / tau)`` and `argmax()` returns the top-P (== lc0
    `bestmove`) move.
    """

    def __init__(self, lc0_path: Path, weights_path: Path, io_timeout: float = 60.0):
        self.io_timeout = io_timeout
        self.proc = subprocess.Popen(
            [str(lc0_path), f"--weights={weights_path}"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        # A background reader thread decouples us from TextIOWrapper buffering
        # (mixing select() with buffered readline() drops lines).
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._send("uci")
        self._read_until("uciok", timeout=self.io_timeout)
        for opt in ("setoption name Threads value 1",
                    "setoption name VerboseMoveStats value true"):
            self._send(opt)
        self._send("isready")
        self._read_until("readyok", timeout=self.io_timeout)

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)  # EOF sentinel

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise EOFError(f"lc0 stdin closed while sending {line!r}") from exc

    def _readline(self, timeout: float | None = None) -> str:
        try:
            line = self._q.get(timeout=timeout if timeout is not None else self.io_timeout)
        except queue.Empty:
            raise TimeoutError("lc0 did not respond within io_timeout")
        if line is None:
            raise EOFError("lc0 stdout closed")
        return line

    def _read_until(self, token: str, timeout: float | None = None) -> list[str]:
        out = []
        while True:
            line = self._readline(timeout=timeout)
            out.append(line)
            if line.strip() == token or line.startswith(token + " "):
                return out

    def _policy(self, board: chess.Board) -> tuple[dict[str, float], str | None]:
        self._send(f"position fen {board.fen()}")
        self._send("go nodes 1")
        policy: dict[str, float] = {}
        best: str | None = None
        while True:
            line = self._readline()
            if line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    best = parts[1]
                break
            m = _P_LINE.match(line)
            if m:
                uci, p = m.group(1), m.group(2)
                try:
                    mv = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                if mv in board.legal_moves:
                    policy[uci] = max(0.0, float(p))
        return policy, best

    def argmax(self, board: chess.Board) -> chess.Move | None:
        _, best = self._policy(board)
        if best is None:
            return None
        return chess.Move.from_uci(best)

    def sample(self, board: chess.Board, tau: float, rng: random.Random) -> chess.Move | None:
        policy, best = self._policy(board)
        if not policy:
            return chess.Move.from_uci(best) if best else None
        moves = list(policy.keys())
        inv = 1.0 / max(tau, 1e-6)
        weights = [policy[m] ** inv for m in moves]
        total = sum(weights)
        if total <= 0:
            return chess.Move.from_uci(best) if best else chess.Move.from_uci(moves[0])
        cum, acc = [], 0.0
        for w in weights:
            acc += w / total
            cum.append(acc)
        u = rng.random()
        i = min(bisect.bisect_left(cum, u), len(moves) - 1)
        return chess.Move.from_uci(moves[i])

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def open_stockfish_engine(stockfish_path: Path) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci([str(stockfish_path)])
    engine.configure({"Threads": 1, "Hash": 16})
    return engine


def open_lc0_cheater_engine(lc0_path: Path, weights_path: Path) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci([str(lc0_path), f"--weights={weights_path}"])
    engine.configure({"Threads": 1})
    return engine


# --------------------------------------------------------------------------- #
# Opening tables
# --------------------------------------------------------------------------- #
class OpeningTable:
    """A per-band opening-line frequency table with temperature-weighted
    sampling. Sampling weight for a line with raw frequency c is
    ``c ** temperature``; temperature < 1 (default 0.7) flattens toward the
    long tail so rare-but-real lines still appear, per the design.
    """

    def __init__(self, path: Path, temperature: float):
        data = json.loads(Path(path).read_text())
        self.band = data.get("band")
        self.line_plies = data.get("line_plies", 12)
        raw = data["lines"]  # [[uci_str, count], ...]
        self.lines = [tuple(item[0].split()) for item in raw]
        weights = [float(item[1]) ** temperature for item in raw]
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"opening table {path} has non-positive total weight")
        self._cum, acc = [], 0.0
        for w in weights:
            acc += w / total
            self._cum.append(acc)
        self.temperature = temperature
        self.n_lines = len(self.lines)
        if self.n_lines == 0:
            raise ValueError(f"opening table {path} has no lines")

    def sample(self, rng: random.Random) -> tuple[str, ...]:
        u = rng.random()
        i = bisect.bisect_left(self._cum, u)
        return self.lines[min(i, self.n_lines - 1)]


def replay_opening(line: tuple[str, ...], cutoff: int) -> tuple[chess.Board, list, list, list]:
    """Replay the first `cutoff` plies of a UCI opening line onto a fresh board.
    Returns (board, positions, moves, clock_placeholder). Raises ValueError if a
    move does not apply (caller resamples)."""
    board = chess.Board()
    positions, moves = [], []
    for uci in line[:cutoff]:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"illegal opening move {uci} at ply {len(moves)}")
        board.push(mv)
        positions.append(board_to_array(board))
        moves.append(uci)
    return board, positions, moves, [None] * len(moves)


# --------------------------------------------------------------------------- #
# One game
# --------------------------------------------------------------------------- #
def play_one_game(
    *,
    suspect_maia: MaiaPolicyEngine,
    opponent_maia: MaiaPolicyEngine,
    cheater_engine: chess.engine.SimpleEngine | None,
    cheater_limit: chess.engine.Limit | None,
    substitution_rate: float,
    suspect_is_white: bool,
    opening_line: tuple[str, ...],
    opening_cutoff: int,
    maia_temp: float,
    maia_temp_cutoff_idx: int,
    sub_eligible_from_idx: int,
    resample_until_differs: bool,
    resample_cap: int,
    max_plies: int,
    move_rng: random.Random,
    sub_rng: random.Random,
) -> dict:
    """Play one synthetic game. Substitution: from ply index
    `sub_eligible_from_idx` onward, on the suspect's move, a running `owed`
    counter is incremented with probability `substitution_rate`. While
    `owed > 0` an engine move is drawn; if it differs from Maia's argmax the
    move is substituted and `owed` decremented. If it does not differ and
    resample_until_differs is set, the substitution is deferred (Maia's move is
    played) for up to `resample_cap` consecutive deferrals, after which the
    engine move is played anyway (recorded as a substitution with
    engine_differs_from_maia=False)."""

    board, positions, moves, clocks = replay_opening(opening_line, opening_cutoff)
    n_book = len(moves)
    move_is_substituted = [False] * n_book
    engine_differs: list[bool | None] = [None] * n_book
    maia_argmax_move: list[str | None] = [None] * n_book

    owed = 0
    consec_deferrals = 0
    n_suspect_plies = 0
    n_eligible_plies = 0

    ply = n_book
    while not board.is_game_over() and ply < max_plies:
        suspect_to_move = (board.turn == chess.WHITE) == suspect_is_white
        mover = suspect_maia if suspect_to_move else opponent_maia

        substituted = False
        differs: bool | None = None
        argmax_uci: str | None = None
        eligible = (suspect_to_move and ply >= sub_eligible_from_idx
                    and cheater_engine is not None)

        if suspect_to_move:
            n_suspect_plies += 1
        if eligible:
            n_eligible_plies += 1
            if sub_rng.random() < substitution_rate:
                owed += 1

        if eligible and owed > 0:
            maia_move = mover.argmax(board)
            argmax_uci = maia_move.uci() if maia_move else None
            eng_res = cheater_engine.play(board, cheater_limit)
            eng_move = eng_res.move
            if eng_move is None:
                played_move = maia_move
            elif argmax_uci is not None and eng_move.uci() != argmax_uci:
                played_move = eng_move
                substituted, differs = True, True
                owed -= 1
                consec_deferrals = 0
            elif resample_until_differs and consec_deferrals < resample_cap:
                played_move = maia_move            # defer; keep `owed`
                consec_deferrals += 1
            else:
                played_move = eng_move
                substituted, differs = True, False
                owed -= 1
                consec_deferrals = 0
        elif ply < maia_temp_cutoff_idx:
            played_move = mover.sample(board, maia_temp, move_rng)
        else:
            played_move = mover.argmax(board)
            if suspect_to_move and ply >= sub_eligible_from_idx:
                argmax_uci = played_move.uci() if played_move else None

        if played_move is None:
            break
        board.push(played_move)
        positions.append(board_to_array(board))
        moves.append(played_move.uci())
        clocks.append(None)
        move_is_substituted.append(substituted)
        engine_differs.append(differs)
        maia_argmax_move.append(argmax_uci)
        ply += 1

    n_substituted = sum(1 for x in move_is_substituted if x)
    n_differs = sum(1 for x in engine_differs if x is True)
    rate_info = {
        "n_suspect_plies": n_suspect_plies,
        "n_eligible_plies": n_eligible_plies,
        "n_substituted": n_substituted,
        "n_differs": n_differs,
        "nominal": (n_substituted / n_eligible_plies) if n_eligible_plies else 0.0,
        "effective": (n_differs / n_eligible_plies) if n_eligible_plies else 0.0,
    }

    return {
        "Positions": positions,
        "Moves": moves,
        "Clocks": clocks,  # placeholder Nones; filled by caller
        "Result": board.result(claim_draw=True),
        "move_is_substituted": move_is_substituted,
        "engine_differs_from_maia": engine_differs,
        "maia_argmax_move": maia_argmax_move,
        "nominal_vs_effective_rate": rate_info,
        "n_book_plies": n_book,
    }


def _game_metadata(game: dict, *, args, suspect_is_white: bool, base: int, inc: int,
                   opening_line: tuple[str, ...], opening_cutoff: int,
                   engine_setting: dict | None, suspect_band: int, opponent_band: int,
                   game_index: int) -> None:
    game["WhiteElo"] = args.maia_band
    game["BlackElo"] = args.maia_band
    game["Time"] = f"{base}+{inc}"
    game["suspect_color"] = "white" if suspect_is_white else "black"
    game["engine"] = None if args.engine == "none" else args.engine
    game["engine_setting"] = engine_setting
    game["substitution_rate"] = 0.0 if args.engine == "none" else args.substitution_rate
    game["maia_band"] = args.maia_band
    game["suspect_band"] = suspect_band
    game["opponent_band"] = opponent_band
    game["hard_negative"] = bool(args.hard_negative)
    game["time_control"] = {"base": base, "inc": inc}
    game["opening_line"] = list(opening_line[:opening_cutoff])
    game["opening_line_full"] = list(opening_line)
    game["maia_temperature"] = args.maia_temp
    game["maia_temp_cutoff_ply"] = args.maia_temp_cutoff_ply
    game["clock_mode"] = args.clock_mode
    game["corpus_version"] = "v2"
    game["global_seed"] = args.global_seed
    game["game_index"] = game_index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maia-band", type=int, required=True, choices=list(range(1100, 2000, 100)))
    parser.add_argument("--engine", choices=["stockfish16", "lc0", "none"], required=True,
                        help="'none' generates a 0%% clean-control / hard-negative case (no substitution).")
    parser.add_argument("--substitution-rate", type=float, required=True)
    parser.add_argument("--num-games", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maia-weights-dir", type=Path, required=True)
    parser.add_argument("--lc0-path", type=Path, required=True)
    parser.add_argument("--stockfish-path", type=Path, required=True)
    parser.add_argument("--lc0-net-path", type=Path, help="reference net for the Lc0 'cheater' side")

    parser.add_argument("--opening-table", type=Path, required=True,
                        help="openings_band{B}.json for this band (build_opening_tables.py)")
    parser.add_argument("--opening-temperature", type=float, default=0.7,
                        help="line weight = count ** T; T<1 flattens toward the tail (default 0.7)")
    parser.add_argument("--opening-cutoff-min", type=int, default=6)
    parser.add_argument("--opening-cutoff-max", type=int, default=10)

    parser.add_argument("--maia-temp", type=float, default=0.9,
                        help="temperature for Maia policy sampling over the opening (default 0.9)")
    parser.add_argument("--maia-temp-cutoff-ply", type=int, default=17,
                        help="first ply (1-based) that uses argmax Maia (default 17)")

    parser.add_argument("--sub-eligible-from-ply", type=int, default=17,
                        help="first ply (1-based) eligible for substitution (default 17)")
    parser.add_argument("--no-resample-until-differs", dest="resample_until_differs",
                        action="store_false")
    parser.add_argument("--resample-cap", type=int, default=8)

    parser.add_argument("--engine-strength", choices=["sample", "fixed"], default="sample")
    parser.add_argument("--stockfish-depth", type=int, default=18,
                        help="used only with --engine-strength fixed")
    parser.add_argument("--stockfish-time-cap-ms", type=int, default=1500,
                        help="per-move wall-time ceiling on the Stockfish substitution search")
    parser.add_argument("--lc0-nodes", type=int, default=800,
                        help="used only with --engine-strength fixed")

    parser.add_argument("--clock-mode", choices=["countdown", "constant"], default="countdown")
    parser.add_argument("--clock-module-dir", default=None)

    parser.add_argument("--hard-negative", action="store_true",
                        help="strength-mismatch clean cell: suspect plays a stronger Maia band "
                             "(band + 200..600, sampled per game) vs a clean same-band opponent.")

    parser.add_argument("--max-plies", type=int, default=MAX_PLIES_DEFAULT)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--io-timeout", type=float, default=60.0)
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    if args.engine == "lc0" and not args.lc0_net_path:
        parser.error("--lc0-net-path is required when --engine lc0")
    if args.hard_negative and args.engine != "none":
        parser.error("--hard-negative requires --engine none (no substitution in these cells)")
    if args.opening_cutoff_min < 1 or args.opening_cutoff_max < args.opening_cutoff_min:
        parser.error("bad --opening-cutoff range")

    clock_mod, clock_dir = _import_clock_module(args.clock_module_dir)
    clock_method = clock_mod.METHOD_CONSTANT if args.clock_mode == "constant" else clock_mod.METHOD_PARAMETRIC
    clock_params = clock_mod.ClockParams(method=clock_method)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    opening_table = OpeningTable(args.opening_table, args.opening_temperature)
    if opening_table.line_plies < args.opening_cutoff_max:
        parser.error(f"opening table has {opening_table.line_plies}-ply lines "
                     f"but --opening-cutoff-max is {args.opening_cutoff_max}")

    maia_temp_cutoff_idx = args.maia_temp_cutoff_ply - 1
    sub_eligible_from_idx = args.sub_eligible_from_ply - 1

    def open_maia(band: int) -> MaiaPolicyEngine:
        w = args.maia_weights_dir / f"maia-{band}.pb.gz"
        if not w.exists():
            parser.error(f"missing Maia weights: {w}")
        return MaiaPolicyEngine(args.lc0_path, w, io_timeout=args.io_timeout)

    opponent_maia = open_maia(args.maia_band)

    hard_neg_engines: dict[int, MaiaPolicyEngine] = {}
    hard_neg_bands: list[int] = []
    if args.hard_negative:
        for d in HARD_NEG_DELTAS:
            b = min(BAND_HI, args.maia_band + d)
            if b != args.maia_band and b not in hard_neg_engines:
                hard_neg_engines[b] = open_maia(b)
                hard_neg_bands.append(b)
        if not hard_neg_bands:
            parser.error(f"band {args.maia_band} has no stronger Maia band for a hard-negative cell")

    cheater_engine = None
    if args.engine == "stockfish16":
        cheater_engine = open_stockfish_engine(args.stockfish_path)
    elif args.engine == "lc0":
        cheater_engine = open_lc0_cheater_engine(args.lc0_path, args.lc0_net_path)

    t_start = time.monotonic()
    eff_ratios: list[float] = []
    try:
        for i in range(args.num_games):
            side_rng = random.Random(stable_seed(args.global_seed, "side", args.maia_band, i))
            suspect_is_white = side_rng.random() < 0.5

            op_rng = random.Random(stable_seed(args.global_seed, "opening", args.maia_band, i))
            for _attempt in range(12):
                line = opening_table.sample(op_rng)
                cutoff = op_rng.randint(args.opening_cutoff_min, args.opening_cutoff_max)
                try:
                    replay_opening(line, cutoff)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("could not sample a legal opening line after 12 tries")

            eng_rng = random.Random(stable_seed(args.global_seed, "engstr", args.maia_band, args.engine, i))
            engine_setting = None
            cheater_limit = None
            if cheater_engine is not None:
                if args.engine == "stockfish16":
                    depth = eng_rng.choice(SF_DEPTHS) if args.engine_strength == "sample" else args.stockfish_depth
                    engine_setting = {"axis": "depth", "value": int(depth)}
                    cheater_limit = chess.engine.Limit(depth=int(depth),
                                                       time=args.stockfish_time_cap_ms / 1000.0)
                else:
                    nodes = eng_rng.choice(LC0_NODES) if args.engine_strength == "sample" else args.lc0_nodes
                    engine_setting = {"axis": "nodes", "value": int(nodes)}
                    cheater_limit = chess.engine.Limit(nodes=int(nodes))

            suspect_band = args.maia_band
            suspect_maia = opponent_maia
            if args.hard_negative:
                hn_rng = random.Random(stable_seed(args.global_seed, "hardneg", args.maia_band, i))
                suspect_band = hn_rng.choice(hard_neg_bands)
                suspect_maia = hard_neg_engines[suspect_band]

            sub_rng = random.Random(
                stable_seed(args.global_seed, "sub", args.maia_band, args.engine, args.substitution_rate, i))
            move_rng = random.Random(stable_seed(args.global_seed, "move", args.maia_band, i))

            game = play_one_game(
                suspect_maia=suspect_maia,
                opponent_maia=opponent_maia,
                cheater_engine=cheater_engine,
                cheater_limit=cheater_limit,
                substitution_rate=args.substitution_rate if args.engine != "none" else 0.0,
                suspect_is_white=suspect_is_white,
                opening_line=line,
                opening_cutoff=cutoff,
                maia_temp=args.maia_temp,
                maia_temp_cutoff_idx=maia_temp_cutoff_idx,
                sub_eligible_from_idx=sub_eligible_from_idx,
                resample_until_differs=args.resample_until_differs,
                resample_cap=args.resample_cap,
                max_plies=args.max_plies,
                move_rng=move_rng,
                sub_rng=sub_rng,
            )

            tc_rng, tt_rng = clock_mod.game_streams(args.global_seed, args.maia_band, i)
            base, inc = clock_mod.draw_time_control(tc_rng, clock_params.tc_mix)
            clk = clock_mod.synthesize_game_clocks(
                len(game["Moves"]), base, inc, clock_params, tt_rng, substituted=None, band=args.maia_band)
            game["Clocks"] = clk["clocks"]

            ri = game["nominal_vs_effective_rate"]
            if ri["nominal"] > 0:
                eff_ratios.append(ri["effective"] / ri["nominal"])

            _game_metadata(game, args=args, suspect_is_white=suspect_is_white, base=base, inc=inc,
                           opening_line=line, opening_cutoff=cutoff, engine_setting=engine_setting,
                           suspect_band=suspect_band, opponent_band=args.maia_band, game_index=i)

            with open(args.output_dir / f"game_{i:05d}.pkl", "wb") as f:
                pickle.dump(game, f)

            if (i + 1) % args.log_every == 0:
                elapsed = time.monotonic() - t_start
                mr = (sum(eff_ratios) / len(eff_ratios)) if eff_ratios else float("nan")
                print(f"generated={i + 1}/{args.num_games} elapsed_s={elapsed:.1f} "
                      f"s_per_game={elapsed / (i + 1):.3f} eff/nom={mr:.3f}", flush=True)
    finally:
        opponent_maia.quit()
        for e in hard_neg_engines.values():
            e.quit()
        if cheater_engine is not None:
            cheater_engine.quit()

    total_elapsed = time.monotonic() - t_start
    summary = {
        "corpus_version": "v2",
        "maia_band": args.maia_band,
        "engine": args.engine,
        "hard_negative": bool(args.hard_negative),
        "substitution_rate": args.substitution_rate,
        "num_games": args.num_games,
        "clock_mode": args.clock_mode,
        "clock_module_dir": clock_dir,
        "opening_table": str(args.opening_table),
        "opening_temperature": args.opening_temperature,
        "mean_effective_over_nominal": (sum(eff_ratios) / len(eff_ratios)) if eff_ratios else None,
        "total_elapsed_s": total_elapsed,
        "s_per_game": total_elapsed / args.num_games if args.num_games else 0.0,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

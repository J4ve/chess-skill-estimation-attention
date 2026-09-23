#!/usr/bin/env python3
"""Generate an adversarial PGN suite for the scan-phase equivalence check.

A real Lichess sample is ~99.6% eligible, so it barely exercises the REJECT
paths and almost never exercises the awkward corners of the eligibility rule.
Those corners are exactly where a reimplementation drifts, so they get their own
file with a hand-declared expected verdict per game.

The interesting cases, and why each one matters:

  * The `[%clk ...]` pattern is `\\[%clk\\s+([^\\]]+)\\]`. Because `\\s+`
    backtracks and `[^\\]]+` can itself match whitespace, `[%clk ]` (one space)
    does NOT match but `[%clk  ]` (two spaces) DOES. Any port that reduces this
    to a substring test for "[%clk" gets both of these wrong.
  * The 100-ply cap means a clock missing at ply 100 rejects the game while the
    identical omission at ply 101 does not.
  * `re.search` scans every start position, so a malformed `[%clk` earlier in a
    comment must not stop a well-formed one later from matching.
  * python-chess attaches a pre-move comment to the game node, not to ply 1, and
    concatenates multiple comments on one ply.
  * `has_full_clocks` requires at least one ply, so a moveless game is rejected
    even though it vacuously satisfies "every ply has a clock".

Writes <out>.pgn and <out>.expected (one 1/0 per game, same order).
"""

import argparse
from pathlib import Path

import chess
import chess.pgn


def legal_sans(n: int) -> list[str]:
    """n plies of legal SAN from the start position, deterministically.

    Uses a knight shuffle (Nf3 Nf6 Ng1 Ng8 ...) rather than "first legal move
    each time". A greedy walk reaches checkmate or stalemate well before 150
    plies, and any board reset mid-sequence emits SAN that is illegal in the
    preceding position: python-chess then rejects those moves, so the ply counts
    the two scanners see would no longer be the ones the test intends. The
    shuffle never ends the game, so the 100-ply cap cases test what they claim.
    """
    cycle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    out = [cycle[i % len(cycle)] for i in range(n)]

    # Assert legality rather than trusting the pattern.
    board = chess.Board()
    for san in out:
        board.push(board.parse_san(san))
    return out


CLK = "[%clk 0:03:00]"


def movetext(sans: list[str], comments: list[str | None], leading: str | None = None) -> str:
    """Emit movetext by hand so the exact comment bytes are under our control."""
    parts = []
    if leading is not None:
        parts.append("{ %s }" % leading)
    for i, san in enumerate(sans):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
        c = comments[i]
        if c is not None:
            parts.append("{ %s }" % c)
    parts.append("*")
    return " ".join(parts)


def game(headers: dict, body: str) -> str:
    base = {
        "Event": "Rated Blitz game",
        "White": "alice",
        "Black": "bob",
        "Result": "*",
        "WhiteElo": "1500",
        "BlackElo": "1520",
        "TimeControl": "180+0",
    }
    base.update(headers)
    head = "".join(f'[{k} "{v}"]\n' for k, v in base.items())
    return head + "\n" + body + "\n\n"


def build() -> list[tuple[str, str, bool]]:
    """(name, pgn_text, expected_eligible)"""
    cases: list[tuple[str, str, bool]] = []
    s10 = legal_sans(10)
    s150 = legal_sans(150)

    def add(name, headers, body, expected):
        cases.append((name, game(headers, body), expected))

    # --- baseline shapes -----------------------------------------------------
    add("all-plies-clocked", {}, movetext(s10, [CLK] * 10), True)
    add("no-clocks-at-all", {}, movetext(s10, [None] * 10), False)
    add("missing-clock-first-ply", {}, movetext(s10, [None] + [CLK] * 9), False)
    add("missing-clock-last-ply", {}, movetext(s10, [CLK] * 9 + [None]), False)
    add("missing-clock-middle-ply", {}, movetext(s10, [CLK] * 4 + [None] + [CLK] * 5), False)
    add("no-moves-at-all", {}, "*", False)
    add("single-ply-clocked", {}, movetext(s10[:1], [CLK]), True)
    add("single-ply-unclocked", {}, movetext(s10[:1], [None]), False)

    # --- the 100-ply cap boundary -------------------------------------------
    add("150-plies-clocks-only-through-100",
        {}, movetext(s150, [CLK] * 100 + [None] * 50), True)
    add("150-plies-missing-at-ply-100",
        {}, movetext(s150, [CLK] * 99 + [None] + [CLK] * 50), False)
    add("150-plies-missing-at-ply-101",
        {}, movetext(s150, [CLK] * 100 + [None] + [CLK] * 49), True)
    add("exactly-100-plies-all-clocked", {}, movetext(s150[:100], [CLK] * 100), True)

    # --- regex corners -------------------------------------------------------
    add("clk-one-space-empty", {}, movetext(s10, ["[%clk ]"] * 10), False)
    add("clk-two-spaces-empty", {}, movetext(s10, ["[%clk  ]"] * 10), True)
    add("clk-no-space", {}, movetext(s10, ["[%clk0:03:00]"] * 10), False)
    add("clk-unterminated", {}, movetext(s10, ["[%clk 0:03:00"] * 10), False)
    add("clk-tab-separator", {}, movetext(s10, ["[%clk\t0:03:00]"] * 10), True)
    add("malformed-then-valid", {}, movetext(s10, ["[%clk] junk [%clk 0:03:00]"] * 10), True)
    add("eval-then-clk", {}, movetext(s10, ["[%eval 0.24] [%clk 0:03:00]"] * 10), True)
    add("clk-then-eval", {}, movetext(s10, ["[%clk 0:03:00] [%eval -1.2]"] * 10), True)
    add("eval-only", {}, movetext(s10, ["[%eval 0.24]"] * 10), False)
    add("split-comments-clk-second", {},
        movetext(s10, ["[%eval 0.1] } { [%clk 0:03:00]"] * 10), True)
    add("leading-game-comment-only", {},
        movetext(s10, [None] * 10, leading=CLK), False)
    add("leading-game-comment-plus-clocks", {},
        movetext(s10, [CLK] * 10, leading="a pre-move note"), True)

    # --- header predicate ----------------------------------------------------
    add("casual-not-rated", {"Event": "Casual Blitz game"}, movetext(s10, [CLK] * 10), False)
    add("variant-chess960", {"Variant": "Chess960"}, movetext(s10, [CLK] * 10), False)
    add("variant-standard-explicit", {"Variant": "Standard"}, movetext(s10, [CLK] * 10), True)
    add("elo-question-mark", {"WhiteElo": "?"}, movetext(s10, [CLK] * 10), False)
    add("elo-empty", {"BlackElo": ""}, movetext(s10, [CLK] * 10), False)
    add("elo-non-numeric", {"WhiteElo": "15o0"}, movetext(s10, [CLK] * 10), False)
    add("tc-correspondence", {"TimeControl": "-"}, movetext(s10, [CLK] * 10), False)
    add("tc-no-increment-field", {"TimeControl": "300"}, movetext(s10, [CLK] * 10), False)
    add("tc-three-fields", {"TimeControl": "300+3+1"}, movetext(s10, [CLK] * 10), False)
    add("tc-non-numeric", {"TimeControl": "300+x"}, movetext(s10, [CLK] * 10), False)
    add("tc-zero-increment", {"TimeControl": "300+0"}, movetext(s10, [CLK] * 10), True)

    # --- variations ----------------------------------------------------------
    # has_full_clocks walks node.variation(0), the mainline. A sideline with no
    # clocks must not reject a fully-clocked mainline.
    mv = movetext(s10, [CLK] * 10)
    mv_with_side = mv.replace(" *", " ( 1... e5 { no clock here } ) *")
    add("sideline-without-clocks", {}, mv_with_side, True)

    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="output stem")
    args = ap.parse_args()

    cases = build()
    pgn = args.out.with_suffix(".pgn")
    exp = args.out.with_suffix(".expected")
    names = args.out.with_suffix(".names")

    pgn.write_text("".join(text for _, text, _ in cases))
    exp.write_text("".join("1\n" if ok else "0\n" for _, _, ok in cases))
    names.write_text("".join(f"{n}\n" for n, _, _ in cases))
    print(f"wrote {len(cases)} cases to {pgn}")
    print(f"expected-eligible: {sum(1 for _, _, ok in cases if ok)}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Faster corpus-scan research: measured, not projected

Task: `CLAUDE_TASK.md` Part C. Written 2026-08-17.

**Recommendation: GO, behind a flag, with three conditions before it becomes the
default.** A Rust prototype exists, it is 161x faster than the current scanner on
identical input on the same machine, and it agrees with the current scanner on
626,765 out of 626,765 real games plus a 36-case adversarial suite. It has not yet
run on the HPC or on a complete 30 GB month, so it is not production-blessed.

---

## 1. What the scan phase actually costs today

The scan decides one thing per game: is this game eligible?

    is_eligible(game)                prototype/src/preprocess_lichess.py:66-80
    and has_full_clocks(game, 100)   prototype/src/preprocess_lichess.py:83-99

`has_full_clocks` is already the fast path added on 2026-08-17: it walks the mainline
checking for `[%clk]` and skips the `board.push()` and 12-plane tensor build. Its
contract is that it returns True exactly when `parse_game`
(`prototype/src/format_data.py:39-95`) would return non-None.

The cost is not the clock check. The cost is that `chess.pgn.read_game` fully parses
every game, building a node tree and validating every SAN move against a board, for
all ~92.2M games in a month, so that ~30,000 can be kept. The eligibility decision
never looks at the parsed moves.

Scale check, because it sets the size of the prize. 2021-06 is 29,730,565,470 bytes
and about 92.2M games. At the ~500 games/s the HPC reports post-fix, one month is
about **51 hours** of single-core scanning. At `CONCURRENCY=5` the 34-month list is
roughly 2 to 3 weeks, which matches what the pipeline is being asked to do.

## 2. The candidate

`niklasf/pgn-reader` (Rust), by the same author as `python-chess`. The task cited a
public benchmark of 25 min (Python) versus 15 s (Rust) for tokenizing 1M games. That
benchmark is not our workload, so it was treated as a reason to prototype, not as a
result.

Two things make our workload even more favourable than that benchmark:

- The header predicate can be decided before the movetext is touched. `Visitor::
  end_headers` returns `Skip(true)` for any game failing `is_eligible`, so
  `pgn-reader` skips its movetext entirely instead of tokenizing it.
- We never need a board. `has_full_clocks` counts plies and inspects comments, so the
  prototype does no move validation at all.

## 3. The prototype

`analysis/scanner-rs/` (`clkscan`). Roughly 300 lines, two dependencies
(`pgn-reader 0.26`, `zstd 0.13`). It streams the `.pgn.zst` directly, never
decompressing to disk, and prints `scanned=`, `eligible=` and `games_per_sec=`.

The eligibility rule is ported literally, including two details that a casual port
gets wrong:

- The clock pattern is `\[%clk\s+([^\]]+)\]`. Because `\s+` backtracks and `[^\]]+`
  can itself match whitespace, `[%clk ]` (one space) does **not** match while
  `[%clk  ]` (two spaces) **does**. A substring search for `"[%clk"` gets both wrong.
- The 100-ply cap means a clock missing at ply 100 rejects the game while the same
  omission at ply 101 does not.

Both are covered by the edge-case suite in section 5.

## 4. Measured throughput

Same machine (Apple Silicon, macOS 25.2.0), same bytes, single core.

Sample: the first 200,000,000 bytes of
`lichess_db_standard_rated_2021-06.pgn.zst`, fetched with a ranged HTTP GET, which
is 626,765 complete games. Truncating the sample is the methodology the task
sanctions, and it does not distort the rate: both scanners read the same prefix.

| Scanner | Games | Wall time | Games/sec |
|---|---|---|---|
| Python, the repo's own `is_eligible` + `has_full_clocks` | 626,770 | 647.744 s | **968** |
| Rust `clkscan` | 626,765 | 4.003 s | **156,580** |

**161x on identical input.** Against the ~500 games/s figure quoted for the HPC the
ratio would be about 313x, but that compares different hardware, so the 161x number
is the one to quote.

The Python side is not a reimplementation: `analysis/scripts/scan_equivalence_check.py`
imports `is_eligible`, `has_full_clocks` and `stream_games` from
`preprocess_lichess.py` itself, so this is the production predicate being measured.

## 5. Equivalence evidence

Speed is worthless if the sample changes. Two independent checks.

**Real data, whole sample.** Both scanners emitted one verdict per game, in file
order, and the streams were compared line by line:

    626,765 / 626,765 verdicts identical, 0 disagreements

**Adversarial edge cases.** A real Lichess sample is about 99.5% eligible, so it
barely exercises the reject paths. `analysis/scripts/make_edge_case_pgns.py` builds 36
cases targeting the corners, each with a hand-declared expected verdict, and all three
agree on all 36:

    hand-declared expectation == Python == Rust,  36/36, 0 mismatches

Cases include: the `[%clk ]` / `[%clk  ]` pair, a clock missing exactly at ply 100
versus ply 101, a malformed `[%clk` followed by a valid one in the same comment,
comments split across two brace groups, a pre-move game comment (which attaches to the
game node and not to ply 1), a moveless game, a sideline without clocks under a fully
clocked mainline, and the full header predicate (casual, Chess960, `WhiteElo "?"`,
`TimeControl "-"`, `"300"`, `"300+3+1"`, `"300+x"`).

### The one real divergence, and it is in Python's favour to fix

The two scanners disagreed on the total game count: Python 626,770, Rust 626,765.

This is not a predicate disagreement. All 626,765 games both scanners read got
identical verdicts. The five extra games are Python decoding **past a provably
incomplete zstd frame** at the end of the deliberately truncated sample, and reporting
`truncated=False`.

Root cause, verified directly:

    zstandard.ZstdDecompressor.stream_reader on a file truncated to 50%
      -> returned EOF cleanly, NO exception raised
    the same input via decompressobj()
      -> .eof == False, i.e. the frame is detectably incomplete

`stream_games` (`preprocess_lichess.py:52-63`) uses `stream_reader`, so **the current
pipeline cannot detect a truncated archive.** A month whose download stopped early is
processed as if complete: the reservoir fills from the prefix only, the kept-count gate
of 30,000 still passes, `.corpus_done` is written, and the raw archive is deleted. The
resulting sample is silently biased toward the start of the month, which is precisely
the bias the reservoir sampler was introduced to remove
(`preprocess_lichess.py:11-16`).

This is an existing-pipeline bug, found while benchmarking rather than while looking
for it. `analysis/corpus-parallel-pipeline-review.md` reached the same conclusion
independently from a code reading (finding F1) and reproduced it end to end. `clkscan`
detects and reports truncation, and `preprocess_fast.py` refuses to run on a truncated
archive rather than producing a partial sample.

## 6. End-to-end: the same games out, not just the same verdicts

Verdict equivalence proves the scan agrees. It does not prove a faster pipeline keeps
the same 30,000 games. `analysis/scripts/preprocess_fast.py` closes that gap.

The insight that makes it safe: the reservoir sampler draws from `rng` **only** for an
eligible game once the reservoir is full, and the draw is `rng.randint(0, eligible-1)`
(`preprocess_lichess.py:147-154`). The random sequence is therefore a function of the
eligibility bitstream alone, never of any game's contents. So pass 1's verdict stream
is sufficient to replay Algorithm R exactly and land on the same reservoir in the same
slot order. Sampling is not reimplemented, it is replayed; the kept games are still
parsed by the same `parse_game`.

Verified on a complete (non-truncated) 60,000-game sample, `--max-games 2000`:

| | `preprocess_lichess.py` | `preprocess_fast.py` |
|---|---|---|
| scanned / eligible / kept | 60000 / 59729 / 2000 | 60000 / 59729 / 2000 |
| time-control counts | ub 26, bul 721, bli 972, rap 264, cla 17 | identical |
| `.sampling_summary.json` | | identical |
| wall time | 82.7 s | 19.5 s |

Content comparison of all 2000 kept games (same games, same order, same
`Moves`/`Clocks`/headers, tensors compared with `torch.equal`):

    2000 / 2000 identical, 0 content mismatches

The `.pkl` files are **not** byte-identical, and that is a property of the existing
pipeline rather than of this change: torch serialises each tensor's storage under a key
derived from its memory address. Running the unmodified `preprocess_lichess.py` twice
on the same input with the same seed gives 300/300 files differing byte-wise and 0/300
differing in content. Byte-identity is therefore not an available criterion for
anything here; content-identity is, and it holds.

## 7. Projected effect on a real month

For a 92.2M-game month, single core. The measured rates are the inputs; the totals are
arithmetic, not measurements.

| Stage | Rate | Time |
|---|---|---|
| Current: full parse of every game | 500 g/s (HPC-reported) | **~51 h** |
| `clkscan` pass 1 | 156,580 g/s | ~10 min |
| `skip_game` pass 2 over non-selected games | 28,133 g/s (measured) | ~55 min |
| `parse_game` on the 30,000 kept | 968 g/s | ~31 s |
| **`preprocess_fast.py` total** | | **~65 min** |

About **47x** end to end. Pass 2 is now the bottleneck, and it is python-chess's own
`skip_game`, so no new parsing logic was introduced to get there. If that hour still
matters later, the next step is to have `clkscan` emit the selected games' raw PGN text
in pass 1, dropping pass 2 to seconds; that is a larger change and was not attempted.

## 8. Alternatives considered

- **`pgn-extract`.** Ruled out in the task brief: no clock-annotation filtering, no
  confirmed zstd streaming. Custom work either way.
- **Just raise `CONCURRENCY`.** The honest alternative, since it needs no new code.
  Rejected as strictly worse: it does not reduce per-month latency, it multiplies peak
  raw staging (already about 153 GB at `CONCURRENCY=5` per
  `analysis/corpus-parallel-pipeline-review.md` F6, on a `/home` at ~92% full), and it
  adds CPU pressure to a box with 38 other users. A faster scanner improves wall time,
  disk pressure and neighbourliness at once.
- **PyPy.** Plausibly 3 to 5x on python-chess, not 100x, and the `zstandard` binding
  situation is awkward. Not worth a toolchain change for that ratio.
- **Keep Python.** Defensible only if the 2 to 3 week estimate is acceptable. It
  probably is, given the deadlines. The reason to do this anyway is that it converts a
  three-week unattended run into a roughly one-day run, which removes most of the
  exposure that Parts A, D and E are all about.

## 9. Build and run

The binary is self-contained; nothing else in the repo changes.

    cd analysis/scanner-rs
    cargo build --release
    # binary at target/release/clkscan

    # scan a month
    ./target/release/clkscan --input data/raw_zst/lichess_db_standard_rated_2021-06.pgn.zst

    # emit a per-game verdict stream for equivalence checking
    ./target/release/clkscan --input MONTH.pgn.zst --verdicts rs.txt

    # the Python oracle, using the repo's own predicates
    python analysis/scripts/scan_equivalence_check.py --input MONTH.pgn.zst --verdicts py.txt
    cmp rs.txt py.txt        # must be identical

    # the adversarial suite
    python analysis/scripts/make_edge_case_pgns.py --out /tmp/edge
    zstd -q -f /tmp/edge.pgn -o /tmp/edge.pgn.zst
    ./analysis/scanner-rs/target/release/clkscan --input /tmp/edge.pgn.zst --verdicts /tmp/rs.txt
    python analysis/scripts/scan_equivalence_check.py --input /tmp/edge.pgn.zst --verdicts /tmp/py.txt
    diff /tmp/edge.expected /tmp/rs.txt && diff /tmp/edge.expected /tmp/py.txt

    # the fast preprocessor
    python analysis/scripts/preprocess_fast.py --input MONTH.pgn.zst \
        --output-dir out/ --max-games 30000 --clkscan analysis/scanner-rs/target/release/clkscan

`corpus_stream_parallel.sh --scanner rust` uses it when both the binary and
`preprocess_fast.py` are present, and falls back to the Python path with a log line
otherwise. The default is `--scanner python`.

## 10. What must happen before this becomes the default

Not yet blessed. Three conditions, in order.

1. **Run the equivalence check on one complete 30 GB month on the HPC.** Everything
   above is a 200 MB prefix on a Mac. The prefix is the start of the month, and Lichess
   dumps are chronological, so annotation density and game length later in the month
   are assumed rather than measured. `cmp` on the two verdict streams must be empty.
   This costs one 51-hour Python scan, once, and it is the whole basis for trusting
   every month afterwards.
2. **Build on the HPC.** There is no Rust toolchain there today. Either install one
   (`rustup`, no root needed) or ship a static `x86_64-unknown-linux-musl` binary.
   Record the exact `rustc` and crate versions used, alongside the conda pins.
3. **Re-measure on HPC hardware.** The 30 GB read comes off a shared, roughly 92%-full
   spinning `/home`. `clkscan` is fast enough that it may well become I/O bound there,
   which would cut the speedup. It would still be a large win, but the number in
   section 7 should be replaced with a measured one.

Until all three pass, `--scanner python` stays the default and the Rust path is opt-in.
A fast pipeline that silently drops or miscategorises games is worse than a slow correct
one.

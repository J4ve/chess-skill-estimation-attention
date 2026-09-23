# A one-pass Rust preprocessor for the corpus pipeline

**Status:** built, tested and verified equivalent on real Lichess data on a local
macOS machine, then HPC-validated and put into production on 2026-08-18 (see
`AGENTS.md`, which records the flip and the one deployment bug it surfaced). The
pre-flight checklist below, [What firstmate should HPC-validate](#what-firstmate-should-hpc-validate),
was the list used before the flip; results are now filled in against each item
(source: `rust-scanner-hpc-validation` and `onepass-preprocessor-hpc-validation`
firstmate reports, both 2026-08-18). All five passed. The one real divergence found
across all testing was a truncation-handling difference on an incomplete archive
(Python silently decodes past a truncated zstd frame, Rust detects and refuses it) -
it cannot occur on a complete month, where `clkscan` reports `truncated=false` and
the pipeline refuses to run on a truncated archive regardless. Two further
intentional, documented port narrowings (ASCII-only `\s`/`isdigit`, case-sensitive
header-key matching) are theoretical only - neither can fire on real Lichess data.

`--scanner rust` is now the production choice for the live corpus run, and it is
passed explicitly at launch. The script's own code default is still
`SCANNER="python"` (`corpus_stream_parallel.sh:27`), so rust is selected by the
launch flag, not by a changed default: every launch command has to carry
`--scanner rust` (see `hpc-operations-guide.md` section 5).

**Branch:** `fm/claude-rust-pass2-preprocessor` (merged to `main`)
**Date:** 2026-08-18; HPC-validation and production-flip outcomes folded in 2026-08-19

---

## 1. What this replaces, and why

Part C established `clkscan`, a Rust reimplementation of the *scan-phase
eligibility decision* (`is_eligible` + `has_full_clocks`), and `preprocess_fast.py`,
a two-pass driver around it. That fixed pass 1. It left pass 2 as the bottleneck:

| Stage | Rate | Time / month |
|---|---|---|
| `clkscan` pass 1 | 72,717 g/s (HPC, real) | ~21 min |
| `skip_game` pass 2 over non-selected games | ~28,000 g/s | ~55 min |
| `parse_game` on the ~30,000 kept | ~968 g/s | ~31 s |
| **`preprocess_fast.py` total** | | **~76 min** |

The `skip_game` walk is pure waste. `preprocess_fast.py` re-opens the archive and
drags python-chess's parser across every one of the month's ~92M games purely to
advance the stream to the ~30,000 it wants. It has to, because pass 1 only
recorded a *verdict bit* per game; the games themselves were gone by the time the
reservoir was known.

The fix Part C named and deliberately did not attempt: have the scanner keep the
selected games' text while it still has them in hand. That is what this change
does.

    preprocess_lichess.py   python-chess fully parses all ~92M games
    preprocess_fast.py      clkscan scans, then python-chess skip_games all ~92M
    preprocess_onepass.py   clkscan scans AND samples AND extracts, one pass;
                            python-chess only ever sees the ~30,000 kept

The archive is read once, decompressed once, and parsed once.

### The design decision that makes it work

Running Algorithm R *inline* is the whole trick, and it is only sound because of
the property Part C already established: the reservoir's random draws depend
solely on the running count of eligible games, never on any game's content
(`preprocess_lichess.py:152`, `rng.randint(0, eligible - 1)`). So at the moment
`clkscan` finishes reading a game it already knows everything the sampler needs.
There is no reason to defer the decision to a replay pass. `preprocess_fast.py`
replayed the bitstream because the bitstream was all it had kept; keeping the
game instead removes the need for the replay entirely.

The reservoir therefore holds up to `--max-games` rendered PGN strings, and the
selected games are written out in **reservoir-slot order**, which is the order
`preprocess_lichess.py` writes `game_0000000.pkl`, `game_0000001.pkl`, ... in.
Slot order is part of the equivalence contract, not an incidental detail.

---

## 2. Why the RNG is a hand-port of CPython's Mersenne Twister

This is the part most likely to look like over-engineering and is in fact the
load-bearing piece.

"The same 30,000 games in the same slot order" means reproducing
`random.Random(42)` exactly. Not "a uniformly random sample of the same size" and
not "MT19937". CPython's `random` module has three behaviours that a stock
MT19937 crate does not give you:

1. **Seeding is `init_by_array`**, over the seed's absolute value split into
   32-bit little-endian words (`_randommodule.c: random_seed`), not the plain
   `init_genrand` that most MT19937 implementations expose as `new(seed)`.
2. **`getrandbits(k)` for `k <= 32` is `genrand_uint32() >> (32 - k)`.** It takes
   the *high* k bits of one word. An implementation that masks the low bits
   instead produces a different, equally uniform, completely wrong stream.
3. **`randint(0, n-1)` is rejection sampling.** `_randbelow_with_getrandbits`
   draws `getrandbits(n.bit_length())` and *redraws* while `r >= n`. Those
   redraws consume extra words, so a single off-by-one in `bit_length`
   desynchronises every subsequent draw, not just one.

Get any of these wrong and the tool still produces a perfectly plausible sample
of exactly 30,000 games. Nothing downstream would notice. The month's archive is
deleted after processing, so nothing later could reconstruct what the sample
should have been. That is why this is a literal port with its own test suite
rather than a dependency.

`src/pyrandom.rs` implements it; `src/reservoir.rs` transcribes
`preprocess_lichess.py:147-154` line for line, including the detail that the RNG
is **not consulted at all while the reservoir is still filling**.

### Evidence: CPython equivalence tests

`analysis/scanner-rs/tests/py_random.rs`, run against vectors generated by the
interpreter itself (`analysis/scripts/gen_py_random_vectors.py`):

```
running 4 tests
test genrand_uint32_stream_matches_cpython ... ok
test getrandbits_matches_cpython_across_widths_and_seeds ... ok
test randint_below_matches_cpython ... ok
test algorithm_r_reproduces_cpython_reservoir ... ok

test result: ok. 4 passed; 0 failed
```

Concretely, these compare against real CPython output:

- 24 consecutive raw 32-bit outputs for seed 42.
- 14 mixed widths (1, 2, 3, 7, 8, 17, 31, 32, 33, 53, 64, 64, 32, 5) across seven
  seed shapes: 0, 1, 42, 12345, 2^31, 2^32 and 2^64-1. This deliberately covers
  the multi-word `getrandbits` path and multi-word seeding, neither of which the
  reservoir itself reaches, so the port is correct for anyone who reuses it.
- 3,000 consecutive `randint(0, n-1)` draws for n = 30,001 .. 33,000, plus 500
  draws for n = 1 .. 500 where the narrow bit widths hit the rejection loop
  hardest.
- Four full Algorithm R replays over a 400,000-entry synthetic eligibility
  bitstream (28,000 eligible), at reservoir sizes 1000/1000/30000/137 and seeds
  42/7/42/99, compared slot by slot against the Python implementation's output.

---

## 3. Equivalence evidence on real data

Two independent legs, because each covers what the other cannot.

### Leg A: against the true production path, `preprocess_lichess.py`

This is the one that matters most: it compares against the code that actually
built the corpus, not against another optimisation of it.

**Input:** the first 200,000 complete games of the real
`lichess_db_standard_rated_2017-04.pgn.zst`, recompressed as a self-contained
non-truncated archive.

| | `preprocess_lichess.py` | `preprocess_onepass.py` |
|---|---|---|
| scanned | 200,000 | 200,000 |
| eligible | 198,710 | 198,710 |
| kept | 1,000 | 1,000 |
| time-control mix | ultrabullet=45 bullet=247 blitz=446 rapid=232 classical=30 | identical |

With `--max-games 1000` against 198,710 eligible games, the reservoir fills and
then cycles **198x over**, so this exercises the replacement path hard rather
than just the fill path.

```
games compared:   1000
games differing:  0
other problems:   0
RESULT: IDENTICAL - 1000/1000 games match on headers, Moves, Clocks and all position tensors
```

Repeated at the **production reservoir size** on the same input, so the
`--max-games 30000` configuration the corpus run actually uses is covered
directly rather than by extrapolation from a smaller reservoir:

| | `preprocess_lichess.py` | `preprocess_onepass.py` |
|---|---|---|
| scanned / eligible | 200,000 / 198,710 | 200,000 / 198,710 |
| kept | 30,000 | 30,000 |
| time-control mix | ub=1369 b=7291 bl=13744 r=6965 c=631 | identical |
| wall clock | 6 min 18 s (529 g/s) | 3 min 34 s |

```
games compared:   30000
games differing:  0
other problems:   0
RESULT: IDENTICAL - 30000/30000 games match on headers, Moves, Clocks and all position tensors
```

A third comparison, `preprocess_fast.py` (Part C) against
`preprocess_onepass.py` on the same input at `--max-games 1000`, also returned
1000/1000 identical. So all three implementations agree with each other, and
Part C's script still works unmodified against the new binary, which is what the
pipeline's middle fallback rung depends on.

### Leg B: full month, production reservoir size

**Input:** the complete, non-truncated `lichess_db_standard_rated_2017-04.pgn.zst`,
3,481,200,992 bytes, downloaded in full and size-verified against the server's
`content-length`. `truncated=false` on every run.

**Reference:** `preprocess_fast.py` (Part C), which Part C proved equivalent to
`preprocess_lichess.py` and which Leg A re-confirms here.

```
scanned    11,348,506
eligible   11,273,096   (99.34%)
kept           30,000
```

11,273,096 eligible against a 30,000 reservoir means Algorithm R fills once and
then cycles **375x over**, with roughly 375,000 successful replacements. This is
the condition this change's own task brief asked for ("eligible needs to be
several multiples of 30,000"), exceeded by two orders of magnitude. That brief was
deleted with the task by the leave-no-trace convention; it is recoverable at
`git show 1dab1ed:CLAUDE_TASK.md`.

```
games compared:   30000
games differing:  0
other problems:   0
RESULT: IDENTICAL - 30000/30000 games match on headers, Moves, Clocks and all position tensors
```

The two `.sampling_summary.json` files are identical field for field, including
the realized time-control mix (ultrabullet=959 bullet=7589 blitz=14048
rapid=6840 classical=564).

### Leg C: the 36-case adversarial suite

`analysis/scripts/make_edge_case_pgns.py` regenerated and re-run against the new
binary, in both modes.

Scan mode, verdict stream diffed against the hand-declared expectations:

```
scanned=36 eligible=15 truncated=false
36/36 cases match
```

Select mode is checked more strictly than Part C checked scan mode: the 15
eligible cases are extracted, re-parsed and compared *by content* against what
`preprocess_lichess.py` produces from the same file.

```
games compared:   15
games differing:  0
RESULT: IDENTICAL - 15/15 games match on headers, Moves, Clocks and all position tensors
```

This leg is the one that validates the PGN re-emission specifically, because the
suite contains exactly the comment shapes a naive re-emitter mangles: `[%clk  ]`
with two spaces (matches) versus one space (does not), tab separators,
`[%clk] junk [%clk 0:03:00]` where a malformed tag precedes a valid one, comments
split across two `{}` blocks on one ply, pre-move game-node comments, and a
sideline whose moves must not be emitted at all.

### Why content equivalence and not byte equivalence

The `.pkl` files are not byte-comparable, and that is a property of the existing
pipeline rather than of anything new here: torch serialises each tensor's storage
under a key derived from its memory address, so `preprocess_lichess.py` does not
reproduce its own bytes across two runs on identical input either (Part C
measured this: 300/300 files differ byte-wise, 0/300 differ in content). The
meaningful criterion is content, and
`analysis/scripts/compare_processed_dirs.py` checks all of it: filename sequence,
the six header fields `parse_game` emits, the full `Moves` list, the full
`Clocks` list, `torch.equal` on every one of the ~100 position tensors per game,
and the `.sampling_summary.json` counts.

---

## 4. How the games are re-emitted, and the one place it is not verbatim

`clkscan` does not copy raw bytes out of the archive. `pgn-reader` gives the
visitor structured callbacks, not source offsets, so the tool **reconstructs**
PGN from what it saw: every header verbatim and in order, then the mainline SAN
tokens with their comments attached.

This is worth being explicit about, because "emit the selected games' PGN" sounds
like it should be a byte copy and is not.

What makes the reconstruction safe:

- **All headers are captured, not just the six `parse_game` reads.** That keeps a
  `FEN`/`SetUp` pair intact, so `game.board()` still starts from the right
  position, and it does not silently break the day `parse_game` reads a seventh
  header.
- **`parse_game` stores `move.uci()`, not SAN.** So even if `pgn-reader`'s SAN
  rendering differed cosmetically from the source text, the move resolves
  identically against the same board and the stored value is unchanged. The
  reconstruction only has to preserve the move *sequence*, which it does.
- **Comments are preserved byte for byte** inside the braces, so the
  `\[%clk\s+([^\]]+)\]` capture sees exactly what it saw before. Multiple
  comments on one ply are joined with a single space, which cannot affect the
  capture because the separator lies outside the `[...]`.
- **Variations are dropped**, matching `has_full_clocks` and `parse_game`, which
  both walk `node.variation(0)` only.

Deliberate divergences, both of which are reported rather than silent:

1. A `{` or `}` inside a comment would terminate the re-emitted comment early. It
   cannot occur in well-formed PGN (it would have ended the comment during the
   original parse), so any occurrence is neutralised to a space and **counted**,
   and a nonzero count prints a warning at the end of the run. Zero occurred
   across the runs reported here.
2. Invalid UTF-8 bytes are rendered as U+FFFD. This matches what the Python side
   already does: `preprocess_lichess.py:58` wraps the stream with
   `errors="replace"`, so both paths agree even on malformed input.

The pre-existing narrowing from Part C is unchanged and still applies:
`is_digits` accepts ASCII digits only, while Python's `str.isdigit()` also
accepts non-ASCII digit codepoints. Lichess Elo and TimeControl headers are
ASCII, so it cannot change a verdict on real dump data.

---

## 5. CLI and operational behaviour

The Part C prototype was a benchmark harness: hand-rolled argument parsing, no
`--help`, and no output at all during a 21-minute run. That is acceptable for a
prototype and not for something replacing a production step, so:

**Progress.** `--log-every N` (default 200,000) prints
`scanned=N eligible=M reservoir=K` to **stderr**. The shape matches
`preprocess_lichess.py:140-146` deliberately: `corpus_stream_parallel.sh:534`
already probes progress with `grep -o 'scanned=[0-9]*' ... | tail -1`, and the
pipeline merges stdout and stderr into one per-month log, so existing monitoring
keeps working with no change. stderr specifically, so stdout stays a clean
machine-readable summary for `preprocess_fast.py`'s existing parser.

**`--help` / `-h`**, listing both modes, every option, and the exit-status
meanings.

**Specific errors, not panics.** Verified behaviour:

| Situation | Message | Exit |
|---|---|---|
| missing input | `input not found: <path>` | 1 |
| non-zstd / corrupt file | `could not read a single complete game (Unknown frame descriptor). This is a malformed, corrupt or non-zstd input rather than a truncated archive; ...` | 1 |
| truncated, select mode | `ends mid-game after N games. Refusing to write a sample from a truncated archive: the reservoir would be drawn from a prefix of the month, not the month.` | 1 |
| unknown flag | `unrecognised argument "--frobnicate"` + `Try 'clkscan --help'` | 2 |
| flag missing its value | `--input requires a path` | 2 |
| non-numeric value | `--max-games: "abc" is not a number` | 2 |

The corrupt-file case is a behaviour change worth calling out. Previously *any*
read error was reported as `truncated=true` with a success exit status. A file
that yields zero complete games is not a truncated archive, it is the wrong file:
a non-zstd input, a corrupt download, or an HTML error page saved under a `.zst`
name. Reporting that as truncation sends the operator hunting for a download that
in fact completed fine.

**Truncation refusal** carries `preprocess_fast.py:87-91`'s behaviour into the
merged tool: select mode refuses a truncated archive outright, because a partial
month silently produces a partial, non-uniform sample and the archive is deleted
after processing. `--allow-truncated` overrides it for deliberate
prefix benchmarking. Scan mode's original contract is untouched, so
`preprocess_fast.py` keeps working exactly as before.

**Backwards compatibility.** The scan-mode stdout line is unchanged:

```
scanned=N eligible=M elapsed_s=S games_per_sec=R truncated=BOOL
```

`preprocess_fast.py` is not modified and continues to work against this binary.

---

## 6. Pipeline integration

`corpus_stream_parallel.sh` already had the `--scanner rust` wiring waiting. It
now prefers the one-pass extractor and falls back in two steps rather than one:

```
--scanner rust + clkscan + preprocess_onepass.py   -> one-pass extractor
--scanner rust + clkscan + preprocess_fast.py      -> Part C two-pass replay
otherwise                                          -> preprocess_lichess.py
```

The middle rung matters operationally: the HPC has its own deployed copies of
these scripts (`~/<workdir>/`), and a host that has synced `clkscan` and
`preprocess_fast.py` but not yet the new script keeps its existing speedup
instead of silently dropping all the way back to pure Python.

`SCANNER` still defaults to `"python"` in the code, and that is still true today.
The decision this change was waiting on was resolved on 2026-08-18 in the other
direction: once the HPC validation below passed, the live pipeline was stopped and
relaunched with `--scanner rust` passed explicitly, so rust became the production
choice operationally rather than by editing line 27. Do not read "the default" and
"the production path" as the same thing here.

---

## 7. Performance

Measured locally on macOS (Apple silicon), not on the HPC. Absolute rates are not
comparable to the HPC figures quoted at the top of this document; the *ratios*
and the memory numbers are the results worth carrying forward. The HPC
re-measurement of 2026-08-18 is in the validation section below and supersedes
these absolute numbers for operational planning.

**Full month, 2017-04, 11,348,506 games, `--max-games 30000`:**

| Path | Pass 1 | Pass 2 | Total | Peak RSS |
|---|---|---|---|---|
| `preprocess_fast.py` (Part C) | 66.4 s (170,979 g/s) | 494.6 s | **561.0 s** | **5.86 GiB** |
| `preprocess_onepass.py` (new) | 110.2 s (103,016 g/s) | 210.0 s | **320.2 s** | **203 MiB** |
| `clkscan --emit-pgn` alone | 109.1 s (104,023 g/s) | n/a | 109.1 s | 96.7 MiB |

**200,000-game sample, `--max-games 30000`, against the true production script:**

| Path | Total |
|---|---|
| `preprocess_lichess.py` | 378 s (529 g/s) |
| `preprocess_onepass.py` | 214 s |

Three things are worth reading out of this, and one of them was a surprise.

**1. Select mode costs about 40% of scan throughput.** `clkscan` scans at
~171,000 g/s in verdict-only mode and ~104,000 g/s while also buffering and
rendering. That is the price of keeping the games, and it is worth paying: it
buys the removal of a 494.6 s pass.

**2. Pass 2 is no longer a function of month size.** It went from 494.6 s to
210.0 s here, but the important part is not the 2.4x. `preprocess_fast.py`'s pass
2 scales with the *number of games in the month*; the new pass 2 scales with
`--max-games`. On 2017-04 (11.3M games) that is a 2.4x saving. On a real corpus
month like 2021-06 (92.2M games, 8x larger) the old pass 2 grows roughly 8x while
the new one does not move at all.

**3. Peak memory dropped 29x, from 5.86 GiB to 203 MiB.** This was not a design
goal and is the most operationally significant number here.
`preprocess_fast.py` accumulates all 30,000 parsed games in `parsed_by_slot`
before writing any of them, and each parsed game carries ~100 12x8x8 float32
tensors (~307 KiB), so the dictionary alone is ~9 GiB of tensors before Python
overhead. `preprocess_onepass.py` pickles each game the moment it is parsed and
never holds more than one. `preprocess_lichess.py` has the same problem in a
different shape: it holds 30,000 live `chess.pgn.Game` objects in the reservoir.

That matters because `corpus_stream_parallel.sh` sets `CONCURRENCY=5` on the HPC.
Five concurrent workers at 5.86 GiB is ~29 GiB of resident memory for
preprocessing alone; at 203 MiB it is ~1 GiB. Confirmed on the HPC 2026-08-18
(item 2 below), with one Linux-specific correction: the floor there is ~506 MiB
rather than 203 MiB, because importing `torch` costs ~500 MiB before a single game
is touched, so five concurrent workers on a full month came to ~3 GiB total rather
than ~1 GiB. Against ~10.3 GiB for the Python default path and ~29 GiB for the
two-pass version, the reason to prefer the new path survives the correction.

The local `parse_game` rate (~143 g/s) is well below the ~968 g/s the HPC
measured. That is a property of this laptop's torch build, not of the change:
`parse_game` is unmodified production code in both paths, and it is why the
absolute timings here need re-measuring on the HPC.

---

## 8. Rough edges and things I am not claiming

Written out rather than buried, in the same spirit as Part C's disclosure of the
header-key case-sensitivity narrowing.

- **No HPC run behind sections 3 and 7.** Everything measured in this document is
  local; the 21-minute / 72,717 g/s figures in the table at the top are Part C's
  HPC measurements, quoted, not reproduced. The separate HPC validation of
  2026-08-18 (section below) has since supplied real HPC numbers and revised two
  of the local ones.
- **The full-month leg uses `preprocess_fast.py` as its reference**, not
  `preprocess_lichess.py`. That is sound (Part C proved the two equivalent, and
  Leg A re-proves it here at 1,000 and 30,000) but it is a chain of
  two links rather than one. A direct full-month
  `preprocess_lichess.py` run would take hours locally. The HPC collapsed the
  chain on 2026-08-18, though bounded rather than full-month: a direct
  2,000,000-game comparison against `preprocess_lichess.py` on real 2021-06 data
  returned 30,000/30,000 content-identical (item 1 below).
- **Memory is now proportional to the sample rather than constant.** The
  reservoir holds up to `--max-games` rendered PGN strings: 96.7 MiB peak RSS for
  `clkscan` at 30,000 games, and the emitted PGN for a full month is 69.9 MiB.
  That is far below what it replaces (see section 7), but it is a *different*
  scaling law, and a much larger `--max-games` would eventually need a
  spill-to-disk design. There is no guard against that today; it would simply get
  slower and then fail on allocation.
- **The month tested is 2017-04, not a 2021-2024 corpus month.** It was chosen
  because it is a *complete* 3.2 GiB archive that could be downloaded and
  validated end to end on a slow link, where 2021-06 is 27.7 GiB. Lichess PGN
  structure is stable across this range and the eligibility rate observed
  (99.4%) matches what Part C saw, but this is an assumption, not a measurement,
  for the specific months the corpus run will process.
- **Variations are dropped from re-emitted games.** Correct for this pipeline,
  since both `has_full_clocks` and `parse_game` walk the mainline only. It does
  mean the emitted PGN is not a byte-faithful archival copy of the original game,
  and should not be treated as one if anybody later wants to reuse
  `--emit-pgn` output as a general-purpose corpus.
- **`month_label` is unchanged and still fragile.** It regexes `(\d{4}-\d{2})`
  out of the *whole path*, so a temp directory containing digits can produce a
  nonsense month label. It bit me during testing (`month=3863-57`). Pre-existing,
  not introduced here, and harmless in production where the path is the real
  dump filename, but worth knowing before somebody debugs it twice.

---

## What firstmate should HPC-validate

In priority order, before `--scanner rust` goes near the live corpus run:

1. **A full real corpus month, end to end, against `preprocess_lichess.py`
   directly.** Ideally 2021-06, since that is the month Part C and the
   independent re-validation both used, so the scan-phase numbers are directly
   comparable. Compare with `analysis/scripts/compare_processed_dirs.py`. This is
   the one test that collapses the two-link chain in Leg B into a single direct
   comparison, and it is cheap on the HPC.
   **Result: PASS.** Bounded 2M-game direct comparison against the real 2021-06
   archive: 30000/30000 games content-identical (headers, Moves, Clocks, all
   position tensors, filename sequence, `.sampling_summary.json` counts).
2. **Peak RSS on a real month at `--max-games 30000`.** Local runs used a 1,000
   and 30,000-game reservoir on a 200k-game sample plus a full month;
   confirm the memory figure holds on a 92M-game month with the production
   reservoir, and that it fits alongside 5-way concurrency
   (`corpus_stream_parallel.sh` sets `CONCURRENCY=5` for the HPC). Five
   concurrent workers each holding a 30,000-game reservoir is the real
   constraint, and it is the single most likely way this change causes an
   operational surprise.
   **Result: PASS.** ~506 MiB (driver) + ~84 MiB (clkscan) ≈ 0.6 GiB/worker on a
   full 92.19M-game month - comfortably fits 5-way concurrency.
3. **Wall-clock for one month**, to confirm the projected ~21 min and to replace
   the local timing numbers in section 7 with HPC-consistent ones.
   **Result: 32.3 min total** (pass 1 = 24.5 min at 62,800 games/s) - slower than
   the ~21 min projection but still well within budget.
4. **The `truncated` refusal against a genuinely partial download**, since that
   path now exits nonzero where the old scan mode exited zero, and
   `preprocess_month()` treats a nonzero exit as a retryable failure.
   **Result: PASS.** Verified end to end: exit 1 with a refusal message on a
   genuinely partial download; the retry plumbing picks it up correctly.
5. **`cargo test` on the HPC toolchain.** The CPython-equivalence vectors were
   generated against Python 3.13 locally; the HPC runs CPython 3.12 in
   `ratingnet2`. MT19937 and `_randbelow` have been stable since 3.2 so this
   should be a formality, but it is a cheap formality and it is the assumption
   the whole sample rests on.
   **Result: PASS.** 4/4 tests pass on the HPC toolchain; vectors byte-identical
   to the local 3.13-generated ones.

---

## Files

| File | What it is |
|---|---|
| `analysis/scanner-rs/src/main.rs` | CLI, visitor, PGN re-emission, both modes |
| `analysis/scanner-rs/src/pyrandom.rs` | byte-exact CPython MT19937 port |
| `analysis/scanner-rs/src/reservoir.rs` | Algorithm R, transcribed from the Python |
| `analysis/scanner-rs/tests/py_random.rs` | CPython-equivalence suite |
| `analysis/scanner-rs/tests/py_random_vectors.txt` | vectors generated by CPython |
| `analysis/scripts/gen_py_random_vectors.py` | regenerates those vectors |
| `analysis/scripts/preprocess_onepass.py` | the new driver |
| `analysis/scripts/compare_processed_dirs.py` | content-equivalence oracle |
| `corpus_stream_parallel.sh` | `--scanner rust` now prefers the one-pass path |

Build: `cd analysis/scanner-rs && cargo build --release`, then deploy
`target/release/clkscan` to `bin/clkscan` (the path `CLKSCAN_BIN` expects).

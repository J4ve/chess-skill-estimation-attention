# Corpus-pipeline bottleneck: verdict on the root-cause diagnosis

**Date:** 2026-08-17 (same day as the speed investigation)
**Scope:** Analysis only, per CLAUDE_TASK.md. No code was changed or committed to the pipeline.
**Question answered here:** does reservoir eligibility during the scan phase genuinely only need
clock/move-count data, or does anything in the current eligibility logic actually require the
built 12-plane tensor?

## Verdict

**The diagnosis is correct, and the fix is as clean as the report suggests.** The eligibility
decision made during the scan never reads the tensors that dominate its cost. The scan phase can
make the identical keep/skip decision for every game, in the identical order, consuming the
identical RNG stream, without building a single tensor. This is provable line-by-line from the
current code (details below), not just plausible.

One important nuance the report's framing undersells: **the fix alone does not make the corpus
schedule acceptable.** At the ~99M games/month these dumps actually contain (see "Evidence
base"), a 4-5x speedup turns the report's "1-4+ years" into roughly 10-13 months serial on the
HPC (~9-12 days/month). Hitting a thesis-compatible timeline additionally requires the report's
second recommendation: running months in parallel and/or on the faster local core. The fix plan
(`corpus-pipeline-fix-plan.md`) treats parallelization as part of the deliverable, not an
optional extra.

## Evidence base

- Read directly: `prototype/src/preprocess_lichess.py` (whole file, scan loop at lines 115-131,
  write loop at 140-151), `prototype/src/format_data.py` (`parse_game`, `board_to_array`),
  `corpus_stream.sh`, `AGENTS.md`.
- **Not available:** `data/local-preprocess-speed-test/report.md` is referenced by CLAUDE_TASK.md
  but is not in the repository (the `data/` tree is untracked) and is not in git history on any
  branch. The measured rates below (24 games/s HPC degraded, ~105 games/s local, ~8ms/game, 81%
  of scan time in tensor build, projected ~1.5-2ms post-fix) are taken from CLAUDE_TASK.md's
  quotation of that report and from the task-placement entry in AGENTS.md; they could not be
  independently re-checked. **Recommendation: commit the report (it is a .md; the `data/`
  gitignore rules only exclude .pgn/.csv/.jsonl) so reviewers of this analysis can audit the
  methodology.** The code-level verdict below does not depend on those numbers; only the
  magnitude of the win does.
- **Month volume anchor:** ~99-103M games/month, from the published Lichess counts
  (database.lichess.org/standard/counts.txt: 2021-04 = 99,184,138; 2024-01 = 98,994,760),
  corroborated in-repo by `corpus_stream.sh`'s EXPECTED archive sizes (~32.1 GB for 2021-04 ≈
  ~324 compressed bytes/game at 99.2M, a typical Lichess figure) and by CLAUDE_TASK.md's own
  local number (~105 games/s yet "~1 year single-core" implies ~95-100M/month). Note one
  internal inconsistency in the quoted report figures: "5% through month 1 in 21+ hours" at a
  constant 24 games/s would be only ~1.8M games ≈ 1.8% of the month; either the 5% is measured
  some other way (bytes, or an earlier faster rate before degradation) or one of the two numbers
  is off. The fix plan's schedule table uses the ~99M anchor and says to re-anchor from the
  relaunched scan's own counter within hours.

## What the scan phase actually decides, line by line

The scan loop (`preprocess_lichess.py:115-131`) applies two filters before a game can enter the
reservoir:

1. **`is_eligible(game)` (lines 62-76): headers only.** Variant == Standard, "Rated" in Event,
   WhiteElo/BlackElo are digits, TimeControl parses as `base+inc`. No move data, no board, no
   tensor.

2. **`parse_game(game, max_plies) is None` (line 121): this is the expensive one.** `parse_game`
   (`format_data.py:39-95`) walks the mainline up to `max_plies` (100) and per ply does three
   things: `board.push(move)`, `board_to_array(board)` (the 12x8x8 float32 tensor, appended to
   `Positions`), and a regex search for `[%clk ...]` in the node comment (appended to `Clocks`).
   It returns `None` in exactly two cases (lines 74-79):
   - `not clocks` (no clock annotation found on any traversed ply), or
   - `len(clocks) != len(positions)` (some traversed ply lacked a clock comment).

   Since `positions` gains exactly one entry per traversed ply and `clocks` gains at most one,
   the None/non-None decision reduces to a pure predicate:

   > **eligible ⇔ (at least 1 mainline ply) AND (every mainline ply up to ply 100 carries a
   > `[%clk ...]` comment).**

   Nothing in that predicate reads a board or a tensor. `Positions` contributes only its
   *length*, which equals the ply count. The Elo/result/time-control fields in the returned dict
   are header lookups that `is_eligible` already covers or that the decision ignores.

3. **The reservoir itself (lines 124-131) stores the `chess.pgn.Game` object, not the parse.**
   Tensors are (re)built at write time (line 143) for only the ~30,000 survivors. So the scan's
   tensor work is 100% discarded for every game that is not ultimately kept - which at ~30k kept
   out of tens of millions eligible-scanned per month is effectively all of them.

Supporting details that keep the fix clean:

- **`board.push()` is also skippable, not just `board_to_array()`.** The mainline of a game
  returned by `chess.pgn.read_game` contains only moves the parser already validated (illegal
  SAN ends the mainline and is recorded in `game.errors`), so the push never changes the
  eligibility outcome and never raises on these games. A clock-completeness walk needs only node
  traversal (`node.variations` / `node.variation(0)`) and the comment regex.
- **The RNG stream is untouched by the change.** `rng` (line 110) is consumed only inside the
  reservoir branch (line 129), once per eligible game after the reservoir fills. Identical
  eligible-game sequence in, identical `randint` draws out, identical reservoir contents,
  identical `game_{k:07d}.pkl` ordering and `.sampling_summary.json`. **Identical sampling
  output is therefore the expected result of a correct fix**: byte-identical
  `.sampling_summary.json`, identical kept-game identity and pickle ordering, and semantically
  identical pickle contents (`torch.equal` per Positions tensor; equal Clocks/Moves/headers).
  One caution for validators: raw `game_*.pkl` bytes are NOT comparable across runs even for
  unchanged code, because pickling a torch tensor goes through torch's legacy storage
  serializer, which embeds a runtime storage address into the byte stream. The validation bar
  in the fix plan is therefore summary byte-identity plus per-field semantic equality, not
  directory checksums.
- **The write phase is unchanged and cheap.** Re-parsing 30k kept games at ~8ms each is ~4
  minutes per month; irrelevant next to the scan.

## Where the fix is NOT as clean as it sounds (caveats, all manageable)

1. **The equivalence must be *exact*, and the failure mode is nasty if it is not.** The fast
   check must return True precisely when `parse_game(...) is not None`, including the
   `max_plies=100` truncation (a game missing clocks only *after* ply 100 is eligible) and the
   exact `\[%clk\s+([^\]]+)\]` pattern. If the fast path is ever *more* permissive, ineligible
   games reach the reservoir, the write-phase `parsed is None` guard (line 145) silently drops
   them, `kept` lands below 30,000, and `corpus_stream.sh:95` treats the month as failed -
   re-running a multi-day scan three times and then dying `FATAL`. If *less* permissive, the
   sample is silently biased. This is why the fix plan requires the A/B equivalence test plus a
   per-game assertion before the HPC relaunch, and adds a loud warning to the write-phase
   guard.
2. **The projected 4-5x is a report number, not re-measured here.** Amdahl caps the speedup at
   ~5.3x if the 81% attribution is exact; the residual cost is dominated by
   `chess.pgn.read_game` itself (SAN parsing against an internal board), which the fix cannot
   remove because the Game object is needed for the reservoir and the final parse. The fixture
   A/B run in the fix plan doubles as a cheap re-measurement; treat 4-5x as an estimate until
   then.
3. **The fix does not reduce memory.** The reservoir holds 30,000 full `chess.pgn.Game` objects
   (move trees plus comments) in RAM either way. Unchanged behavior, but worth knowing before
   anyone runs several months in parallel on one box.
4. **Fix-in-place alone still leaves ~9-12 days/month on the HPC, ~10-13 months serial for the
   34-month window** (throughput table in the fix plan, at the ~99M games/month anchor). The
   schedule is rescued by fix + parallelism + machine placement together; approving only the
   code change and calling it done would be a mistake.

## Bottom line

Eligibility during the scan needs exactly two facts per game: "does it have at least one
mainline ply?" and "does every mainline ply up to 100 carry a clock comment?" - both derivable
from the node walk alone. The tensor build is pure waste in the scan phase, the report's ~4-5x
projection is structurally credible, and a sampling-identical fix is straightforward. See
`corpus-pipeline-fix-plan.md` for the concrete diff, validation protocol, rollout sequence
around the in-flight HPC job, and the parallelization/placement recommendation.

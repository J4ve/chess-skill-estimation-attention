# Corpus-pipeline fix plan: skip tensor build during the reservoir scan

**Date:** 2026-08-17. Companion to `corpus-pipeline-bottleneck-verdict.md` (read that first for
the proof that this change is semantics-preserving).
**Status: PLAN ONLY. Nothing here is implemented or committed. Firstmate verifies against the
code before any implementation; the captain decides whether/when to touch the live HPC run.**

## Summary and recommendation

1. **Implement the scan-phase fast path below** (one new ~15-line function in
   `prototype/src/preprocess_lichess.py`, one 2-line call-site swap). Expected ~4-5x scan
   speedup (report's measured 81% tensor-build share; Amdahl cap ~5.3x).
2. **Validate sampling-identical output on a small fixture first** (protocol below; hours, not
   days). The bar: byte-identical `.sampling_summary.json`, identical kept-game filenames, and
   per-field semantic equality of every pickle pair. (Raw pkl bytes are NOT comparable across
   runs - see §3c - so directory checksums are deliberately not the bar.)
3. **Restart the HPC `corpus_stream` with the fix.** The in-flight month-1 scan (21+ h) is
   safely discardable: the pipeline checkpoints per month, the 30 GB download is preserved, and
   the lost position is regained in hours at the new rate (§4).
4. **Then add bounded month-parallelism (3-4 workers) on the HPC, plus optional local months
   when the captain is online** (matches the recorded local-when-online / HPC-when-not
   preference in AGENTS.md). Fix alone leaves ~9-12 days/month serial on the HPC (~10-13 months
   total); fix + 4 workers lands the full 34 months in roughly 2.5-3.5 months of wall-clock,
   and sustained local months can shave that further. Re-anchor these estimates from the
   relaunched scan's own `scanned=` counter within the first hours.

## 1. The exact change

New function in `prototype/src/preprocess_lichess.py` (module-level regex near `TC_BUCKETS`,
function after `is_eligible`), plus the call-site swap. `format_data.py` is deliberately left
untouched: it is shared with the inference API and the write phase still uses `parse_game`.

```diff
--- a/prototype/src/preprocess_lichess.py
+++ b/prototype/src/preprocess_lichess.py
@@
 TC_BUCKETS = ["ultrabullet", "bullet", "blitz", "rapid", "classical"]

+# Must stay byte-identical to the [%clk ...] pattern inside format_data.parse_game:
+# the scan-phase fast path below is only valid while both use the same match.
+CLOCK_COMMENT_RE = re.compile(r"\[%clk\s+([^\]]+)\]")
+
@@ def is_eligible(...) -> bool:  (unchanged, after it:)
+def has_full_clocks(game: chess.pgn.Game, max_plies: int) -> bool:
+    """Scan-phase eligibility fast path.
+
+    Contract: returns True exactly when parse_game(game, max_plies) would return
+    non-None, i.e. the game has at least one mainline ply and every mainline ply
+    up to max_plies carries a [%clk] comment. Walks the same mainline as
+    parse_game but skips board.push() and the 12-plane tensor build, which the
+    eligibility decision never reads.
+    """
+    node = game
+    ply_count = 0
+    while node.variations and ply_count < max_plies:
+        node = node.variation(0)
+        if not CLOCK_COMMENT_RE.search(node.comment):
+            return False  # missing or partial clock annotation
+        ply_count += 1
+    return ply_count > 0
+
@@ scan loop, lines 119-122
             if not is_eligible(game):
                 continue
-            if parse_game(game, max_plies=args.max_plies) is None:
-                continue  # no clock annotations
+            if not has_full_clocks(game, max_plies=args.max_plies):
+                continue  # missing or partial clock annotations
             eligible += 1
@@ write loop, lines 143-145
         parsed = parse_game(game, max_plies=args.max_plies)
         if parsed is None:
-            continue  # unreachable: eligibility already verified; guard anyway
+            # Unreachable while has_full_clocks matches parse_game exactly. A
+            # nonzero count here means the fast path diverged: the month's kept
+            # count will fall short and corpus_stream will retry/FATAL, so make
+            # the divergence loud in the log instead of silent.
+            print("WARNING: reservoir game dropped at write time "
+                  "(fast-path divergence?)", flush=True)
+            continue
```

Equivalence argument (early exit included): `parse_game` returns `None` iff `clocks` is empty
or `len(clocks) != len(positions)`; `positions` gains one entry per traversed ply (max 100),
`clocks` one per ply whose comment matches the regex. So non-None ⇔ ply_count ≥ 1 and every
traversed ply matched. `has_full_clocks` returns False on the first non-matching ply
(equivalent to the length check failing at the end) and otherwise returns `ply_count > 0`.
Identical traversal idiom (`node.variations` / `node.variation(0)`), identical regex, identical
`max_plies` cap. `board.push()` is dropped safely: `read_game` already validated the mainline,
so the push cannot change the outcome (verdict doc, "supporting details").

**Determinism consequence:** the RNG (`rng.randint`, one draw per eligible game once the
reservoir is full) sees the identical eligible-game sequence, so reservoir contents, kept-game
identity, pickle ordering, and `.sampling_summary.json` are all unchanged. `.sampling_summary.json`
is byte-reproducible; the pickles are *semantically* identical but not byte-identical across
runs, because pickling a torch tensor goes through torch's legacy storage serializer, which
embeds a runtime storage address (heap-address `storage_key`) into the byte stream. Validation
must therefore compare pickle *contents*, not checksums (§3c).

## 2. Expected throughput

Rates marked (R) are from the investigation report as quoted in CLAUDE_TASK.md/AGENTS.md (the
report file itself is not in the repo; see verdict doc). Days/month and totals are this plan's
derivations at **~99-103M games/month** (call it ~100M), anchored on the published Lichess
per-month counts (2021-04 = 99,184,138; 2024-01 = 98,994,760), corroborated in-repo by
`corpus_stream.sh`'s EXPECTED sizes (~324 compressed B/game at that volume) and by the report's
own local figure (~105 games/s yet "~1 year single-core" ⇒ ~95-100M/month). The report's
"5% of month 1 in 21 h" is inconsistent with its own 24 games/s (that is ~1.8M games ≈ ~1.8%);
if the ~36M/month that the 5% figure implies were somehow right, all times below shrink ~2.7x -
re-anchor from the relaunched scan's `scanned=` counter within the first hours.

| Scenario                          | games/s        | days/month | 34 months          |
|-----------------------------------|----------------|------------|--------------------|
| HPC today, degraded, serial       | ~24 (R)        | ~48        | ~4.5y (the "4+ years") |
| Local today, serial               | ~105 (R)       | ~11        | ~1y of on-time (the "~1 year") |
| HPC post-fix, serial              | ~95-120 (Amdahl cap ~125) | ~9.5-12 | ~325-415 days (~10-13 months) |
| HPC post-fix, 4 workers           | ~4x above      | -          | **~80-105 days (~2.5-3.5 months)** |
| Local post-fix, serial            | ~420-525       | ~2.2-2.8   | ~75-95 on-days     |
| HPC x4 + sustained local months   | -              | -          | **~2-3 months**    |

Scan phase only; the write phase (30k games x ~8ms ≈ 4 min/month) is unchanged and negligible.
The fixture A/B below doubles as a re-measurement of the actual speedup (`time` both runs), and
the first post-fix hours give the real games/s and games/month to replace this table.

## 3. Validation: sampling-identical output (before the fix touches the real corpus)

Run on the machine that will run production (HPC, conda env `ratingnet2`). The HPC tree at
`~/<workdir>` is a **file-synced deployed copy, not a git checkout** (AGENTS.md sync
rule), so the A/B works from two directories, not git checkouts: run A uses the untouched
deployed `src/` (old code); stage the reviewed branch's `src/` (fixed `preprocess_lichess.py`
plus its `format_data.py` sibling) in a scratch dir, e.g. `~/<workdir>/validation_new/src/`,
for run B. Production `src/` is not touched until rollout step 5, after (b)+(c) pass; the
in-flight job is killed at step 4 either way, so nothing re-execs mid-validation.

**(a) Build a small real fixture** from the already-downloaded month-1 dump, on a complete-game
boundary so both runs see identical bytes. Lichess dumps are long-window zstd, so the CLI needs
`--long=31` (this mirrors `max_window_size=2**31` in `stream_games`; plain `zstdcat` refuses
the frame):

```bash
zstdcat --long=31 data/raw_zst/lichess_db_standard_rated_2021-04.pgn.zst \
  | awk '/^\[Event /{n++} n>20000{exit} {print}' \
  | zstd -q -o /tmp/fixture_20k.pgn.zst
```

**(b) Per-game equivalence assertion** (strongest check, catches divergence even off the
sampled path). Save as a throwaway script inside `validation_new/src/` so Python's script-dir
sys.path entry resolves both imports:

```python
from preprocess_lichess import stream_games, is_eligible, has_full_clocks
from format_data import parse_game
mismatch = total = 0
with open("/tmp/fixture_20k.pgn.zst", "rb") as fh:
    for game in stream_games(fh):
        if not is_eligible(game):
            continue
        total += 1
        if has_full_clocks(game, 100) != (parse_game(game, max_plies=100) is not None):
            mismatch += 1
print(f"checked={total} mismatches={mismatch}")   # require mismatches == 0
```

**(c) End-to-end A/B**, same fixture, same seed, small reservoir:

```bash
python -u src/preprocess_lichess.py --input /tmp/fixture_20k.pgn.zst \
    --output-dir /tmp/ab_old --max-games 500 --seed 42          # run A: deployed old code
python -u validation_new/src/preprocess_lichess.py --input /tmp/fixture_20k.pgn.zst \
    --output-dir /tmp/ab_new --max-games 500 --seed 42          # run B: fixed code
diff /tmp/ab_old/.sampling_summary.json /tmp/ab_new/.sampling_summary.json  # require: empty
```

then compare pickle contents (NOT checksums - pkl bytes differ on every run even for unchanged
code, because pickled torch tensors embed a runtime storage address):

```python
import pickle, torch
from pathlib import Path
old = sorted(Path("/tmp/ab_old").glob("game_*.pkl"))
new = sorted(Path("/tmp/ab_new").glob("game_*.pkl"))
assert [p.name for p in old] == [p.name for p in new], "filename sets differ"
for po, pn in zip(old, new):
    a = pickle.load(open(po, "rb")); b = pickle.load(open(pn, "rb"))
    assert a.keys() == b.keys(), po.name
    for k in a:
        if k == "Positions":
            assert len(a[k]) == len(b[k]) and \
                all(torch.equal(x, y) for x, y in zip(a[k], b[k])), po.name
        else:
            assert a[k] == b[k], (po.name, k)
print(f"OK: {len(old)} pickle pairs semantically identical")
```

**Pass bar: (b) zero mismatches AND (c) empty summary diff, equal filename sets, and zero
assertion failures.** Any failure = stop; do not rationalize a diff.

**(d) Free in-situ check after relaunch:** the old run's `logs/preprocess_2021-04.log` contains
`scanned=X eligible=Y` lines every 5000 games. The stream is deterministic, so the new run's
`eligible` at the same `scanned` checkpoints must match the old log exactly. Any drift = stop
and investigate. **Copy that log aside before relaunching** - `preprocess_month` re-runs
`tee` (no `-a`) over the same path and would overwrite the 21 hours of evidence.

## 4. Safe rollout around the in-flight HPC job

Direct answers to the task's questions first:

- **Can the fix be swapped in without restarting from month 1?** The month is the pipeline's
  atomic checkpoint unit (`.corpus_done` marker; `preprocess_month` does `rm -rf "$outdir"` on
  every attempt). No month in the 2021-04..2024-01 window is complete, so the swap restarts
  month 1's *scan* - but that is the right call: the 30 GB month-1 download is preserved
  (size-verified skip in `download_month`), and the position the old run spent 21 h reaching
  (~1.8M games if the degraded 24 games/s held throughout; up to ~5M if the "5% through" figure
  is by game count) is regained in ~4-15 h at post-fix HPC speed, or ~1-3 h if month 1 moves
  local - hours either way, so discarding the partial scan is safe.
- **Does reservoir-sampled state need to be preserved?** No, and it cannot be: the reservoir
  lives only in process memory (no checkpoint exists), and a partial-month reservoir is not a
  valid uniform sample of the month anyway. Determinism also makes preservation pointless: same
  seed + same stream + unchanged eligibility semantics reproduce sampling-identically what the
  old code would eventually have produced.

Sequence (each step cheap and reversible until the relaunch):

1. Land the diff on a branch; firstmate reviews it against `format_data.parse_game` and the
   equivalence argument above.
2. Stage the fixed `src/` at `~/<workdir>/validation_new/src/` and run validation
   (a)-(c) there per §3. Do not proceed on any failure.
3. Pre-flight on the HPC (shared box, be a good citizen):
   - `cat logs/corpus_stream.progress` to confirm current phase (expected: mid-preprocess
     2021-04) and **copy `logs/preprocess_2021-04.log` aside** for check (d).
   - `df -h /home` - output is ~6 GB/month x 34 ≈ 200 GB onto a spinning-HDD `/home` already
     ~92% full per AGENTS.md. **If free space < ~250 GB this plan is blocked until space is
     negotiated; check before killing anything.**
   - Confirm the seed-rerun matrix is still 10/10 COMPLETE and no training procs (the
     `wait_for_training` gate will re-verify, but check first so the relaunch doesn't sit in
     the wait loop unexpectedly).
4. Kill the running job: `tmux kill-session -t corpus_stream`. Safe at any instant by design:
   downloads resume via `curl -C -` + size check; the partial `$OUT/2021-04` has no
   `.corpus_done` and is wiped on the next attempt.
5. Sync the fixed `preprocess_lichess.py` from the reviewed branch to
   `~/<workdir>/src/` (AGENTS.md rule: repo and HPC copies stay in sync).
6. Relaunch exactly as documented in `corpus_stream.sh`'s header. It will skip the completed
   download, pass the training gate, and rescan month 1 with the fix.
7. Watch check (d) for the first ~30 min, then the month-1 `.sampling_summary.json` and
   `kept=30000` at completion; note the realized games/s and re-anchor §2's schedule.

**Liveness risk to keep in view:** if the fast path were more permissive than `parse_game`,
write-time drops would push `kept` below 30000 and `corpus_stream` would burn 3 multi-day
attempts before dying FATAL (`corpus_stream.sh:95-104`). Validation (b)+(c) is the defense; the
new write-phase WARNING makes any residual divergence visible in `logs/preprocess_<m>.log`
immediately rather than days later.

## 5. Fix-in-place vs. also parallelizing; where to run

**Do both, sequenced: fix first (this week), then parallelism (next).** Fix-in-place alone
leaves ~10-13 months serial on the HPC - technically finite, practically still a thesis-killer.
But parallelizing the *current* slow code instead would waste 4-5x more CPU on a shared box and
still lose to the combination. The fix is also the low-risk half (sampling-identical,
validated); parallelism is pure orchestration and touches no sampling semantics, so staging
them keeps each step independently reviewable.

**Parallelism sketch (smallest change that works):** a parameterized worker variant of
`corpus_stream.sh`, then 3-4 tmux sessions each owning a disjoint month slice. The
parameterization must cover more than MONTHS - as written the script cannot run twice
concurrently:

- **`MONTHS`** (the worker's slice), **`LOG`/`PROG`** (per-worker suffix), **the tmux session
  name** (the documented launch hardcodes `-s corpus_stream`; a second worker fails with
  "duplicate session"), and **`RAW`** (per-worker staging dir).
- **Remove the hardcoded Phase A `download_month 2021-04` (`corpus_stream.sh:135`)** - it runs
  unconditionally in every launch, so every worker would re-download month 1's ~32 GB, and
  concurrent `curl -C -` resumes against the same shared path can corrupt the file and burn the
  6-attempt retry loop. Phase C already downloads each month just-in-time; Phase A only existed
  to overlap the first download with training and is obsolete once the training gate passes.
- **Gate the Phase C loop with the done marker before downloading**:
  `[ -f "$OUT/$m/.corpus_done" ] && { log "$m done elsewhere; skipping"; continue; }` ahead of
  `download_month "$m"` - otherwise a month completed by another worker still costs a ~30 GB
  download that is then thrown away.
- **Stage `RAW` on the NVMe `/tmp`** (wiped on reboot - fine: raw archives are re-downloadable
  with resume, and `.corpus_done` lives with the output on `/home`). This protects the
  ~92%-full spinning `/home` from concurrent 30 GB streams and the seek-thrash that would erode
  the parallel win. 30 GB x workers of NVMe needed.
- **Politeness: `nice`/`ionice` the workers and cap at 3-4 cores**; this is a multi-user box
  and the report's "degraded to 24 games/s under load" cuts both ways.
- **RAM: workers x (reservoir of 30k Game objects + decompress buffers)** - measure month-1's
  peak RSS post-fix before choosing the worker count.
- The parameterized `corpus_stream.sh` is itself a deployed copy; AGENTS.md's sync rule covers
  it too - deploy from the reviewed branch, keep repo and HPC in sync.

**Placement recommendation: HPC as the always-on backbone, local as an opportunistic
accelerator - not local-primary.** Rationale against local-primary despite its 4-5x per-core
edge: (1) the local machine is only available while the captain's PC is on (recorded
captain preference/constraint in AGENTS.md), so unattended completion must never depend on it;
(2) HPC-x4 post-fix already lands in ~2.5-3.5 months with zero coordination overhead; (3) the
pkl output must end up on the HPC for training - each ~6 GB month uploads in ~13 h at 1 Mbps /
~3 h at 5 Mbps sustained and can overlap the next month's scan, so upload only throttles a
local worker below roughly 0.5-1 Mbps sustained up-bandwidth; measure the captain's real
sustained uplink before committing many local months, but availability, not bandwidth, is the
decisive constraint.

Local-worker mechanics (when used):

- **Static ownership, not ad-hoc tail-eating:** give the local worker an explicitly reserved
  slice (e.g. 2023-07..2024-01) excluded from every HPC worker's MONTHS, with a pre-agreed rule
  for handing months back to an HPC worker if the captain stays offline. "Both ends work the
  same month" must be impossible by construction: `.corpus_done` is checked only at
  `preprocess_month` entry, so a marker that arrives mid-attempt is destroyed by the next
  attempt's `rm -rf`.
- **Download with resume (curl -C - plus the EXPECTED-size check, as `download_month` does)
  rather than `--url` streaming.** The `--url` path is a bare `curl -s` with no retry/resume; a
  dropped connection can surface as a clean EOF, the reservoir is already full, `kept` still
  hits 30000, and the month is silently biased toward its early days with no failing signal.
  If `--url` is used anyway, require the month's `scanned` count in `.sampling_summary.json` to
  land near the ~99-103M expected volume before accepting the month.
- **The local run must create `.corpus_done` itself:** only `corpus_stream.sh`'s
  `preprocess_month` writes the marker (after its kept-count check), and that wrapper cannot
  run locally as-is (hardcoded conda activation, HPC paths). After the local run, verify
  `game_*.pkl` count == 30000 and the SUMMARY line shows kept=30000, then touch
  `.corpus_done` - only then transfer.
- **Transfer atomically:** rsync transfers files in sorted order, so `.corpus_done` would land
  *before* the 30k pickles and a partially-synced month could look complete to an HPC worker.
  Rsync into `data/processed_games/<month>.incoming/` and `mv` into place (or
  `--exclude=.corpus_done` and copy the marker last).
- Sync month directories as directories; never flatten without the `{month}_` prefix
  (AGENTS.md: per-month pkl filenames repeat, a bare copy silently overwrites down to ~30k
  files).

## 6. Explicitly out of scope now (noted for later)

- **Skip movetext for header-ineligible games** (python-chess visitor `SKIP` from
  `end_headers`, or `read_headers`-based skimming): real further headroom on top of 4-5x, but
  it changes how the stream is consumed and needs its own equivalence validation. Not worth
  coupling to this change while a live run waits.
- **Shrinking the pkl format** (positions are one-hot float32 - hugely compressible, or
  rebuildable from `Moves` at load time): would cut the ~200 GB footprint and any local-upload
  tax by an order of magnitude, but touches `ChessGamesDataset` and every existing processed
  month. Separate proposal, after the corpus lands.
- **Reservoir checkpointing** for mid-month resume: heavy (pickling 30k Game objects) and
  unnecessary once months take ~2-12 days each.

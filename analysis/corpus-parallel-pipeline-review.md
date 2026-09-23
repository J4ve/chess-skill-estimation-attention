# Review: `hpc_corpus_stream_parallel.sh` (parallel corpus download pipeline)

Review pass only, nothing was fixed. Requested in `CLAUDE_TASK.md` Part A.

**Revision note.** Every load-bearing claim in the first draft was re-checked by two
independent adversarial verifiers. Four claims were narrowed and one was withdrawn
entirely. The withdrawn claim and the narrowed-away parts are recorded in section 6,
"Considered and ruled out", rather than deleted silently, so nobody re-derives them.

**Where the reviewed code lives.** The script is not a file in this repo. It exists only
embedded inside `CLAUDE_TASK.md`, lines 86 to 195. All citations below are
`CLAUDE_TASK.md:<line>`, with the script's own internal line number in parentheses where
it helps. The mapping is `script line N = CLAUDE_TASK.md line N + 85`.

**Comparison baseline.** `corpus_stream.sh` (repo root) is the strictly serial predecessor.
Several findings are dropped safety properties, meaning the serial version had a guard and
the parallel rewrite does not.

**Verification note.** Several behaviours below were confirmed by running them, not by
reading docs from memory: curl's handling of 404 and 416 under `-C -`, `xargs -P` exit
propagation, `$0` resolution after an internal `cd`, `zstandard`'s handling of a truncated
archive, and in-place editing of a running script. Each is marked VERIFIED with the observed
result. Anything I could not run is marked UNCERTAIN with the exact check needed. I have no
HPC access, so no claim about live disk state, live process state, or the contents of
`local_corpus_stream_parallel.sh` is made. One further reproduction (bash 3.2's handling of
`declare -A`) was run on a Mac that is neither of this project's two machines, and the
finding drawn from it has been withdrawn: see section 6.

---

## 1. Claim-lock correctness

**Verdict: does not hold as a locking protocol.** The `mkdir` itself is atomic and no two
workers ever win the same `mkdir`. The protocol built around it is unsound in three ways.

`try_claim` is `mkdir "$CLAIM_DIR/$1" 2>/dev/null` (CLAUDE_TASK.md:140, script 55). That
single call is genuinely atomic on POSIX, so the narrow question ("can two workers both
receive success from `mkdir`?") answers **no**. The problems are around it.

**(a) The claim is never released on success.** The success path is
`touch "$outdir/.corpus_done"; rm -f "$zst"` (CLAUDE_TASK.md:179, script 94) followed by a
log line and an implicit return. `release_claim` appears only on the two failure paths
(CLAUDE_TASK.md:168 and :177). So a claim directory means either "a worker is holding this
right now" or "this finished successfully, possibly weeks ago", and nothing on disk
distinguishes them. This is not merely untidy: it removes the only signal a monitor or a
staleness sweeper could use. Confirmed by grep over the whole script: no `trap`, no `flock`,
no `$$`, no hostname and no timestamp written anywhere into the claim.

**(b) Nothing marks a month as in flight, so clearing a claim by hand can start a second
worker on a live month.** The `.corpus_done` marker is written only at CLAUDE_TASK.md:179,
at the very end of a successful month, and the claim directory carries no metadata (see
(a)). Between those two facts there is no on-disk state that says "a worker is here now".
The only recovery this design offers an operator is to remove a claim directory by hand,
and an operator who does that on a month whose worker is actually still alive starts a
second worker on it. Both workers then run `rm -rf "$outdir"; mkdir -p "$outdir"`
(CLAUDE_TASK.md:171, script 86) and write into the same directory using the same
`game_%07d.pkl` names (`prototype/src/preprocess_lichess.py:177`), with no per-worker temp
directory and no atomic rename.

Two corrections to how this was originally written up. First, the ordering of the
`.corpus_done` check at :147 and the claim at :148 is **not** the cause and reordering them
would fix nothing: `mkdir` is atomic, so two workers that both pass the :147 test still
contend at :148 and exactly one wins. Checking the marker first is also strictly cheaper,
since a finished month with an unreleased claim is skipped without touching the claim
directory. Second, the damage is not a mixed sample. The invocation at CLAUDE_TASK.md:172-173
never passes `--seed`, so both runs use the default 42 (`preprocess_lichess.py:119`, and
`rng = random.Random(args.seed)` at :133) over the same input file, and the two reservoirs
are identical in content. The real damage is one worker's `rm -rf` landing under the other
worker's live writes, and `.corpus_done` possibly being touched over a half-populated
directory.

**(c) On the local machine, a failed SSH is indistinguishable from a lost claim.** The task
states the local copy performs `try_claim` and `release_claim` over SSH. Both helpers
discard stderr and are consumed purely as a boolean:
`try_claim "$m" || { log "$m: claim lost, skipping"; return 0; }` (CLAUDE_TASK.md:148,
script 63). `ssh` returns 255 for any transport failure, which this code reads as "someone
else owns the month" and quietly skips. Two bad outcomes follow, and neither is logged
distinguishably:

- SSH fails before the remote `mkdir` runs: the month is skipped although nobody owns it.
  A rerun recovers it, so this one is survivable.
- The remote `mkdir` succeeds but the connection drops before the exit status returns: the
  claim now exists on the HPC forever, and the local worker believes it lost the race. That
  month is in the local list only, so the HPC will never pick it up. It is stranded
  permanently, and a rerun does not help because the claim is still there.

The same swallow-everything pattern applies to `release_claim` (CLAUDE_TASK.md:141), so a
failed SSH during release also silently leaks a claim. A verified supporting detail: `mkdir`
on a path whose parent does not exist fails exactly like `mkdir` on an existing directory
(VERIFIED), so a wrong remote working directory also reads as "claim lost".

**(d) Minor, and only on a tree the parent has never run in.** The worker branch
(CLAUDE_TASK.md:186-189) returns before `mkdir -p logs "$RAW_DIR" "$OUT_DIR" "$CLAIM_DIR"`
at CLAUDE_TASK.md:191, and the worker's own `mkdir -p` at :151 covers only `RAW_DIR` and
`outdir`, and only after the claim. On a fresh checkout where `data/.claims` does not exist,
`bash hpc_corpus_stream_parallel.sh --worker 2021-04` therefore fails its claim (stderr
discarded at :140) and reports the misleading "claim lost, skipping". On the deployed tree
the parent has already run and those directories exist, so hand-running a single month this
way works today and is a usable recovery invocation. Hoisting the `mkdir -p` above the
branch is one line of hardening, not a live defect.

---

## 2. Failure and retry correctness

**Verdict: real bugs found.** The two *checked* failure paths do release the claim. Almost
everything else about failure handling is broken or was dropped from the serial version.

**Every explicitly handled failure does release.** Download failure at CLAUDE_TASK.md:168
and short kept-count at CLAUDE_TASK.md:177 both call `release_claim` before returning 1.
That part is correct.

**A killed worker never releases.** There is no `trap` anywhere in the script (VERIFIED by
grep across CLAUDE_TASK.md:86-195: zero matches for `trap`). SIGKILL, the OOM killer, a node
reboot, or `tmux kill-session` therefore leaves the claim directory in place with no
metadata, and nothing in the script ever reclaims it. That month is stranded until a human
notices and runs `rmdir` by hand. This is the exact scenario Part D was written to fix, and
it is worth noting the OOM path is not hypothetical: `preprocess_lichess.py` holds up to
30000 live `chess.pgn.Game` objects in the reservoir simultaneously
(`prototype/src/preprocess_lichess.py:134` and :149-154), and `CONCURRENCY=5` multiplies that
by five. I did not measure the per-process resident size, so the absolute number is
UNCERTAIN, but the multiplication by five is certain and unguarded.

**There is no preprocess retry at all.** The serial version retried preprocessing three
times before giving up (`corpus_stream.sh:89-101`), and only then exited FATAL so a human
would see it. The parallel version runs the preprocess exactly once
(CLAUDE_TASK.md:172-173), and on a short kept-count it releases the claim and returns
(CLAUDE_TASK.md:177). Because `xargs` is fed each month exactly once from
`printf '%s\n' $MONTHS` (CLAUDE_TASK.md:193), nothing revisits that month for the remainder
of the run. One transient failure, for example a network blip or a momentarily full disk,
silently costs a month for the whole multi-week run. This is a dropped safety property.

**The preprocess exit status is never examined.** CLAUDE_TASK.md:172-173 redirects the
python process to a log and discards `$?`. The only gate is the kept file count. A hard
python crash (the missing-conda-activation class of bug that was already hit once) is
therefore reported as `FAILED - only 0/30000 kept`, which points the operator at sampling
rather than at the interpreter.

**A month that fails preprocessing leaks its full ~30 GB raw archive.** `rm -f "$zst"`
appears twice in the script. The one at CLAUDE_TASK.md:160 deletes the archive on every
size-mismatch attempt inside the download loop, and the one at CLAUDE_TASK.md:179 deletes it
on the success path. There is no third. So the kept-count failure return at
CLAUDE_TASK.md:177 leaves the complete archive, roughly 30 GB, sitting in `data/raw_zst`,
and because that month's claim was released while nothing revisits it in this run (F4) and
`xargs` never re-feeds it (CLAUDE_TASK.md:193), nothing ever reclaims the space. In the
serial design a failure exited the whole script (`corpus_stream.sh:80` and :104) and the raw
file was deleted inside the loop (`corpus_stream.sh:144`), so leakage could not accumulate.
Here five workers keep going on a filesystem CLAUDE.md:27 puts at about 92% full.

The download-failure return at CLAUDE_TASK.md:168 does **not** leak, and the first draft was
wrong to say it did. `:167` is
`if [ -n "$exp" ] && [ "$(file_size "$zst")" != "$exp" ]`, so :168 is reachable only when
`exp` is set, and with `exp` set the only way out of the loop without a `break` is the
mismatch branch at :159-160, which has just run `rm -f "$zst"` on attempt 6. Reaching :168
implies the archive was deleted moments earlier.

**The run reports success no matter how many months failed.** VERIFIED: a worker exiting 1
does not stop `xargs -P`, and although `xargs` then exits nonzero, its status is never
checked, and the final `log` at CLAUDE_TASK.md:194 becomes the script's last command and
resets the exit status to 0. My reproduction printed `PARENT SCRIPT EXIT = 0` with a worker
that deliberately exited 1. The closing message, "all months in this host's list processed
or claimed elsewhere", is printed unconditionally and is false whenever anything failed.
There is no summary of how many months succeeded.

---

## 3. The `bash "$0" --worker <month>` re-invocation

**Verdict: works in the intended case, but has two real failure modes.**

The core mechanism is sound. Each worker is a fresh `bash` reading the script file from the
top, so `declare -A EXPECTED_SIZES` (CLAUDE_TASK.md:125-135) is re-declared naturally in
every worker process, which is the stated goal. `set -u` combined with
`exp="${EXPECTED_SIZES[$m]:-}"` (CLAUDE_TASK.md:154) is the correct way to read a possibly
absent key without tripping `-u`, and `local kept; kept=$(...)` at CLAUDE_TASK.md:175 is the
correct two-step form that avoids masking the command's exit status. Those details were done
right.

**Failure mode A: `$0` is resolved after the script has already changed directory.**
CLAUDE_TASK.md:106 (script 21) runs `cd ~/<workdir>` and CLAUDE_TASK.md:193 then uses
`bash "$0"`. If the script is launched by a relative path from any directory other than its
own, `$0` stays relative and is re-resolved against the new working directory. VERIFIED: I
reproduced this with a script launched as `bash proj/s2.sh` that cd's into `proj`. Every
worker died with `No such file or directory`, and the parent still printed its final line and
exited 0. Applied here that means all 35 months are skipped in under a second while the log
says the run completed. The current launch style is not visible from the repo, so whether
this fires today is UNCERTAIN. To check: look at how the tmux session was started, or simply
make the re-invocation path absolute.

**Failure mode B: editing the deployed script while it is running.** This is not theoretical
for this pipeline. The parent parks inside `xargs` at CLAUDE_TASK.md:193 for two to three
weeks, two bugs were already fixed on day one, and CLAUDE.md:26 instructs agents to keep the
repo and HPC copies in sync. Two distinct effects:

- New workers spawned after an edit run the new code while already-running workers run the
  old code, so a single run can mix versions with no record of which month got which.
- VERIFIED, and worse than the version skew: bash reads a script incrementally by byte
  offset. I rewrote a running script in place with `cat >` while its parent was parked in a
  `sleep`, and the parent resumed at a stale offset, executed a fragment of a comment line as
  a command (it invoked `top` with garbage arguments), then ran the edited script's body.
  Control test: editing via atomic rename (`mv new old`, which is what `vi` and `sed -i` do)
  was completely safe, because the running parent keeps the original inode. So the rule is
  narrow and worth writing down: never edit the deployed script with `cat >`, `>`, `>>` or
  anything else that truncates in place while a run is live.

**Dead code worth removing for clarity.** `export -f log file_size try_claim release_claim
process_one_month` and the `export` of the scalars (CLAUDE_TASK.md:183-184) have no effect
under this design, because each worker re-reads every definition from the file. They imply
state flows to workers through the environment, which is not how this works, and they also
push `BASH_FUNC_*` entries into the environment of every `curl` and `python` child.

---

## 4. Resource behaviour

**Verdict: real bug found, disk is the sharp edge.**

**Peak raw staging is about five times what the serial design documented, on a nearly full
shared disk.** `corpus_stream.sh:15-16` documents "Peak raw staging ~30 GB (one archive)".
With `CONCURRENCY=5` there are up to five archives staged at once. Computed from the
script's own `EXPECTED_SIZES` table (CLAUDE_TASK.md:125-135): mean month 30.6 GB, so a
typical five-way peak is about **153 GB**, and the five largest months together are
**168.2 GB**. Add the roughly 6 GB per month of output that accumulates
(`corpus_stream.sh:16`) across 35 months, about 210 GB, which is unchanged from the serial
design. CLAUDE.md:27 states the shared `/home` is a slow spinning HDD at about 92% full, and
`~/<workdir>` sits on it. The script contains **no `df` check, no `ulimit`, no free
space precondition anywhere** (VERIFIED by grep: zero matches for `df`, `ulimit`, `nice`).
I cannot see live free space without HPC access, so whether 153 GB actually fits is
UNCERTAIN and should be checked before anything else in this review.

The failure mode if it does not fit is nasty and self-amplifying: a full disk truncates
curl's output, the truncated file fails the size check, CLAUDE_TASK.md:160 deletes it, and
the worker re-downloads roughly 30 GB from scratch, up to six times, across five workers at
once. That is sustained network and disk load on a shared box for hours, ending with every
month marked FAILED, and it also affects the 38 other users on that machine.

**Concurrency is not exceeded.** Each worker is internally sequential, download then
preprocess, so at most five downloads or five preprocesses are live, never ten. That part
matches the intent.

**Orphaned processes: yes, if the parent is killed by anything other than a terminal
interrupt.** There is no `trap` and no process-group cleanup. A Ctrl-C in the foreground
sends SIGINT to the whole process group and does clean up. But `kill <parent_pid>`, the
usual way to stop a background or nohup'd run, terminates only the parent and leaves `xargs`,
five worker shells, and their `curl` and `python` children running and reparented, with all
five claim directories still held. Stopping this pipeline cleanly currently requires killing
the process group by hand and then removing claim directories by hand.

**Minor:** `sleep 60` at CLAUDE_TASK.md:165 also executes after the sixth and final attempt,
and every retry sleep is spent holding the claim while doing nothing, up to five idle minutes
per failing month.

---

## 5. General review

**Verdict: real bugs found, including one with irreversible data consequences.**

### 5.1 The headline finding: a short download can be silently accepted, marked done, and its raw archive deleted

The script's own header comment (CLAUDE_TASK.md:100-103) states the design rationale:

> EXPECTED_SIZES: ... Missing/wrong entries just skip the size check (a warning, not a
> failure) rather than blocking, preprocess_lichess.py's own kept-count check is the real
> correctness gate.

**That rationale is empirically false, and I verified it end to end against the real
production script.** Two facts combine:

1. VERIFIED: `zstandard`'s `ZstdDecompressor.stream_reader`, as used at
   `prototype/src/preprocess_lichess.py:56-57`, **does not raise on a truncated archive**. It
   returns the intact prefix and then reports clean EOF. I tested truncation at 50%, 90% and
   99.9% of a single frame, and at a multi-frame boundary. Every case completed with no
   exception.
2. VERIFIED: I ran the real `prototype/src/preprocess_lichess.py` (with a numpy shim for
   `torch.zeros`, the only torch call `format_data.parse_game` makes) against an intact
   synthetic month and against the same month truncated to 45% of its bytes. The truncated
   run **exited 0, printed no warning, and produced a completely full reservoir**:

   ```
   intact    : scanned=12000  eligible=12000  kept=3000
   truncated : scanned=4410   eligible=4409   kept=3000     <- gate passes
   ```

So the kept-count gate at CLAUDE_TASK.md:176 passes on a truncated archive, and the script
proceeds to CLAUDE_TASK.md:179, which touches `.corpus_done` and then deletes the raw `.zst`.
The month is permanently recorded as complete while its 30000 games were drawn only from the
downloaded prefix. Because Lichess dumps are ordered chronologically, that is precisely the
"first thin time slice of the month, starving the slow time-control buckets" bias that the
reservoir sampling was introduced to eliminate, documented at
`prototype/src/preprocess_lichess.py:11-16`. The raw archive is gone, no game ID is stored,
and CLAUDE_TASK.md:355-360 explains that recovery requires a full re-download plus a full
re-scan.

**Scope, stated carefully.** This path is reachable only when `exp` is empty, that is when
the month has no `EXPECTED_SIZES` entry, because otherwise the size comparisons at
CLAUDE_TASK.md:159 and :167 catch the short file. I checked the embedded HPC copy
programmatically: all 35 months in `MONTHS` have an entry, and every entry shared with
`corpus_stream.sh` is byte-identical. **So the HPC copy as embedded is not exposed today.**
The exposure is:

- **`local_corpus_stream_parallel.sh`, which I cannot read.** Its months are "2024-09 onward"
  (CLAUDE_TASK.md:118-119) and none of those are in the HPC table. If its own table was not
  extended, every month it processes runs down the unchecked branch. This is the single
  highest-value thing to check in this whole review, and it should be checked before that
  script processes another month.
- **Any future edit to `MONTHS` that forgets the table.** That edit already happened once
  this session: 2024-08 was added (CLAUDE_TASK.md:123) and it is absent from
  `corpus_stream.sh`. It was remembered in the table this time. The serial version made this
  impossible by asserting at startup that every month has an expected size
  (`corpus_stream.sh:56`, FATAL). That assertion was dropped.

Also relevant: the `.sampling_summary.json` written at
`prototype/src/preprocess_lichess.py:194` records `scanned`, which is exactly the number that
would expose a truncated archive (4410 versus 12000 above). The script never reads it.

### 5.2 `curl` lost its `-f`, so HTTP error bodies are written to the archive file

Serial: `curl -fsSL -C - --retry 3 --retry-delay 30` (`corpus_stream.sh:70`).
Parallel: `curl -sL -C - -o "$zst" "$url"` (CLAUDE_TASK.md:156). VERIFIED against a local
server: without `-f`, a 404 response writes the 54-byte HTML error body into the output file
and **curl exits 0**. Combined with 5.1's unchecked branch, `[ "$sz" -gt 0 ]` at
CLAUDE_TASK.md:162 then logs "trusting non-empty download" and breaks out of the retry loop.
With an expected size present the size check catches it, but only after six attempts and five
wasted minutes of sleeps. `--retry`/`--retry-delay` were also dropped, so transient 5xx
responses now consume a whole attempt each.

I also tested the case I initially suspected, `curl -C -` against an already-complete file,
and it is **fine**: curl 8.7.1 recognises the 416 and leaves the file untouched with exit 0.
The serial version's explicit `rc=33` handling (`corpus_stream.sh:71`) is not needed on
modern curl. No finding there.

### 5.3 `-C -` resume is dead code, so every retry re-downloads about 30 GB from zero

CLAUDE_TASK.md:160 executes `rm -f "$zst"` on every size mismatch, and the next loop
iteration then calls `curl -C -` on a file that no longer exists. The resume flag can
therefore never do anything within a run. The serial version deliberately kept the partial
file and let `-C -` continue it, deleting only when the file was *oversized*
(`corpus_stream.sh:65-77`). This is a dropped safety property with a direct cost: a month
that repeatedly loses its connection near the end now restarts from byte 0 every time, up to
six times, roughly 180 GB of transfer, and can realistically never finish, whereas the serial
version would have inched forward.

### 5.4 Both startup validations were dropped

`corpus_stream.sh:53-57` validated, before doing any work, that every month matched an
approved-window regex and that every month had an expected size, exiting FATAL otherwise.
Neither check exists in the parallel version. A typo such as `2024-13` or `2021-4` now costs
a claim, six download attempts and a misleading `FAILED - only 0/30000 kept` before anyone
sees it, and per 5.1 it can be worse than that. Re-adding a preflight loop is cheap and would
have caught the entire 5.1 exposure class at startup.

### 5.5 `cd` lost its guard

Serial: `cd ~/<workdir> || exit 1` (`corpus_stream.sh:22`).
Parallel: `cd ~/<workdir>` (CLAUDE_TASK.md:106), no guard, and `set -e` is not in
effect (CLAUDE_TASK.md:105 is `set -uo pipefail`). If that `cd` ever fails, for example an
unset `HOME` under a stripped cron or systemd environment, every subsequent path is relative
and the worker happily creates `data/raw_zst` wherever it happens to be and downloads 30 GB
into it.

### 5.6 Observability regressions over a multi-week unattended run

- `log()` no longer tees to a persistent file. Serial: `... | tee -a "$LOG"`
  (`corpus_stream.sh:32`). Parallel: bare `echo` (CLAUDE_TASK.md:137). Unless the launch
  command redirects, the entire run record lives only in tmux scrollback and dies with the
  session or the box.
- The machine-readable progress file is gone entirely (`corpus_stream.sh:27` and :33). That
  is exactly what a `--status` command would have read, which is relevant to Part D.
- Five workers write interleaved lines to one stream with no per-month log file for the shell
  half of the work. The python half does go to `logs/preprocess_${m}.log`
  (CLAUDE_TASK.md:173), and `--log-every 200000` at the ~500 games/s in CLAUDE.md:32 is a
  line about every 6 to 7 minutes, which is a usable liveness signal (the serial version's
  `--log-every 5000` at `corpus_stream.sh:93` was much finer). The gap is not that a hung
  worker is indistinguishable from a slow one, since that log file's mtime does distinguish
  them. The gap is that nothing in the script or in any tooling looks at it.

### 5.7 Smaller items

- **String versus numeric size comparison.** CLAUDE_TASK.md:159 uses `=` where
  `corpus_stream.sh:67` used `-eq`. It happens to work because both operands are bare
  integers, but it would silently mis-compare on any whitespace or leading zero.
- **The kept-count gate demands exactly `>= 30000`.** A month with fewer than 30000 eligible
  games would fail permanently every run with no path to acceptance. Unlikely for these
  months given their size, so this is a note rather than a defect.

---

## 6. Considered and ruled out

These were in the first draft and did not survive verification. They are kept here with the
reason so they are not re-derived.

**bash 3.2 and BSD `stat` portability (was F8, withdrawn as a live defect).** The first draft
claimed that on bash 3.2 the `EXPECTED_SIZES` literal is *silently* reinterpreted as
arithmetic subscripts, producing a scrambled table in which months collide and some resolve
to another month's size. That is wrong on two counts.

- The failure is loud, not silent. Running the verbatim literal from CLAUDE_TASK.md:125-135
  under bash 3.2 aborts at the first invalid-octal subscript with
  `value too great for base (error token is "08")`, at the `2021-08` element. The array is
  left holding four entries at arithmetic indices 2014 to 2017, and the lookups that do
  resolve return their own correct sizes, because the assignment aborts long before any
  colliding month is ever assigned. The real bash 3.2 behaviour, for the record, is that
  months from 2021-09 onward get an empty `exp` and fall into the 5.1 unchecked branch, and
  any month containing `-08` or `-09` makes the lookup at :154 raise an arithmetic error that
  abandons `process_one_month` with the claim already taken at :148 and no log line.
- It does not bite either deployment target. The HPC is Ubuntu 22.04
  (`hpc-operations-guide.md:15`), so bash 5 and GNU coreutils. The captain's local machine is
  Windows with WSL, not BSD: `hpc-operations-guide.md:32` refers to "a Windows/WSL machine",
  and CLAUDE.md:44-45 describes winget-installed draw.io, `C:\`-style path handling and
  `HOME=C:\Users\<user>`. WSL and Git Bash both give bash 4 or newer and GNU `stat`. The
  reproduction that produced the original claim was run on a reviewer's Mac, which is neither
  of this project's machines.

What survives is a portability note, not a finding: `declare -A` (CLAUDE_TASK.md:125) needs
bash 4 or newer, and `file_size() { stat -c%s ...; }` (CLAUDE_TASK.md:138) is GNU-only, so on
a BSD userland `stat -c%s` is rejected (`illegal option -- c`, confirmed) and the `|| echo 0`
fallback makes every size check fail. That matters only if someone runs this script on macOS,
including an agent testing it locally. It is not a defect on the machines this pipeline
actually runs on.

**The `.corpus_done`-before-claim ordering (was the stated cause of F11).** Checking the
marker at :147 before claiming at :148 opens no window: `mkdir` is atomic, both orderings
behave identically, and checking the marker first is cheaper. The hazard is real but its
cause is the missing in-flight state described in 1(b), not the ordering. Reordering the two
lines would fix nothing.

**The narrow kill window between `touch .corpus_done` and `rm -f "$zst"` (was F17's framing).**
CLAUDE_TASK.md:179 is `touch "$outdir/.corpus_done"; rm -f "$zst"` on one line and in that
order, so the window exists, but it is two consecutive fork/exec calls wide, with 35
opportunities in the whole run. It is not the operative risk. Because the archive is deleted
only on the success path, *any* interruption during the hours-long preprocess of a month
leaves the same ~30 GB orphan, and what stops a later run from reclaiming it is the stranded
claim (F3), not the `.corpus_done` short-circuit at :147. F17 below is restated in those
terms.

**The download-failure return at :168 leaking an archive (was half of F6).** Refuted; see the
end of section 2. That path always passes through the `rm -f` at :160 first.

---

## Findings, ordered by severity

| # | Severity | Finding | Line(s) | Evidence |
|---|---|---|---|---|
| F1 | **Blocker** | A truncated download passes the kept-count gate; the month is marked `.corpus_done` and its raw archive deleted, leaving a silently time-biased sample that cannot be recovered. Reachable only when the month has no `EXPECTED_SIZES` entry. | CLAUDE_TASK.md:154, :162, :167, :176, :179 | VERIFIED end to end with the real `preprocess_lichess.py`: truncated to 45%, exit 0, `scanned=4410` vs 12000, `kept=3000` (full). `zstandard` never raises on truncation. |
| F2 | **Blocker** | `curl` lost `-f`, so a 404 or 5xx body is written into the `.zst` and curl exits 0. Feeds F1 and wastes whole months. | CLAUDE_TASK.md:156 vs `corpus_stream.sh:70` | VERIFIED: 404 wrote a 54-byte HTML body, `rc=0`. |
| F3 | **High** | Claims are never released on SIGKILL, OOM, or reboot (no `trap`), and never released on success either, so a claim cannot distinguish "running" from "finished" from "stranded". Stranded months need manual `rmdir`. | CLAUDE_TASK.md:140-141, :168, :177, :179 | VERIFIED by grep: no `trap`, `flock`, `$$`, hostname or timestamp anywhere. |
| F4 | **High** | Preprocess has zero retries and `xargs` feeds each month exactly once, so one transient failure drops a month for the entire multi-week run. Serial retried 3 times. | CLAUDE_TASK.md:172-177, :193 vs `corpus_stream.sh:89-101` | Direct comparison. |
| F5 | **High** | The run always exits 0 and always prints "all months ... processed", regardless of how many months failed. `xargs`'s status is discarded and the trailing `log` resets `$?`. | CLAUDE_TASK.md:193-194 | VERIFIED: reproduced `PARENT SCRIPT EXIT = 0` with a deliberately failing worker. |
| F6 | **High** | Peak raw staging is about 153 GB typical / 168 GB worst case at `CONCURRENCY=5`, versus the serial design's documented ~30 GB, on a `/home` reported at ~92% full. No free-space check exists. | CLAUDE_TASK.md:112, :125-135 vs `corpus_stream.sh:15-16`, CLAUDE.md:27 | Computed from the script's own size table; grep confirms no `df`/`ulimit`. Live free space UNCERTAIN. |
| F6b | **High** | A month that fails the kept-count gate leaks its complete ~30 GB archive: the only `rm -f` on a finished month is the success path at :179, and the :177 return skips it. Nothing reclaims it, because F3's stranded state and F4's single-pass feed mean no later run revisits the month. (The :168 download-failure return does not leak: :160 already deleted the file on that path.) | CLAUDE_TASK.md:160, :177, :179, :193 vs `corpus_stream.sh:80`, :104, :144 | Direct read of both `rm -f` sites; serial deletes the raw inside its loop and exits on failure. |
| F7 | **High** | `bash "$0"` is resolved after `cd`, so a relative-path launch makes every worker die instantly while the run reports success. | CLAUDE_TASK.md:106, :193 | VERIFIED: reproduced, all workers `No such file or directory`, parent exit 0. Whether the live launch uses a relative path is UNCERTAIN. |
| F9 | **Medium** | `-C -` resume is defeated by the `rm -f` on every mismatch, so each retry re-downloads ~30 GB from zero, up to ~180 GB per stubborn month. `--retry`/`--retry-delay` also dropped. | CLAUDE_TASK.md:156, :160 vs `corpus_stream.sh:65-77` | Direct comparison. |
| F10 | **Medium** | Both startup validations were dropped: the approved month-window regex and the mandatory "every month has an expected size" assertion. The latter is exactly what would have prevented F1's exposure class. | CLAUDE_TASK.md:105-135 vs `corpus_stream.sh:53-57` | Absent from the parallel script. |
| F11 | **Medium** | Nothing marks a month as in flight (`.corpus_done` is written only at :179, the claim carries no metadata), so an operator clearing a claim, which F3 makes the only recovery path, can start a second worker on a live month. Both then `rm -rf` and repopulate the same output directory. Content is identical (same default `--seed 42`, same input), so the damage is an `rm -rf` under a live writer and a possible `.corpus_done` over a partial directory. | CLAUDE_TASK.md:171, :179, `preprocess_lichess.py:119`, :133, :177 | Requires operator intervention. The :147/:148 ordering is not the cause: see section 6. |
| F12 | **Medium** | The preprocess exit status is discarded, so a python crash is reported as "only 0/30000 kept". | CLAUDE_TASK.md:172-173 | No `$?` check, `set -e` not in effect. |
| F13 | **Medium** | `cd ~/<workdir>` lost its `|| exit 1`; a failed `cd` sends a 30 GB download to a wrong relative path. | CLAUDE_TASK.md:106 vs `corpus_stream.sh:22` | Direct comparison. |
| F14 | **Medium** | Editing the deployed script in place while a run is live corrupts the parked parent's byte offset and executes garbage. Atomic-rename edits are safe. Also causes silent old/new version skew across workers. | CLAUDE_TASK.md:193 | VERIFIED: in-place `cat >` made the parent execute a comment fragment as a command; `mv` was clean. |
| F15 | **Medium** | Observability regressions: no `tee` to a persistent log, no progress file, so a multi-week run's record lives only in tmux scrollback. | CLAUDE_TASK.md:137 vs `corpus_stream.sh:27`, :32-33 | Direct comparison. |
| F16 | **Low** | Killing the parent with `kill <pid>` leaves `xargs`, five worker shells, and their `curl`/`python` children orphaned with claims still held. Only a foreground Ctrl-C cleans up. | CLAUDE_TASK.md:193 | No `trap`, no process-group handling. |
| F17 | **Low** | Any interruption after the download completes and before :179 (kill, OOM, node reboot) leaves the ~30 GB archive behind, since `rm -f "$zst"` runs only on the success path, and the stranded claim (F3) stops any later run from revisiting the month to clean it. Disk waste an operator can see with `ls`, not data loss. | CLAUDE_TASK.md:147, :179 | Same code path as F6b, different trigger. The narrow `touch`/`rm` window originally reported is negligible: see section 6. |
| F18 | **Low** | `export -f` / `export` are inert under this design and misleadingly imply state reaches workers through the environment. | CLAUDE_TASK.md:183-184 | Each worker re-reads the whole file. |
| F20 | **Low** | `[ "$sz" = "$exp" ]` is a string compare where the serial used `-eq`; and the kept-count gate demands `>= 30000` with no path to accept a genuinely smaller month. | CLAUDE_TASK.md:159, :176 vs `corpus_stream.sh:67`, :95 | Works today, fragile. |
| F19 | **Low (not live)** | On a tree where the parent has never run, `--worker` fails because the `mkdir -p` of `logs`/`CLAIM_DIR` at :191 sits after the worker branch. On the deployed tree those directories exist, so hand-running `--worker <month>` works and is a usable recovery path. One-line hardening. | CLAUDE_TASK.md:186-189 vs :191 | VERIFIED: `mkdir` on a missing parent fails identically to `mkdir` on an existing dir, and stderr is discarded at :140. |

F8 (bash 3.2 / BSD `stat`) was withdrawn after verification. See section 6.

---

## What to fix first

1. **Check `local_corpus_stream_parallel.sh` for F1 exposure right now, before it finishes
   another month.** Confirm every month in its `MONTHS` (2024-09 onward) has an
   `EXPECTED_SIZES` entry. If any does not, stop that script. This is the only finding here
   that destroys data that cannot be regenerated by a retry, and it is destroying it while the
   script runs.
2. **Audit already-completed months for F1 damage.** For every directory holding
   `.corpus_done`, read `.sampling_summary.json` and compare `scanned` against the other
   months. A month whose `scanned` is far below its neighbours was truncated and must be
   redone. This is cheap, it needs no re-download, and it tells you exactly how much damage
   exists. `kept` alone will read 30000 for both good and bad months, so `scanned` is the
   field that matters.
3. **Add `-f` back to curl and make the size check mandatory (F2, F10).** Restore the
   serial version's startup assertion that every month has an expected size, and make a
   missing entry a hard startup failure rather than a skipped check. Together these close F1
   at the source.
4. **Check free space on `/home` before raising anything, and consider lowering
   `CONCURRENCY` until that is known (F6).** 153 GB of staging on a disk reported at 92% full
   is the most likely cause of a mass failure this week, and it would also hit the other 38
   users. While you are there, `ls data/raw_zst` and remove any archive left by a failed or
   interrupted month (F6b, F17).
5. **Make the claim self-describing and add a `trap` (F3).** Write host, PID and a timestamp
   into the claim directory, release on success as well as on failure, and trap
   `EXIT INT TERM` to release. That single change is the foundation for the staleness
   detection and auto-retry Part D asks for, it is what makes the local machine's
   power-cycle scenario recoverable, and it is what would let an operator tell a live worker
   from a dead one before clearing a claim (F11).
6. **Make failures visible (F5, F4).** Track per-month results, print a real summary, and
   exit nonzero if any month failed. Then re-add the serial version's preprocess retry loop,
   and delete the raw archive on the failure path as well (F6b).
7. **Make the re-invocation path absolute (F7)** and write down the rule that the deployed
   script is only ever edited via atomic rename, never in place, while a run is live (F14).

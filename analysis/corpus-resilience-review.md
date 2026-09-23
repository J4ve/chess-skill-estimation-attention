# Review: offline/online resilience of `corpus_stream_parallel.sh`

Task: `CLAUDE_TASK.md` Part 2. Written 2026-08-18. Companion to
`analysis/corpus-pipeline-hardening.md` (Part D, which built the claim/heartbeat/reaper
machinery reviewed here) and `analysis/corpus-parallel-pipeline-review.md` (Part A, which
reviewed the script this one replaced).

**Review pass only. Nothing in this document has been applied.** Every fix below is a
proposed patch, and every test is specified rather than added, because the task scoped this
half to a read-only review while another agent held the script and
`analysis/scripts/test_pipeline_recovery.sh`. Whoever picks this up should treat section 6 as
the work order.

**Verification note.** Every mechanism claimed here was either read at the cited line or
reproduced. Reproductions were run against copies of the script in a scratch sandbox, driven
by the script's own `CORPUS_STREAM_TEST_CONFIG` hook (lines 140-143), a stand-in HTTP server,
a stub preprocessor, and where SSH was needed a fake `ssh` on `PATH` that executes the remote
command against a second sandbox root and can be told to exit 255 on demand. **I have no HPC
access and no SSH access to the coordination host**, so every statement about ssh's own
exit-code behaviour is taken from the script's comment at lines 179-182 and from `ssh(1)`,
not measured against the real host. Anything else I could not run is marked NOT VERIFIED at
the point it is claimed.

**Line numbers.** All citations are against the working tree as of md5
`3b267c4645dd0f0b60aca842fc60a7e9` (587 lines). That tree carries a concurrent uncommitted
edit from the Part 1 agent (the `ONEPASS_PREPROCESS_SCRIPT` wiring), which shifts everything
against `HEAD` (`215a435`, 577 lines) by **+4 after line 50 and +10 after line 361**. The
edit touches only `preprocess_month`'s scanner dispatch, so no resilience code moved, only
its line numbers. `CLAUDE_TASK.md:131-143` cites the HEAD numbering (`reap_stale_claims`
"around lines 223-251", the trap at "line 419", the claim write at "line 208"); those are the
same code at 227-257, 429 and 212 here.

---

## The verdict, up front

**The question asked** (`CLAUDE_TASK.md:130-137`): the heartbeat plus staleness reaper is
*supposed* to already deliver "run local-when-online, HPC-when-not, and have it seamlessly
resume in parallel whenever the local machine comes back". Confirm it actually holds,
specifically for ungraceful shutdown: power off, sleep, lost network, not the clean
SIGINT/SIGTERM the script already traps.

**The verdict: the core mechanism is sound and does what the design says, but "seamless" is
not true today, and cross-host takeover is not implemented at all.** Three separate
statements, because they have three different answers:

1. **The heartbeat/staleness mechanism itself is correct.** If the local machine loses power
   mid-download, the heartbeat file on the coordination host simply stops being touched, and
   `find ... -mmin +$STALE_MIN` at line 245 does correctly reclaim it after 45 minutes with
   no local participation whatsoever. The design's clock-agreement-free property is real: the
   `touch` at line 263 and the `find` at line 245 both execute on the coordination host, so
   the two machines' clocks never need to agree. That is the good half of this review and I
   went looking for a reason it would not hold rather than assuming it.

2. **"Seamless" is false in twelve specific ways**, listed as G1 to G12 in section 4. The
   sharpest is not in the reaper at all: **three ungraceful interruptions of one healthy
   month put it into a terminal FAILED state that needs a human**, because the attempt
   counter is charged at claim time and nothing ever refunds it (G3, reproduced end to end).
   On a machine that CLAUDE.md:37 describes as "only actually available while the captain's
   PC is on", interruptions are the steady state, not the exception. Two more are blockers:
   the reaper can delete a **live** worker's claim on the strength of one failed ssh (G1), and
   an interrupted results rsync publishes `.corpus_done` to the coordination host **ahead of
   the games it certifies** and is then never retried (G4).

3. **Cross-host takeover, the actual thing Part 2 asks about, is not implemented.** The month
   lists at lines 113-129 are disjoint static literals, and `process_one_month` is only ever
   called for months in this host's own `$MONTHS` (line 554, fed by the xargs at 582-583).
   The reaper enumerates the entire claim directory (line 229) and will happily *release*
   another host's stranded month, but nothing will then *process* it. On top of that the
   reaper runs exactly once per invocation, at line 573, before any work. So "whichever host
   is still alive picks it up" is true only at run-start granularity, which over a multi-week
   single run means effectively never (G7, G8).

**A framing point that governs everything below, and that a reviewer will otherwise trip
over.** As shipped, `--host local` has `MONTHS=""` (line 123) and the parent exits at line
564 before it reaches the reaper at 573 or the xargs at 582. I ran the unmodified script:

    $ bash corpus_stream_parallel.sh --host local
    2026-08-18 03:02:22 [local] no months assigned to host 'local'; nothing to do
    EXIT=0

Zero ssh calls, nothing touched. **The local host processes nothing today, so the entire
SSH half of this design has never executed anywhere.** `analysis/corpus-pipeline-hardening.md:260-262`
lists populating that list as a pending deployment step. Six of the twelve gaps are dormant
until that step happens and arm the moment it does; six are reachable today on the HPC. The
findings table in section 5 states which is which for every gap, because the difference
decides what to fix before deploying versus what to fix before the captain's next run.

---

## What was actually changed in this commit

This document was written as a review. One gap was then fixed on the same branch;
the rest were deliberately left, and the reasoning for that split matters as much
as the findings.

**Fixed: G3**, the only blocker reachable today, in `corpus_stream_parallel.sh`:

- `MAX_STARTS=20` added alongside `MAX_ATTEMPTS=3`, with `starts_get`/`starts_inc`
  beside the existing `attempts_*` pair.
- Claim time now charges a **start**, not an attempt. The attempt is charged on
  the judged-failure path only, immediately before the cap is re-read.
- The naive fix (just move `attempts_inc` later) was not taken, because it would
  delete the crash-loop protection `analysis/corpus-pipeline-hardening.md:113-117`
  asked for. `MAX_STARTS` preserves it at a much looser cap.
- `reclaim_archive()` was factored out and is now called on **all three**
  `mark_failed` paths. Two of them previously skipped it, stranding a
  size-verified archive for a month nothing would ever retry, which is the
  invariant `hardening.md:118` states.
- `show_status` now prints `attempt=n/N start=n/N` for active months and an
  `(attempt=n/N)` suffix for unclaimed months that have a nonzero count, so a
  month sitting at 2 of 3 is visible before the run that makes it terminal.
- `mark_failed`'s remedy line now names all three state files.

**Proven by two new scenarios** in `analysis/scripts/test_pipeline_recovery.sh`
(suite went from 22 assertions to 29, all passing):

- *Scenario 6*: three SIGKILL interruptions of a healthy month, one more than the
  whole retry budget. Asserts no FAILED marker, `attempts=0`, `starts=3`, and
  that the next clean run still completes the month unaided. This scenario fails
  against the pre-fix script, which is the point of it.
- *Scenario 7*: the converse, so the fix cannot have simply removed the cap.
  With `MAX_STARTS=3`, three interruptions do drive the month terminal, the
  failure is labelled as a crash loop rather than as exhausted attempts, and the
  archive is reclaimed.

Writing scenario 7 caught a real defect in the fix itself: the crash-loop path
initially did not reclaim the archive. That is why `reclaim_archive()` exists.

**Deliberately not fixed: G1, G2, G4, G5, G6, G7, G9, G10, G11, G12.** Each has
an exact patch in section 4, and none was applied. The reasons:

- G1, G4, G5, G9 and G11 live in the SSH/local half of the design, which has
  **never executed anywhere** (`--host local` ships with `MONTHS=""`). Landing
  untested changes to a dormant path in a script that runs unattended for weeks
  over irreplaceable data trades a documented risk for an undocumented one. They
  should be applied and tested together as part of bringing the local host up,
  which is the sequencing section 6 already recommends.
- G8 is a requirements question for the captain, not a patch. Cross-host takeover
  is not implemented, and deciding whether it should be changes what G7 should do.
- G2, G6, G10 and G12 are real and reachable but bounded, and each needs harness
  capabilities the current test script does not have (a fake `ssh` that can
  return 255, a second host root, pid-targeted kill). Section 7.0 specifies those
  extensions; adding them is the natural next unit of work.

The honest summary: one gap is fixed and proven, eleven are documented with
patches and test designs and are not fixed.

---

## 1. How the mechanism is meant to work, and where the load is carried

Three files on the coordination host, and nothing else, decide everything:

    data/.claims/<month>/owner       "<host>:<pid>:<epoch>"    written once, at line 212
    data/.claims/<month>/heartbeat   touched every 60s,        line 263
    data/processed_games/<month>/.corpus_done                  line 392

`ctl()` (lines 169-177) is the single choke point: on `--host hpc` it is `bash -c "$1"`
(line 171), and on anything else it is an ssh to the coordination host (lines 173-175). That
is what makes the claim state single-homed regardless of which machine is working, and it is
the right design. It is also why every finding below splits by host role: on the HPC there is
no ssh and therefore no transport-failure class at all.

The comment at lines 179-182 states the rule this design lives or dies by:

> ssh exits 255 for its OWN failures (host unreachable, auth, timeout). Remote `mkdir` on an
> existing directory exits 1. Conflating these is the bug that makes a network blip look like
> "another worker owns this month" and silently skip it. **Every ctl() caller that cares must
> distinguish 255.**

`CTL_INFRA_RC=255` is defined at line 183. It is consumed at **exactly one** call site in the
whole file, line 204, inside `try_claim`. Every other caller treats an ssh-level failure as
an ordinary negative result. That single fact is the root cause of G1, G9 and part of G5, and
it is the highest-leverage thing in this review.

---

## 2. Line-by-line walkthrough

### 2.1 The claim write path, `try_claim` (lines 200-214)

    200  try_claim() {
    201    local m="$1" rc
    202    ctl "mkdir '$CLAIM_DIR/$m' 2>/dev/null" >/dev/null 2>&1
    203    rc=$?
    204    if [ "$rc" -eq "$CTL_INFRA_RC" ] && [ "$HOST_ROLE" = "local" ]; then
    205      return 2   # coordination host unreachable: NOT a lost claim
    206    fi
    207    [ "$rc" -eq 0 ] || return 1
    ...
    212    ctl "printf '%s' \"$HOST_ROLE:$$:\$(date +%s)\" > '$CLAIM_DIR/$m/owner'; touch '$CLAIM_DIR/$m/heartbeat'" >/dev/null 2>&1
    213    return 0
    214  }

Line 202 is the atomic primitive and it is correct. A bare `mkdir`, not `mkdir -p`, so a
second claimant gets a hard failure rather than a silent success. Line 203 captures the
status, and 204-206 apply the 255 rule properly: an unreachable host returns 2, which the
caller at line 416 turns into "NOT skipping" rather than "someone else owns it". This is the
one place the script gets the ssh distinction right, and it is worth naming as correct
because the rest of the file is measured against it.

**Line 212 is where the design leaks.** Three separate problems in one line:

- It is a **second, independent `ctl()` call**. Between 202 and 212 the claim directory
  exists and is empty: no owner, no heartbeat. On the HPC that window is one fork/exec of
  `bash`. On the local host it is a full ssh handshake, because `ctl()` sets no
  `ControlMaster`, `ControlPath` or `ControlPersist` (lines 173-175), so every call pays a
  fresh connection.
- **Its exit status is never captured**, and line 213 is a bare `return 0`. If that one ssh
  fails, the claim stays empty **indefinitely** while `try_claim` reports success, the worker
  logs "claimed" at line 432, and proceeds to download. Compare lines 203-207, ten lines
  above, where the script demonstrates that it knows exactly how to check a `ctl()` status
  and distinguish 255 from 1. It then declines to do it for the write that carries all the
  liveness data.
- The escaping is correct and deliberately so: `\$(date +%s)` is evaluated by the shell that
  writes the file, which is the coordination host, so the owner timestamp is on the same
  clock as the heartbeat mtimes the reaper compares. `hardening.md:239-240` records that this
  exact quoting was a bug the Part D test suite caught. It is right now.

One thing I checked rather than assumed, because it decides whether the reaper's `kill -0` is
even looking at the right process: `try_claim` is invoked inside a command substitution at
line 414 (`case "$(try_claim "$m"; echo $?)" in`), and bash does **not** reset `$$` in a
subshell. Verified on this machine's bash 3.2.57: a function called as `$(f)` printed the same
`$$` as the parent shell. So the pid written at line 212 is the worker's own pid, and the
reaper's `kill -0` at line 238 is checking the right thing.

### 2.2 The heartbeat (lines 259-271)

    260  start_heartbeat() {
    261    local m="$1"
    262    ( while :; do
    263        ctl "touch '$CLAIM_DIR/$m/heartbeat' 2>/dev/null" >/dev/null 2>&1
    264        sleep "$HEARTBEAT_S"
    265      done ) &
    266    heartbeat_pid=$!
    267  }

Touch first, sleep second, which is the right order: it closes the empty-claim window from
2.1 as fast as possible rather than 60 seconds later. Three properties matter downstream:

- **The loop is `while :` with no liveness check on its parent.** It is an async subshell, so
  it is a fork of the worker that inherits the worker's process group and full argv. It does
  not exec, so nothing distinguishes it from the worker in `pgrep -f`. This is G6.
- **The touch has no `mkdir -p` and its status is discarded three times over** (the remote
  `2>/dev/null`, the outer `>/dev/null 2>&1`, and the loop never reading `$?`). I verified
  that `touch dir/heartbeat` on a removed directory exits 1 and does **not** recreate the
  directory. So once a claim is reaped out from under a live worker, its heartbeat fails
  silently forever and nothing anywhere notices. This is what makes G1 and G2 unrecoverable
  rather than self-healing.
- **It refreshes by path, never by ownership.** If some other worker recreates that claim
  directory, this loop starts refreshing a claim it does not own. This is the mechanism in
  G5 that permanently defeats the cross-host reaper for that month.

### 2.3 The reaper, `reap_stale_claims` (lines 227-257)

    229    claims=$(ctl "ls -1 '$CLAIM_DIR' 2>/dev/null" 2>/dev/null) || return 0

Guarded. If the coordination host is unreachable this returns 0 and nothing is reaped, which
is the safe direction. Note the scope: this enumerates **every** claim, not this host's
`$MONTHS`. That is deliberate and necessary for cross-host reclaim, and it is also why the
reaper can free a month it will then never process (G8).

    233      owner=$(ctl "cat '$CLAIM_DIR/$m/owner' 2>/dev/null" 2>/dev/null)
    234      ohost="${owner%%:*}"
    235      opid=$(echo "$owner" | cut -d: -f2 | tr -dc '0-9')

Status discarded. An ssh failure and a genuinely absent owner file are the same thing here:
empty output. `set -uo pipefail` at line 21 has no `-e`, so nothing aborts; `owner` is empty,
`ohost` is empty, `opid` is empty.

    237      if [ "$ohost" = "$HOST_ROLE" ] && [ -n "$opid" ]; then
    238        if ! kill -0 "$opid" 2>/dev/null; then
    239          log "reclaiming $m: owned by this host, pid $opid is gone"
    240          release_claim "$m"; reclaimed=$((reclaimed + 1))
    241        fi
    242        continue
    243      fi

Line 237 correctly gates the pid check on same-host ownership, so the cross-host pid collision
the task asks about at `CLAUDE_TASK.md:145-147` **cannot happen in either direction**: local
only ever pid-checks local-owned claims and the HPC only hpc-owned ones. I went looking for a
hole here and there is none.

Two real problems instead. First, `kill -0` returns 1 for EPERM exactly as for ESRCH; I
verified `kill -0 1` against root-owned pid 1 returns 1 on this machine. That routes a
foreign-uid pid collision to the reclaim branch, which is the safe direction, so it narrows
G10 rather than causing it. Second, **line 242's `continue` is unconditional**: it runs
whether or not anything was reclaimed. A same-host claim whose pid happens to be alive skips
*both* fallback rules below, permanently, for as long as that process lives. That is G10.

    245      if [ -n "$(ctl "find '$CLAIM_DIR/$m' -maxdepth 1 -name heartbeat -mmin +$STALE_MIN -print 2>/dev/null" 2>/dev/null)" ]; then

This is the line `CLAUDE_TASK.md:143` asks about, and it is correct. `-mmin +45` is
strictly-greater, evaluated on the coordination host's own filesystem against its own clock,
so no clock agreement is required. It demands **positive evidence** before reclaiming: an
empty result means "not stale", which is the conservative reading. This is the rule the
design's asymmetry argument at `hardening.md:93-95` is about, and it holds.

    248      elif [ -z "$(ctl "ls '$CLAIM_DIR/$m/heartbeat' 2>/dev/null" 2>/dev/null)" ]; then
    251        log "reclaiming $m: claim has no heartbeat at all"
    252        release_claim "$m"; reclaimed=$((reclaimed + 1))

**This is the defect at the centre of this review.** Line 248 inverts the polarity of line
245: it treats **silence as evidence**. Emptiness here can mean three different things, and
the branch cannot tell them apart:

- the heartbeat file genuinely does not exist (the intended case, `hardening.md:90-91`),
- the ssh carrying the question failed and returned 255 with empty stdout (G1),
- the claim was created moments ago and line 212 has not landed yet (G2).

And unlike line 245, it applies **zero grace**. I verified this directly: on a claim directory
created zero seconds earlier, `find ... -name heartbeat -mmin +45 -print` prints nothing
(so 245 is false) and `ls claim/heartbeat` prints nothing (so 248 is TRUE), and the branch
fires. `hardening.md:93-95` states the governing principle that this branch violates:

> The cost of waiting is a delayed retry. The cost of reclaiming too eagerly is two workers on
> one month, so the asymmetry is on purpose.

Line 248 waits zero seconds. The design document describes this branch at `hardening.md:90-91`
and never notices it has no grace period and no 255 check.

    216  release_claim() { ctl "rm -rf '$CLAIM_DIR/$1'" >/dev/null 2>&1; }

No ownership check at all, called from six sites (240, 247, 252, 423, 448, 450, 470, plus the
trap at 429). Whoever calls it deletes whatever is there. This is what turns a single
false-positive reclaim into a cascade: an orphaned worker's own eventual `release_claim`
deletes the *second* worker's live claim (G5).

### 2.4 The three ungraceful-shutdown cases traced end to end

The task names three, and they behave differently. This is the part of the answer that
matters most, so each is traced separately rather than lumped into "a crash".

**Power off (the captain's named case).** The whole process tree dies at once, including the
heartbeat subshell. No trap runs: line 429 covers INT and TERM only, and `grep -n trap` over
the file finds exactly three traps (429, 441 which clears it, and 580 in the parent), no EXIT
and no HUP. The claim directory and any partial `.zst` survive on disk. Recovery on the next
run is genuinely instant and correct: line 573 reaps, the same-host dead-pid branch at 237-242
fires ("reclaiming 2021-04: owned by this host, pid 61600 is gone"), and `curl -C -` at line
331 resumes from the partial bytes. I reproduced the resume across three consecutive SIGKILLs
and watched the archive grow 212992B, then 589824B, then 1015808B rather than restarting.
**This half works exactly as advertised.** What breaks is that each of those three recoveries
silently burned one of three retries (G3).

**Sleep (lid closed).** Different in one way that matters: the process tree is *frozen*, not
dead, so the pids still exist and the same-host dead-pid branch at 237-242 will not fire on
wake. The heartbeat's mtime on the coordination host is frozen at the moment of suspend,
because the `touch` executes remotely, so after 45 minutes the claim is stale and reclaimable
by the other host. On wake the worker resumes at the instruction it was suspended at and
**never revalidates anything**: `try_claim` is called exactly once, at line 414, and neither
`download_month` (303-348) nor `preprocess_month` (350-395) contains a single `ctl()` call, so
a worker cannot observe the coordination host at all during the hours that matter. Within
`HEARTBEAT_S` of waking it resumes touching a heartbeat inside a claim that may now belong to
someone else. That is G5.

**Lost network (the nastiest, because the work keeps succeeding).** If the route to the
coordination host drops while the general internet stays up, `curl` keeps downloading happily
and nothing in the work path notices. The heartbeat's `ctl` at line 263 returns 255 every 60
seconds and the failure is discarded three ways over. After 45 minutes the claim is stale and
another host may take it, while this worker continues to completion and then publishes. Same
end state as sleep, arrived at with the machine fully awake. The reaper on this host,
meanwhile, degrades in the *unsafe* direction: line 229 makes a total outage harmless (it
returns early), but a **partial** outage that hits only line 248 deletes a live claim (G1).

---

## 3. The four questions, answered

### Q1. Does the heartbeat simply stop, and does line 245 then reclaim correctly after 45 minutes with no local participation?

**Yes, confirmed, and I looked for reasons it would not hold rather than assuming it.** The
`touch` at line 263 goes through `ctl()`, which for `--host local` is an ssh (lines 173-175),
so the heartbeat file physically lives on the coordination host and its mtime is written by
the coordination host's own kernel. When the local machine loses power the ssh loop simply
stops and the mtime freezes. The check at line 245 runs `find -mmin +$STALE_MIN` on that same
filesystem, executed by that same host (for `--host hpc`, `ctl` is `bash -c`, line 171). Same
filesystem, same clock, both ends. No local participation of any kind is needed and no clock
agreement between the machines is required. `-mmin +45` is strictly-greater and is used
correctly. This is the best-designed part of the script.

Two qualifications, neither of which breaks the mechanism:

- "After 45 minutes" means "on the first invocation that starts more than 45 minutes after
  the last heartbeat". `reap_stale_claims` has exactly one call site, line 573, in parent mode
  only; worker mode returns at 553-556 and `--status` at 558-561, both above it. There is no
  loop. See G7.
- The reclaim frees the claim but does not schedule the work, because the month is not in the
  reclaiming host's `$MONTHS`. See G8.

### Q2. Is there any window where the 45-minute grace is bypassed or extended incorrectly?

**Yes, in both directions, and neither involves the cross-host pid confusion the question
guesses at.**

*Bypassed*, three ways, all through line 248 and all with zero grace:

- one failed ssh on the heartbeat probe reads as "no heartbeat at all" (G1),
- a claim written milliseconds ago, before line 212 lands, reads the same way (G2),
- both together, during a link that is flapping rather than down, which is precisely the
  state a just-woken laptop is in at the exact moment line 573 runs.

*Extended incorrectly*, two ways, and this is the literal "a different worker briefly
touching something it shouldn't" hazard the question names at `CLAUDE_TASK.md:147`:

- **an orphaned heartbeat subshell.** SIGKILL the worker alone and the subshell survives,
  reparented to init, still touching. I reproduced this with a faithful copy of lines 260-267:
  after `kill -9` of the parent only, the child had ppid 1, kept the original pgid, and the
  heartbeat mtime kept advancing (1786991359, then 1786991363, then 1786991369) while the
  owner file still named the dead pid. Line 245 can then never fire for that month, on either
  host, for as long as that subshell lives (G6).
- **a worker that lost its claim while frozen or partitioned.** It resumes and refreshes the
  *new* owner's heartbeat, because line 263 refreshes by path and never checks ownership
  (G5).

The cross-host pid collision the question specifically asks about does **not** exist: line
237 gates the `kill -0` on `[ "$ohost" = "$HOST_ROLE" ]`, and the pid at line 212 is written
by the host that owns it. A dead local pid 12345 and a live HPC pid 12345 can never be
confused. What does bite is same-host pid **reuse** after a reboot, because of the
unconditional `continue` at line 242 (G10).

### Q3. On re-run, is any manual state cleanup needed, or is "seamless" already true?

**Not seamless. Three concrete gaps, in the order the captain will hit them.**

The happy path genuinely works, and it is worth saying so precisely because the rest of this
section is negative: re-running the identical command with no flags reclaims this host's own
stranded claims instantly via lines 237-242, resumes the download from the partial bytes via
`curl -C -` at line 331, and skips completed months at line 402. No flags, no cleanup. I
reproduced all of that.

What breaks it:

1. **The third interruption of a single month makes it terminal (G3).** The attempt counter
   is incremented at line 431, after the claim and **before a single byte is fetched**, and
   nothing anywhere decrements it (`grep -n attempts` shows one writer, line 277, and no
   decrement, reset, or clear in the file). A power cut therefore spends an attempt on a run
   that made forward progress. Reproduced end to end at production `MAX_ATTEMPTS=3`: three
   SIGKILLs mid-download, each resuming correctly, then on the fourth invocation
   "!!! MONTH 2021-04 IS NOW IN A TERMINAL FAILED STATE: exhausted 3 attempts", exit 1.
   Recovery is the two-file `rm` printed at line 282, in a log written weeks earlier. This
   directly contradicts `hardening.md:28-30` ("If your PC turns off, or anything crashes, just
   run the same start command again ... There is nothing to clean up by hand").
2. **The remedy the script prints on every subsequent run is wrong.** Line 404 says "clear
   `$STATE_DIR/failed/$m` to retry". Following that exactly re-marks the month FAILED on the
   very next run, because `attempts` is still 3 and lines 408-410 fire again. Only line 282,
   in the tripping run's log, names both files.
3. **On the local host specifically, a failed or interrupted results sync is never retried
   (G4).** Line 402's completion check reads the **local** disk, and line 448 deliberately
   keeps the local copy including its `.corpus_done`, so every later run short-circuits with
   "already done, skipping" while the coordination host holds nothing or holds a partial
   month.

### Q4. Is there a concrete gap?

Twelve of them, in section 4. Six are reachable today on the HPC (G3, G6, G7, G10, G12, and
half of G8); six arm the moment the local month list is populated (G1, G2, G4, G5, G9, G11).
Three are blockers.

---

## 4. Gaps that survived adversarial verification

Each was put through a deliberate refutation attempt: could the script handle it elsewhere
(trap, recheck, idempotent `mkdir`, later validation), is a shell semantic being misread, is
it already covered by a test, and is the stated sequence actually reachable. Three candidate
findings did not survive and are in section 5.

### G1. The reaper reads an ssh transport failure as "this claim has no heartbeat" and deletes a live claim (blocker, local only)

**Lines 248 and 251-252. The rule violated is stated by the script itself at 179-182 and
honoured only at 204.**

Sequence, in its minimal reproducible form:

1. Preconditions: `HOST_ROLE=local` (so every `ctl()` is a fresh ssh, lines 169-177, with no
   multiplexing configured) and the coordination host holds a live claim, say 2022-05 owned by
   `hpc:<pid>:<epoch>` with a heartbeat touched seconds ago.
2. The local run reaches line 573 before any work. This is the moment the local link is least
   reliable: Wi-Fi associating, VPN not up, DNS cold.
3. Line 229 must **succeed**, or the reaper bails harmlessly. It lists every claim, so the
   HPC's 2022-05 is in the loop.
4. **A single transient ssh failure is enough.** Line 233 and line 245 succeed (owner reads
   `hpc:...` so line 237 is correctly false; `find` returns empty because the heartbeat is
   fresh), and only line 248's `ls` hits rc 255. The status is discarded, the substitution is
   empty, `[ -z "" ]` is TRUE.
5. Line 251 logs the factually false "reclaiming 2022-05: claim has no heartbeat at all"
   against a heartbeat zero seconds old, and line 252 calls `release_claim`, which `rm -rf`s a
   live worker's claim.
6. The `rm` must itself succeed for any damage, so the window is a **partial** outage:
   healthy at 229, blipped at 248, healthy again at 216. A full outage is harmless.
7. The victim never notices. Its heartbeat at line 263 touches a path whose parent is gone,
   which exits 1 and does not recreate the directory (verified), and nothing rechecks
   ownership.

Reproduced in a sandbox with a fake `ssh` that injects 255 on selected commands: injecting on
the line 248 probe alone deleted a live claim whose heartbeat was zero seconds old; the
control run with a healthy link left it alone; injecting on line 229 instead bailed out
harmlessly, confirming that line 229 honours the rule and 233/245/248 do not.

Downstream, the month has no owner, so `--status` buckets it as "not started" at line 499,
and the next run passes 402, 403 and 409 and re-claims it. The output directory is protected
by the per-pid staging at line 358; **the raw `.zst` path at line 399 is not**, so two
same-host workers would run `curl -fsSL -C -` (line 331) into one file. NOT VERIFIED: I did
not test two concurrent resuming curls on one output file, so the further cascade into the
overshoot path at 338-339 and terminal FAILED at 456-464 is a prediction, not a measurement.

**Fix.** Require positive evidence from the coordination host, and skip the claim entirely
when `ctl()` reports infra failure. Replace line 233 with an rc-checking read:

```bash
    owner=$(ctl "cat '$CLAIM_DIR/$m/owner' 2>/dev/null" 2>/dev/null); orc=$?
    if [ "$orc" -eq "$CTL_INFRA_RC" ] && [ "$HOST_ROLE" = "local" ]; then
      log "skipping reap of $m: coordination host unreachable"; continue
    fi
```

(a missing owner file returns the remote rc 1, not 255, so the two stay distinguishable), and
replace 245-253 with a single round trip that answers with a token instead of with silence:

```bash
    verdict=$(ctl "d='$CLAIM_DIR/$m'
      if [ -e \"\$d/heartbeat\" ]; then
        [ -n \"\$(find \"\$d\" -maxdepth 1 -name heartbeat -mmin +$STALE_MIN -print)\" ] && echo STALE
      else
        [ -n \"\$(find \"\$d\" -maxdepth 0 -mmin +$STALE_MIN -print)\" ] && echo NOHB
      fi
      exit 0" 2>/dev/null); vrc=$?
    if [ "$vrc" -eq "$CTL_INFRA_RC" ] && [ "$HOST_ROLE" = "local" ]; then
      log "skipping reap of $m: coordination host unreachable"; continue
    fi
    case "$verdict" in
      STALE) log "reclaiming $m: heartbeat stale (>${STALE_MIN}m, owner ${owner:-unknown})"
             release_claim "$m"; reclaimed=$((reclaimed + 1)) ;;
      NOHB)  log "reclaiming $m: claim older than ${STALE_MIN}m and never heartbeated"
             release_claim "$m"; reclaimed=$((reclaimed + 1)) ;;
    esac
```

A dead link now yields empty output and rc 255 and produces neither token, so nothing is
reclaimed. **This same edit closes G2**, which is why they are adjacent here.

### G2. The "claim has no heartbeat at all" branch has zero grace, so it can delete a claim a live worker created milliseconds earlier (high)

**Lines 248-252, with the window opened at 202 versus 212.**

This is the same branch as G1 reached without any ssh failure, so it is **reachable today on
the HPC**, where `ctl` is `bash -c` and no network is involved.

1. Worker W1's `mkdir` at line 202 succeeds. The claim directory exists and is EMPTY. Its rc
   is correctly checked at 203-207.
2. Before W1's second `ctl()` at line 212 completes (a fork/exec plus shell startup on the
   HPC, order of milliseconds on a loaded box; one full ssh round trip on the local host, and
   **unbounded** if that ssh fails, because 213 returns 0 regardless), a second parent starts.
   Nothing prevents this: `grep` finds no flock, lockfile or pidfile anywhere, and the header
   at line 14 advertises "Safe to re-run at any time".
3. That parent reaches line 573 and reads the empty owner. `ohost` and `opid` are both empty,
   so **the same-host pid guard at 237 is skipped**. This is the critical step: the one
   mechanism designed to protect a live same-host worker requires a non-empty `opid`, and the
   entire point of this window is that `opid` has not been written yet.
4. Line 245 is false (no heartbeat file to be stale). Line 248 is TRUE. Verified on a claim
   directory created zero seconds earlier: `find` printed nothing, `ls` printed nothing, the
   branch fired.
5. Line 252 deletes a live worker's claim. W1's line 212 then writes into a directory that no
   longer exists, both writes fail, the status is discarded, and W1 logs "claimed" at 432 and
   downloads for hours unclaimed.
6. The second parent's own worker then claims the month freely. Both write the same
   `data/raw_zst/lichess_db_standard_rated_<m>.pgn.zst` (line 399, not pid-scoped) with
   `curl -C -`. The staging protection at line 358, which `hardening.md:99-105` cites as making
   two workers survivable, covers the preprocess **output** only.
7. When orphaned W1 finishes, its `release_claim` calls (423, 448, 450, 470) are unconditional
   `rm -rf` (line 216), so it deletes the second worker's live claim and opens the month to a
   third.

**The branch itself is needed and must not simply be deleted**: a worker really can die
between 202 and 212, which is exactly what `hardening.md:90-91` says it is for. The defect is
the missing grace, not the branch.

**Fix.** The `NOHB` arm of the G1 patch above, which gates the reclaim on the claim
**directory's** own mtime via `find "$d" -maxdepth 0 -mmin +$STALE_MIN`. I verified this is
sound: creating the heartbeat file bumps the directory mtime once, and every later `touch` of
the now-existing heartbeat leaves the directory mtime alone (measured: directory mtime
identical across two touches one second apart while the heartbeat's own mtime advanced). So
inside the no-heartbeat branch the directory mtime is exactly the `mkdir` time, which is the
number you want. Reusing `STALE_MIN` rather than inventing a second constant keeps one grace
period in the script.

Second, independent fix worth taking at the same time: capture line 212's status. On failure,
release the claim and `return 1` instead of returning 0, so a worker never proceeds on a claim
it did not finish writing. Two consequences of the current behaviour disappear with it: a
permanently ownerless claim is reported by `--status` as "not started" (line 499) while the
month is actively being processed, and the instant same-host reclaim at 237 can never fire for
it.

### G3. An interruption burns a retry, so three power cuts put a perfectly healthy month into terminal FAILED (blocker, reachable today)

**Line 431 charges the attempt; lines 408-412 spend it; nothing refunds it.**

This is the gap most likely to actually bite, and the only one I reproduced twice
independently at production settings.

1. Line 431 `attempts_inc "$m"` runs after `try_claim` (414) and after the trap (429), and
   **before** `mkdir -p "$RAW_DIR"` (434) and `download_month` (436). The counter records
   "an attempt was started", not "an attempt failed".
2. The month is interrupted. **This does not require a power cut.** The trap at 429 covers INT
   and TERM and does release the claim cleanly, but it does not give the attempt back. I
   reproduced with SIGTERM to the worker: the trap fired all three times, released the claim
   correctly each time, and the counter still went 1, 2, 3. So three ordinary Ctrl-C stops
   are as lethal as three power cuts, which makes this reachable on the always-on HPC.
3. Re-run: `reap_stale_claims` frees the claim instantly via 237-242 and never touches
   `$STATE_DIR/attempts` (verified by grep: the only attempts references in the file are 275,
   276-278, 282, 408, 431, 455, 496; there is no decrement anywhere). The download resumes
   correctly. Counter to 2, then to 3.
4. The **fourth** invocation reads 3 at line 408, `[ 3 -ge 3 ]` at 409 is true, and 410 writes
   `data/.state/failed/<m>` with "exhausted 3 attempts". Note the off-by-one: three
   interruptions leave the counter at 3, and the month is not terminal until one more run.
5. Every later run skips at 403-405 and exits 1.

Reproduced with `MAX_ATTEMPTS=3` and `STALE_MIN=45` (so only the dead-pid branch could fire):
attempts 1/3, 2/3, 3/3 across three SIGKILLs with the archive growing 409600, 802816, 1409024
bytes each time, then on run 4 "!!! MONTH 2021-04 IS NOW IN A TERMINAL FAILED STATE: exhausted
3 attempts", exit 1. Clearing both files completed the same month in one go, proving it was
healthy throughout.

**Two consequences beyond the stalled month:**

- **Blast radius is per-run, not per-month.** Line 582 runs `xargs -P "$CONCURRENCY"`
  (5 on the HPC, line 115). One interruption charges an attempt to *every* month in flight.
  Three interrupted sessions can terminally fail five healthy months at once.
- **It strands the archive, contradicting the script's own invariant.** Line 411 returns
  *before* the archive-reclaim block at 456-464, so the `.zst` is never deleted on this path.
  Measured side by side: a genuine preprocess failure reaches 457 and then 462-463 deletes the
  archive, which is the property `hardening.md:118` promises ("its raw archive is deleted,
  reclaiming about 30 GB"). The interruption path leaves it. Worst case, when the
  interruptions land during preprocessing, the archive is already complete and size-verified,
  so a **full ~30GB archive** is stranded on a `/home` that CLAUDE.md:27 puts at ~92% full,
  for a month that will never be retried. The two `mark_failed` call sites disagree about the
  same invariant.

**Fix, and the naive version is a regression.** Do not simply move `attempts_inc` after the
work: charging before it is what stops a poison month (one that OOM-kills the preprocessor,
say) from retry-looping forever, which is the stated goal at `hardening.md:113-117`. Separate
the two meanings:

- line 431 increments `$STATE_DIR/starts/$m` with its own much larger cap (20, say), which
  only catches a genuine crash loop;
- a new `attempts_inc` on the judged-failure path, immediately before the re-read at line 455,
  increments `$STATE_DIR/attempts/$m`, which keeps `MAX_ATTEMPTS=3` meaning "attempts that
  reached a verdict and failed";
- surface both in `show_status`'s active and todo lines (line 496 already prints `attempt=`),
  so a month at 2/3 is visible before it goes terminal. Today the counter is invisible between
  runs: line 496 prints it only for months holding a live claim, so once the reaper releases a
  stranded claim the month prints as a bare name under "not started" at 499 and 518.

Also fix the wrong remedy at line 404, which should name both files exactly as line 282 does.

### G4. The results rsync publishes `.corpus_done` ahead of the games it certifies, and is never retried (blocker, local only)

**Lines 390-392, 444, 447-448, 402 and 482.**

1. A local worker finishes month M. `preprocess_month` passes its kept-count gate against the
   private staging directory (384-388) and publishes: `rm -rf "$outdir"; mv "$staging"
   "$outdir"; touch "$outdir/.corpus_done"` (390-392). That kept-count check is the only one
   in the script and it never runs again.
2. Line 441 clears the trap, line 444 deletes the 28-34GB raw archive (**before** the sync,
   not after), and line 447 rsyncs to the coordination host's **final** path. No `.incoming`
   staging, no `--exclude=.corpus_done`, no `--delay-updates`, no `--partial-dir`, no retry.
   This is the only rsync in the repo.
3. rsync walks a sorted file list, so `.corpus_done` (0x2E) transfers before any `game_*.pkl`
   (0x67), and being a zero-byte `touch` file it lands complete with no partial-file window. I
   verified this rather than assuming it: `rsync -av --dry-run` over a directory of one
   marker, one summary and 60 pickles listed `.corpus_done` first and `.sampling_summary.json`
   second; a real bandwidth-limited transfer SIGKILLed mid-flight left the destination holding
   `.corpus_done`, `.sampling_summary.json` and 6 of 60 pickles. **Caveat: this machine has
   only macOS openrsync (protocol 29); I did not test GNU rsync 3.x.**
   `analysis/corpus-pipeline-fix-plan.md:320` independently asserts the same ordering.
4. The link drops. Two sub-cases, same end state. Either rsync exits nonzero and line 448
   logs "rsync back to HPC FAILED; local copy kept", or, more likely on laptop sleep or a
   route vanishing with no RST, it hangs: the ssh at line 447 carries only `BatchMode=yes` and
   omits the `ConnectTimeout`/`ServerAliveInterval`/`ServerAliveCountMax` that `ctl()` sets at
   173-174. Because the trap was already cleared at 441, an operator Ctrl-C during that hang
   kills the worker with no cleanup at all.
5. The coordination host now holds `.corpus_done` over a lexicographic prefix of the 30,000
   pickles. Nothing counts them.
6. Every later local run short-circuits at line 402 on its own retained `.corpus_done`
   ("already done, skipping"), which returns before `try_claim`, `is_failed` and the attempts
   check, so the upload is never retried and the attempt counter never escalates. `--status`
   reports it done off the first arm of line 482.
7. The archive was deleted at 444, so recovery is a ~30GB redownload nobody knows to start.
   And `.sampling_summary.json`, the second file to land, still reports the full `kept` and
   `scanned`, so the exact audit recommended at
   `analysis/corpus-parallel-pipeline-review.md:479-484` reports the month as healthy while
   its directory holds almost no games.

`analysis/corpus-pipeline-fix-plan.md:319-323` specified this mitigation ("Rsync into
`data/processed_games/<month>.incoming/` and `mv` into place (or `--exclude=.corpus_done` and
copy the marker last)") and it was not implemented. `test_pipeline_recovery.sh:102` runs
`--host hpc` only, so this branch has no coverage.

**Fix.** Three parts, and all three are needed:

```bash
# (a) local: the shared marker stops meaning "built", and starts meaning "delivered".
#     At line 392, on the local host, touch .local_built instead; make 402 and 421 read
#     .local_built locally and .corpus_done everywhere else.

# (b) stage the transfer and promote it on the receiver, with the count checked
#     on the side that matters:
rsync -a --partial --exclude='.corpus_done' \
      -e "ssh -i $HPC_SSH_KEY -o BatchMode=yes -o ConnectTimeout=15 \
          -o ServerAliveInterval=15 -o ServerAliveCountMax=3" \
      "$outdir/" "$HPC_SSH:$REMOTE_ROOT/$OUT_DIR/$m.incoming.$$/" \
  || { log "$m: rsync FAILED; local copy kept at $outdir, will retry next run"; release_claim "$m"; return 1; }
ctl "n=\$(find '$OUT_DIR/$m.incoming.$$' -maxdepth 1 -name 'game_*.pkl' | wc -l); \
     [ \"\$n\" -ge $MAX_GAMES ] || exit 1; \
     rm -rf '$OUT_DIR/$m'; mv '$OUT_DIR/$m.incoming.$$' '$OUT_DIR/$m'; \
     touch '$OUT_DIR/$m/.corpus_done'" \
  || { log "$m: remote promote FAILED (short count); local copy kept"; release_claim "$m"; return 1; }

# (c) delay line 444's `rm -f "$zst"` until after the promote succeeds, so an
#     interrupted month can still be rebuilt without a 30GB redownload.
```

With `.local_built` gating the preprocessing skip, a re-run re-enters the delivery step and
rsync sends only the missing bytes.

### G5. A worker never re-checks that it still owns its claim, so a sleep or outage longer than STALE_MIN puts two workers on one month and the loser publishes over the winner (blocker, local only for the cross-host form)

**Lines 414 (the only `try_claim`), 263, 216, 390-392 and 447.**

The string `owner` appears at exactly three sites in the file: 212 (write), 233 (the reaper
reads it), 488 (`show_status` prints it). **It is never once compared against the running
worker.** `download_month` and `preprocess_month` contain no `ctl()` call at all, so a worker
is structurally incapable of noticing that it lost its claim. The only post-claim recheck,
line 421, tests `.corpus_done` on the local disk, not ownership.

1. A worker claims M, starts its heartbeat (430) and downloads for hours.
2. The lid closes, or the route to the coordination host drops while the internet stays up.
   The heartbeat mtime freezes (sleep) or every touch returns 255 and is discarded (network).
3. At T+45min the claim is stale by line 245 and a reaper on any other run releases it.
4. A second worker claims M. The most reachable route is the documented recovery action
   itself: `hardening.md:28-30` tells the captain to re-run the same command, and that run's
   reaper finds no claim, so the same-host protection at 237-242 (which would have seen the
   first worker's pid alive and skipped) never gets a chance to run.
5. The first worker wakes and resumes mid-instruction. Nothing revalidates. Within
   `HEARTBEAT_S` it is refreshing the **second** worker's heartbeat, because line 263
   refreshes by path. From that point line 245 can never mark M stale again while worker one
   lives, so if worker two dies hard the month is unreapable by the cross-host mechanism.
   (Before worker two recreated the directory, the touches simply fail: `touch` does not
   create a missing parent, verified.)
6. Whichever worker finishes first calls `release_claim` (450), an unconditional `rm -rf`
   (216) with no ownership comparison, deleting the other's live claim and opening M to a
   third claimant.
7. Publication collides with no guard in either direction. Each worker promotes at 390-392 on
   its own disk, and the local one then rsyncs into the coordination host's live
   `data/processed_games/M/` at 447 with no `--delete` and no lock. An HPC promote at 390-391
   `rm -rf`s whatever was rsynced in; a local rsync landing after an HPC promote overwrites
   file by file into a finished month. `hardening.md:99-105` covers the intra-host build only,
   so "two workers on one month is now survivable anyway" does not extend to either publish
   path.

One thing that limits the damage and that I checked before writing it down as worse than it
is: both workers produce the **identical** kept set, because `preprocess_lichess.py` seeds
`random.Random(args.seed)` with the default 42 and the input archive is size-verified
identical, so a merged directory is content-benign. The harm is structural (a `rm -rf` under a
live writer, a deleted live claim, a defeated reaper), not a mixed sample. That equivalence
breaks if the two hosts ever run different `--scanner` values or drifted script copies, which
CLAUDE.md:26 says are hand-synced.

**Fix.** Three surgical changes, none of which needs a redesign:

```bash
# (a) the heartbeat must not refresh a claim it does not own (line 263):
start_heartbeat() {
  local m="$1" wpid=$$
  ( while kill -0 "$wpid" 2>/dev/null; do
      ctl "grep -q '^$HOST_ROLE:$wpid:' '$CLAIM_DIR/$m/owner' 2>/dev/null && \
           touch '$CLAIM_DIR/$m/heartbeat'" >/dev/null 2>&1
      sleep "$HEARTBEAT_S"
    done ) &
  heartbeat_pid=$!
}
# the `kill -0 $wpid` guard also bounds any orphan at one heartbeat interval (G6).

# (b) gate publication on ownership, immediately before line 444, so this runs
#     before the archive is deleted and before anything is promoted or synced:
own=$(ctl "cat '$CLAIM_DIR/$m/owner' 2>/dev/null" 2>/dev/null); orc=$?
if [ "$orc" -ne "$CTL_INFRA_RC" ]; then
  case "$own" in
    "$HOST_ROLE:$$:"*) ;;
    *) log "$m: LOST CLAIM to '${own:-none}' while working; NOT publishing, local copy kept at $outdir"
       return 1 ;;
  esac
fi
# skipping the abort on CTL_INFRA_RC so a partition does not itself discard good work.

# (c) ownership-check release_claim (line 216) so a late finisher cannot delete a live claim:
release_claim() {
  ctl "grep -q '^$HOST_ROLE:$$:' '$CLAIM_DIR/$1/owner' 2>/dev/null && rm -rf '$CLAIM_DIR/$1'" >/dev/null 2>&1
}
```

(b) converts a silent cross-host overwrite into a loud, safe no-op for the cost of one ssh
round trip per month. Note that (c) needs care where `release_claim` is called from the
reaper (240, 247, 252), which by definition is not the owner: that path should keep the
unconditional form, so the ownership-checked variant belongs on a separate helper used by
process_one_month's own call sites.

### G6. The heartbeat subshell outlives an ungracefully killed worker and keeps a dead claim alive forever (medium, reachable today)

**Lines 260-267 and 429.**

Scoped honestly: this does **not** fire for any of the three named scenarios in isolation.
Power off kills the subshell too, sleep freezes it alongside everything else, and a network
partition leaves it alive with every touch failing (which is the correct outcome). It converts
a *targeted single-process* kill, an operator killing a month that looks stuck in `ps`, into a
permanently stranded month.

1. The subshell at 262-265 is a fork of the worker: same process group, same argv, no exec.
2. The worker dies from SIGKILL. The trap at 429 covers INT and TERM only, and there is no
   EXIT and no HUP trap anywhere in the file.
3. The subshell survives, reparented to init. Verified: after `kill -9` of the parent alone,
   ppid 1, original pgid retained, command identical to the parent's, and the heartbeat mtime
   kept advancing while the owner file still named the dead pid.
4. It touches forever, one ssh per minute from the local host. Line 245 can therefore never
   fire for that month, on either host, and line 248 is dead too because the file plainly
   exists. **The cross-host staleness reaper is unconditionally defeated for that month for as
   long as the subshell lives.**
5. Recovery collapses to the same-host dead-pid path at 237-242, which requires a full run on
   the **owning** host (worker mode returns at 553-556 and `--status` at 558-561, both above
   the reaper at 573). If that host stays powered on and idle, the other host can never take
   the month, because its only lever is the heartbeat this orphan keeps fresh.

Two corrections worth recording so nobody over-claims it later. "Forever" is wrong: the hiding
lasts exactly as long as the orphan lives, and the moment it exits the next run on the owning
host reclaims via the dead-pid branch. And the OOM killer is not a realistic trigger, because
it scores by RSS and would select the fat python preprocess child, whose death makes
`preprocess_month` return 1 and releases the claim cleanly at 470.

Invisible to the current suite: `hard_kill` (`test_pipeline_recovery.sh:113-123`) kills every
process matching the sandbox basename, which is present in the subshell's **inherited** argv,
so the existing SIGKILL scenarios kill the heartbeat along with the worker.

**Fix.** The `kill -0 "$wpid"` loop guard in G5(a), plus widening the trap at 429 to
`INT TERM HUP` and adding `trap 'stop_heartbeat' EXIT`. Capture `$$` into `wpid` outside the
subshell for clarity even though bash keeps `$$` stable.

### G7. The reaper runs exactly once per invocation, so a host that dies mid-run is never reclaimed until a human starts another run (medium, reachable today)

**Line 573 (the only call site), 553-556, 558-561, 582-583.**

`grep -n reap_stale_claims` returns exactly two hits: the definition at 227 and the call at
573. Worker mode exits at 555 and `--status` at 560, both above it, and nothing loops around
the xargs at 582-583.

1. A run starts, reaps once, then feeds all 34 months to `xargs -P 5`.
2. Two hours in, the host holding month M dies. Its claim goes stale 45 minutes later.
3. Nothing is looking. Whichever surviving worker reaches M calls `try_claim`, gets rc 1
   (on the HPC `ctl` is `bash -c`, so a colliding `mkdir` gives 1 and the 255 escape at 204 is
   gated on `HOST_ROLE = local` and cannot fire), and line 415 logs "claimed by another
   worker, skipping" and returns 0. M is **consumed from the xargs stream** and never
   revisited. `hardening.md:110` states the underlying property outright: "xargs feeds each
   month exactly once".
4. The run drains, logs "pass complete" at 585, and exits.
5. **It exits 0.** M is counted in `active_c` at 495-496, not `failed_c`, and line 538 returns
   1 only when `failed_c > 0`. So the run reports a clean pass while one month of 34 was never
   touched. Line 18 pre-blesses this ("0 = this host's list is fully done or claimed
   elsewhere"), which is exactly the wording that hides it.

Also worth naming: `--status`, the one command a human would run to check on a weeks-long
run, exits at 560 before the reaper and so cannot reclaim anything either.

**Fix.** Loop instead of exiting after the xargs drains:

```bash
while :; do
  printf '%s\n' $remaining | xargs -P "$CONCURRENCY" -I{} \
    bash "$SELF" --host "$HOST_ROLE" --scanner "$SCANNER" --worker {}
  reap_stale_claims
  prev="$remaining"
  remaining=$(months_not_done_not_failed_not_claimed)   # one ctl round trip per month
  [ -z "$remaining" ] && break
  [ "$remaining" = "$prev" ] && { log "no progress possible this pass; $remaining left claimed elsewhere"; break; }
  sleep "$((STALE_MIN * 60 / 3))"
done
```

The "two consecutive passes claimed nothing new" guard is what keeps the run terminating. A
parent-side supervisor that just calls `reap_stale_claims` every `STALE_MIN/3` is an
equivalent alternative and a smaller diff, but it does not fix the second half of the problem,
which is that a month skipped as "claimed by another worker" is never revisited inside a run.

### G8. The month lists are disjoint by construction, so cross-host takeover is not implemented (medium, requirements conflict)

**Lines 113-129, 229, 554, 582-583.**

`CLAUDE_TASK.md:132-135` asks for "a stale or ownerless claim gets reclaimed by whichever host
is still alive and working". Two independent things block the second half:

1. **Reclaim scope and work scope differ.** The month lists are static literals: `hpc` gets
   the 34-month list at 116-119, `local` gets `MONTHS=""` at 123, and the arg parser at
   100-111 has no `--months` flag. `reap_stale_claims` enumerates the entire claim directory
   at 229 (the only enumeration of `CLAIM_DIR` in the file), while `process_one_month` has
   exactly one call site, 554, fed only by the xargs over `$MONTHS`. So the HPC will reclaim a
   local-owned month and then never process it. It is also invisible to the operator, because
   `show_status` loops `for m in $MONTHS` at 481.
2. **Even with overlapping lists, the reap is single-shot.** See G7.

And a third, which makes it doubly unimplemented today: with `MONTHS=""` the local parent
exits at 564 **before** the reaper at 573, so local currently provides no fallback reaping for
hpc-owned claims either.

**This one is a requirements question before it is a code change, and I am deliberately not
guessing.** Commit `9a9f621`'s wording of the task said the opposite of the current text: "The
HPC side is entirely unaffected either way (it's a separate host with its own list) and keeps
running regardless of whether local is on or off". That paragraph is **not** in the current
`CLAUDE_TASK.md` (grep finds no match) and described the superseded two-script design, so this
is current text versus superseded text rather than two live requirements. Under the
disjoint-list reading, current behaviour is correct and there is no gap. Under
`CLAUDE_TASK.md:132-135`, cross-host takeover is simply not built, and the reaper is not the
missing piece.

**Ask the captain which reading is authoritative before changing behaviour.** If cross-host
takeover is wanted, the smallest honest change is (a) a shared `ALL_MONTHS` list with a
per-host preferred slice, so both hosts may legally claim anything, plus (b) the re-reap loop
from G7. If the disjoint reading is authoritative, no code change is needed, and the writeup
should also note that the local extension months are blocked on `expected_size()` entries
anyway (the table at 76-92 stops at 2024-01 and 2024-08, and `validate_months` at 288-299
refuses any month without one).

### G9. A ctl outage drops months for the rest of the run and still exits 0, despite the log promising a retry (medium, local only)

**Lines 204-206, 416, 582-583, 499, 586-587.**

Two cases of the same root cause. The `return 3` at line 416 is correct behaviour and is
correctly logged; what is missing is any retry behind the promise and any signal in the
result.

*Partial outage.* Every worker reaching `try_claim` during the outage gets rc 255, line 416
logs "coordination host unreachable, NOT skipping, will retry later" and returns 3. Nothing
retries later: xargs feeds each month once and there is no requeue anywhere. The months land
in the plain todo bucket at 499, indistinguishable from months that never started, and no
marker is written under `$STATE_DIR` (correctly so: the drop happens at 416, before
`attempts_inc` at 431, so an infrastructure fault does not burn one of `MAX_ATTEMPTS`).

The drain is fast, not slow, which is the part worth correcting against intuition. Each worker
makes three ctl round trips before it can be dropped (`is_failed` 403, `attempts_get` 408,
`try_claim` 414), and ssh carries `ConnectTimeout=15` (line 173), so against a blackholed
route that is roughly 45s per month per slot with the freed xargs slot refilling immediately.
At the local `CONCURRENCY=3` (line 122) that is about one month burned every 15 seconds, some
40 months in a ten-minute outage, more than the whole list. If ssh fails fast instead
(connection refused, DNS failure) the entire remaining list drains in under a second.
Reproduced: with a 2s stub timeout, 9 months drained in 78s at concurrency 3.

*Total outage.* Line 566's `ctl mkdir` fails silently, line 229 makes the reaper return early,
every month logs the same line, and then line 585 logs "pass complete", 586 runs `show_status`
and 587 exits with its status, which is 0 because `failed_c` is 0. Reproduced: exit 0,
"pass complete", "done 0/2 in progress 0 not started 2 FAILED 0". That contradicts the
script's own contract at lines 18-19 ("0 = this host's list is fully done or claimed
elsewhere"): the list is neither. It is the same class as F5 in
`corpus-parallel-pipeline-review.md:451`, fixed for FAILED months (`hardening.md:155`) and not
for "nothing could be attempted".

Scoped honestly: a human watching the terminal is **not** deceived. They get one explicit
"coordination host unreachable" line per month plus "not started N". Only an exit-code-driven
consumer is, and I grepped the repo: no cron entry, no systemd unit, no wrapper invokes this
script, so that consumer is a plausible future exposure rather than a present one. The
contract violation stands on its own.

**Fix.** (a) On the return-3 path, `touch "$LOG_DIR/.unreachable_$m"`; after the xargs pass,
if any exist, log "coordination host was unreachable for N months, nothing was processed" and
`exit 4` so an automated caller cannot read it as success (clear the markers at the start of
each pass). (b) Make "will retry later" true by retrying `try_claim` about 5 times with a 60s
backoff in-process, and feeding anything still failing into the G7 re-reap loop. (c) Correct
the message at 416 if (b) is not taken.

### G10. A reused PID makes the same-host branch hide a claim from the stale-heartbeat fallback (medium, reachable today)

**Line 242's unconditional `continue`, which skips lines 245 and 248.**

1. A worker claims M and writes owner `hpc:4711:<epoch>` at line 212.
2. Power cut. The claim is stranded with a heartbeat that stops advancing.
3. Reboot. PIDs restart from low numbers and some unrelated process now holds 4711. The
   failure additionally requires that process to be **signalable by the same uid**: I verified
   `kill -0` returns 1 for EPERM exactly as for ESRCH, so a root-owned or other-user squatter
   takes the reclaim branch and self-heals. On a shared HPC most low pids handed out during
   boot are root-owned, which narrows this considerably.
4. Line 237 matches, line 238's `kill -0` succeeds, nothing is reclaimed, and line 242's
   `continue` skips the stale-heartbeat check at 245 entirely. Reproduced with a heartbeat
   aged 180 minutes against `STALE_MIN=45`: `find -mmin +45` **did** print the heartbeat path,
   `kill -0` returned 0, and the claim survived. The control with a dead pid reclaimed
   correctly, and a control with the claim owned by the other host reclaimed via 245,
   confirming it is the branch and not the heartbeat that hides it.
5. The month is hidden from this host's reaper for as long as that process lives. Not
   "forever": the moment it exits, the next run reclaims via 238.
6. It is at least visible. `show_status` renders it from the owner timestamp at 488-496 as
   "on=hpc running=180m", so a human reading `--status` on the owning host can catch it.
   But the run exits 0 (it counts in `active_c` at 495) and it never reaches `attempts_inc`,
   so `MAX_ATTEMPTS` can never promote it to FAILED. Recovery is a manual
   `rm -rf data/.claims/<month>`.

**Fix.** Do not let a live-looking PID veto the heartbeat check. A pid that is alive while its
heartbeat is 45+ minutes old is a reused pid, not our worker:

```bash
    if [ "$ohost" = "$HOST_ROLE" ] && [ -n "$opid" ] && ! kill -0 "$opid" 2>/dev/null; then
      log "reclaiming $m: owned by this host, pid $opid is gone"
      release_claim "$m"; reclaimed=$((reclaimed + 1)); continue
    fi
    # otherwise fall through to the heartbeat rules below
```

Stronger version, if the extra field is acceptable: make the owner record identity that
survives pid reuse, `host:pid:starttime:epoch` with `starttime` from `ps -o lstart= -p $$`,
and treat the claim as dead when the current process with that pid has a different start time.

### G11. Completion is tested on the local filesystem only, so a month already finished by the other host is redone from scratch (medium, local only)

**Lines 402 and 421 versus line 482.**

The script already knows how to ask the right question. Line 482, in `show_status`, is
`[ -f "$OUT_DIR/$m/.corpus_done" ] || [ -n "$(ctl "ls '$OUT_DIR/$m/.corpus_done' ...")" ]`.
The two checks that gate real work, 402 and 421, are the local-filesystem half only. On
`--host local`, `ctl` reaches the coordination host over ssh, but neither work gate uses it.

Reproduced in a two-host sandbox (two script copies, separate roots, a stub ssh executing
against the "HPC" root): run 1 on `--host hpc` completed 2021-04 and released the claim; run 2
on `--host local` for the same month logged "claimed (attempt 2/3), starting", re-downloaded
the archive and invoked the preprocessor a second time. The decisive instrument: the local run
issued 10 ctl commands over the stub ssh and `grep -c corpus_done` on that ledger is **0**. It
never once asked whether the month was already built.

**Worse variant, also reproduced.** The attempts counter is never reset on success. If the HPC
needed all 3 attempts to finish the month, the local run reaches line 409 with `attempts = 3`
and calls `mark_failed` at 410 on a **completed** month, writing
`data/.state/failed/<m>` and logging "exhausted 3 attempts". The same run then printed
"done 1/1 ... FAILED 0", because line 482 asks the coordination host while line 402 asks only
the local disk. One process, two answers, eighty lines apart.

Precondition, stated plainly: this needs a month in both hosts' lists, which is not the shipped
default and is disjoint under `hardening.md:260-261`. Nothing enforces the disjointness though
(`validate_months` at 288-299 checks only `PROTECTED_MONTHS` and `expected_size`, and every HPC
month already has a size entry), and `CLAUDE_TASK.md:148-152` invites exactly this arrangement.

**Fix.** Reuse the `show_status` form at both gates, local test first so the common case costs
no ssh round trip, and short-circuit **before** the attempts check at 409 so a month completed
elsewhere can never be counted as an exhausted retry:

```bash
  if [ -f "$outdir/.corpus_done" ] || \
     [ -n "$(ctl "ls '$OUT_DIR/$m/.corpus_done' 2>/dev/null" 2>/dev/null)" ]; then
    log "$m: already done (here or on the coordination host), skipping"; return 0
  fi
```

Note this interacts with G4: once the receiver only gets `.corpus_done` after a verified
promote, the remote marker becomes trustworthy enough to gate work on. Do G4 first.

### G12. Abandoned `<month>.staging.<pid>` directories are never cleaned up (low)

**Lines 358-359.**

A SIGKILLed worker's staging directory survives, which is the intended isolation and is what
`test_pipeline_recovery.sh:180-182` observes. The next run computes a **new** staging path from
its own `$$`, so line 359's `rm -rf "$staging"` removes only its own name. An exhaustive
enumeration of removals in the file (216, 323, 340, 359, 381, 387, 390, 444, 463) confirms no
glob over `*.staging.*` exists anywhere, and line 390's `rm -rf "$outdir"` deletes
`data/processed_games/2021-04` without touching `2021-04.staging.NNNNN`. Verified by running
it: two crashed runs left two abandoned directories side by side, and a fully successful
re-run of the same month wrote the real output and left both in place.

Two corrections to how this is usually stated. It is not SIGKILL-only: the worker trap at 429
contains no staging cleanup, so an ordinary Ctrl-C leaks one directory per in-flight worker, up
to `CONCURRENCY` at once. And the bound "an empty directory unless the crash lands in the short
write phase" holds for the default Python path (`preprocess_lichess.py` writes pkls only after
the scan completes) but **not** necessarily for `--scanner rust`, where
`analysis/scripts/preprocess_onepass.py:114-115` places `.selected.pgn.tmp` and
`.clkscan_stats.tmp` inside `--output-dir`, that is inside the staging directory, and unlinks
them only on success. NOT VERIFIED: whether clkscan streams that file during pass 1 or writes
it at the end, because reading `analysis/scanner-rs/**` was out of scope for this task.

Downstream risk, reproduced in shape but not in the consumer: `ls data/processed_games/*/game_*.pkl`
does pick up the abandoned directory's partial files alongside the real ones, and the basenames
collide (the hazard CLAUDE.md:35 documents). No flatten script exists in this repo (grep finds
no implementation, only prose at `hpc-operations-guide.md:256-258` and CLAUDE.md:35), so I
cannot say how the real flatten enumerates months. Note that a `{month}_`-prefix flatten would
derive `2021-04.staging.67379_` and therefore silently **add** games from an aborted attempt to
the training set rather than overwrite good ones, which is worse than an overwrite because
nothing would look wrong.

**Fix.** Prune after the reaper at line 573, for months not currently claimed:

```bash
for s in "$OUT_DIR"/*.staging.*; do
  [ -d "$s" ] || continue
  sm=$(basename "$s"); sm="${sm%%.staging.*}"
  [ -n "$(ctl "ls -d '$CLAIM_DIR/$sm' 2>/dev/null" 2>/dev/null)" ] && continue
  log "removing abandoned staging dir $s"; rm -rf "$s"
done
```

---

## 5. Considered and ruled out

Three candidate findings did not survive verification. They are recorded with the reason so
nobody re-derives them.

**The live-pid `continue` strands a month on any normal restart (withdrawn; the corrected form
is G10).** The first framing said a reused pid keeps a claim alive across an ordinary
power-cycle restart "within 45 minutes, which is the normal case". Inside that window the
`continue` is **inert**: I deleted line 242 from an extracted copy and re-ran with a heartbeat
backdated 10 minutes and a live pid, and the month was still not reclaimed, because
`find -mmin +45` matches nothing on a 10-minute-old heartbeat and the heartbeat file plainly
exists so line 248 is false. Neither bypassed rule would have fired anyway. The `continue`
only becomes causal **after** the 45-minute mark, which contradicts the premise the sequence
was built on. G10 is the version that survives.

**PID reuse makes the reaper skip both mechanisms and strands the month until a human deletes
the claim (withdrawn in that form; G10 is narrower).** Two specific errors. First, the
"stranded until a human intervenes" conclusion is false: `reap_stale_claims` iterates
`ls -1 '$CLAIM_DIR'` at line 229, **not** `$MONTHS`, so a claim for a month outside the running
host's list is still reclaimed. I reproduced this: a claim for `2099-01` owned by `local` with a
stale heartbeat was reclaimed by a plain `--host hpc` run. Second, the probability argument was
unsupported: the relevant number is not live-processes over pid-space but the chance that the
stored pid has been reallocated **to a same-uid process** by the time the reaper runs, which is
exactly zero whenever the stored pid exceeds the post-boot allocation high-water mark, and I
could not determine the local host's OS or `pid_max` from this repo.

**`mv` of the staging directory into a concurrently recreated output directory nests the whole
month one level deep and then marks it done (withdrawn).** The underlying `mv` hazard is real
and I reproduced it (`mv M.staging.123 M` with `M` existing yields `M/M.staging.123/...`), but
the alleged sequence does not hold. The nesting window is only the gap between `rm -rf`'s final
`rmdir(2)` at line 390 and `mv`'s `stat(2)` at line 391, two adjacent commands: I measured a
full `mkdir; rm -rf; mv` iteration at 6.8ms over 500 iterations, so the window itself is a low
single-digit millisecond fraction of that. Every other interleaving of a multi-minute 6GB rsync
gives a **different** outcome: an rsync `mkdir` landing before 390 gets deleted mid-transfer and
`mv` then renames cleanly, and one landing after 391 merges into a finished month. And the
stated end state is wrong for the sub-case the sequence itself posits: if rsync creates `M` in
the window and runs to completion, it writes all 30,000 pickles plus the marker, so the flatten
glob sees a complete valid month and the real damage is roughly 6GB of duplicate pickles
stranded in `M/M.staging.<pid>/`, a disk-hygiene bug rather than the corpus-loss bug claimed.
The genuine "marker over a near-empty month" hazard is G4's rsync ordering, which needs no `mv`
race at all.

Two things also checked and found **not** to be defects, worth naming because they look like
they should be:

- **The cross-host pid collision** the task asks about at `CLAUDE_TASK.md:145-147`. Line 237
  gates the `kill -0` on same-host ownership and line 212 writes the pid on the host that owns
  it, so a dead local pid 12345 and a live HPC pid 12345 can never be confused, in either
  direction.
- **A stray heartbeat landing after `stop_heartbeat`.** `stop_heartbeat` (268-271) kills the
  subshell but not an ssh it already has in flight, so a `touch` can land after
  `release_claim`. If another host claimed that month in the gap, the stray touch creates a
  heartbeat inside a claim it does not own, converting an instant no-heartbeat reclaim into a
  45-minute wait. It requires a several-way timing coincidence and the effect is bounded to one
  extra grace period, so it is named here and left alone.

---

## 6. Findings, ordered by severity

| # | Severity | Reachable today? | Finding | Line(s) |
|---|---|---|---|---|
| G3 | **Blocker** | **Yes (HPC)** | An interruption burns a retry, so three power cuts or three Ctrl-Cs put a healthy month into terminal FAILED and strand its ~30GB archive. Charges every in-flight month at once. | 431, 408-412, 456-464 |
| G1 | **Blocker** | No (local) | The reaper reads one failed ssh as "no heartbeat at all" and deletes a live claim with zero grace. | 248, 251-252 (rule at 179-182) |
| G4 | **Blocker** | No (local) | An interrupted results rsync publishes `.corpus_done` ahead of the pickles and is never retried; the archive was already deleted. | 390-392, 444, 447-448, 402, 482 |
| G5 | **Blocker** | No (local) | A worker never re-checks ownership, so a sleep or outage past STALE_MIN gives two workers one month, and the loser publishes over the winner. | 414, 263, 216, 390-392, 447 |
| G2 | **High** | **Yes (HPC)** | The ownerless-claim branch applies zero grace, so it can delete a claim created milliseconds earlier. Same fix as G1. | 202 vs 212, 248-252 |
| G7 | **Medium** | **Yes (HPC)** | The reaper runs once per invocation and xargs feeds each month once, so a mid-run strand waits for a human. The run still exits 0. | 573, 415, 582-587 |
| G8 | **Medium** | **Yes (HPC)** | Month lists are disjoint literals, so a reclaimed month is released and never processed. Cross-host takeover is not implemented. Requirements conflict; ask before changing. | 113-129, 229, 554, 582-583 |
| G10 | **Medium** | **Yes (HPC)** | An unconditional `continue` lets a reused same-uid PID hide a claim from the stale-heartbeat fallback. | 242 (skips 245, 248) |
| G6 | **Medium** | **Yes (HPC)** | The heartbeat subshell outlives a pid-targeted SIGKILL and defeats the cross-host reaper for that month. | 260-267, 429 |
| G9 | **Medium** | No (local) | A ctl outage drops every month it touches for the rest of the run and still exits 0, while the log promises a retry. | 204-206, 416, 582-583, 499, 586-587 |
| G11 | **Medium** | No (local) | Completion is tested on the local disk only, so a month finished by the other host is redone, or wrongly marked FAILED. | 402, 421 vs 482 |
| G12 | **Low** | **Yes (HPC)** | Abandoned `<month>.staging.<pid>` directories are never pruned by any later run. | 358-359 |

**What to fix first.**

1. **G3**, before the captain's next session. It is reachable today, it is the one the captain
   will actually hit, and it turns a good month into a manual-intervention month.
2. **G1 and G2 together**, before the local month list is populated. They are one edit.
3. **G4 and G5**, also before local goes live, in that order. G4 is what silently corrupts the
   corpus; G5's publish gate depends on the claim being trustworthy.
4. **G8** is a question for the captain, not a patch. Resolve it before building anything on
   top of the reaper.
5. The rest can follow.

---

## 7. Test assertions to add to `analysis/scripts/test_pipeline_recovery.sh`

The existing suite is 22 assertions and passes today (I ran it unmodified: 22 passed, 0
failed). Every gap above is a gap in a **green** suite, which is the argument for these
additions. Three structural reasons the current suite cannot see any of this:

- **Every scenario runs `--host hpc`** (line 102), where `ctl` is `bash -c` and can never
  return 255. The entire SSH half of the design, `CTL_INFRA_RC`, the reaper's ssh probes and
  the rsync delivery path, has zero coverage.
- **`reset_month` (124-126) wipes `data/.state/attempts/2021-04` before every scenario**, and
  scenarios 1 and 2 each perform exactly **one** `hard_kill` (148, 175) against
  `MAX_ATTEMPTS=2` (line 87). No scenario accumulates a counter across crashes, so the passing
  month in scenario 1 ends at 2 of 2, one kill short of demonstrating G3.
- **The stale-heartbeat path (line 245) is never executed.** Every kill is same-host, so the
  dead-pid branch at 238 always fires first, and no test ever plants a claim owned by the other
  host. `STALE_MIN=1` is configured at line 85 and never waited on.

### 7.0 Harness extensions these need

Three capabilities the current harness lacks. Add once, near the sandbox setup.

```bash
# ---- a second "host": the coordination host gets its own root, so a local-mode
# run is genuinely cross-machine rather than copying a directory onto itself.
mkdir -p "$SB/remote/data/processed_games" "$SB/bin"

# ---- stand-in for ssh(1). Runs the remote command against $SB/remote (which is
# what REMOTE_ROOT points at), and exits 255 -- ssh's OWN transport-failure code,
# the distinction the script documents at lines 179-182 -- for any command
# matching $SB_SSH_FAIL. That switch is the only way to exercise CTL_INFRA_RC.
cat > "$SB/bin/ssh" <<'SSH'
#!/usr/bin/env bash
# ctl() calls us as: ssh -i KEY -o .. -o .. -o .. -o .. HOST "cd ROOT && CMD"
while [ $# -gt 1 ]; do case "$1" in -i|-o) shift 2 ;; *) shift ;; esac; done
cmd="$1"
if [ -n "${SB_SSH_FAIL:-}" ]; then
  case "$cmd" in *"$SB_SSH_FAIL"*) exit 255 ;; esac
fi
bash -c "$cmd"
SSH
chmod +x "$SB/bin/ssh"

cat >> "$SB/testconf.sh" <<CONF
HPC_SSH="sandbox"
HPC_SSH_KEY="/dev/null"
REMOTE_ROOT="$SB/remote"
CONF

run_pipeline_local() {
  ( cd "$SB" && PATH="$SB/bin:$PATH" bash "$SB/corpus_stream_parallel.sh" --host local "$@" )
}

# ---- portable "N minutes ago" for backdating, in the same style as the harness's
# existing `stat -f%z || stat -c%s` pairs.
ago_stamp() { date -v-"$1"M '+%Y%m%d%H%M' 2>/dev/null || date -d "$1 minutes ago" '+%Y%m%d%H%M'; }
mtime_of()  { stat -f%m "$1" 2>/dev/null || stat -c%Y "$1" 2>/dev/null || echo 0; }

# ---- kill ONLY the worker shell, by pid. `hard_kill` cannot be used for G6:
# it matches the sandbox basename, which is present in the heartbeat subshell's
# INHERITED argv, so it kills the subshell too and hides the orphan entirely.
# The worker is the matching process that is the PARENT of another match.
worker_pid_of() {
  local p q pp
  for p in $(pgrep -f -- "--worker $1" 2>/dev/null); do
    for q in $(pgrep -f -- "--worker $1" 2>/dev/null); do
      pp=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' ')
      [ "$pp" = "$p" ] && echo "$p" && return 0
    done
  done
  return 1
}
```

Note that `testconf.sh` is sourced at script lines 140-143, **after** the host `case` at
113-129, so setting `MONTHS` there overrides local's empty list. That is what makes
`--host local` drivable at all; scenario 5 already relies on the same property (line 228).

### 7.1 G3: three interruptions must not fail a healthy month

```bash
head_ "Scenario 6: three interruptions of a HEALTHY month must not make it FAILED (G3)"
reset_month
sed -i.bak 's/^MAX_ATTEMPTS=.*/MAX_ATTEMPTS=3/' "$SB/testconf.sh"
cut=0
while [ "$cut" -lt 3 ]; do
  ( run_pipeline >"$SB/g3_cut$cut.log" 2>&1 ) &
  # gate on the claim being WON, not on elapsed time: killing before line 431
  # proves nothing, and that is exactly why the existing scenarios miss this.
  for _ in $(seq 1 120); do
    grep -q ": claimed (attempt" "$SB/g3_cut$cut.log" 2>/dev/null && break
    sleep 0.25
  done
  sleep 1
  hard_kill
  cut=$((cut + 1))
done
att=$(cat "$SB/data/.state/attempts/2021-04" 2>/dev/null || echo 0)

run_pipeline >"$SB/g3_final.log" 2>&1
[ -f "$SB/data/.state/failed/2021-04" ] \
  && bad "3 interruptions terminally FAILED a healthy month (attempts=$att, reason: $(cat "$SB/data/.state/failed/2021-04"))" \
  || ok "interruptions did not consume the failure budget (attempts=$att after 3 cuts)"
[ -f "$SB/data/processed_games/2021-04/.corpus_done" ] \
  && ok "month still completed on the run after the third interruption" \
  || bad "month did not complete after three interruptions"
# the archive-reclaim invariant (hardening.md:118) must hold on BOTH mark_failed paths
if [ -f "$SB/data/.state/failed/2021-04" ] && ls "$SB/data/raw_zst"/*.zst >/dev/null 2>&1; then
  bad "a FAILED month left its raw archive on disk (line 411 returns before 456-464)"
else
  ok "no FAILED month is holding a raw archive"
fi
sed -i.bak 's/^MAX_ATTEMPTS=.*/MAX_ATTEMPTS=2/' "$SB/testconf.sh"
```

### 7.2 G1: an SSH blip must never read as "no heartbeat"

```bash
head_ "Scenario 7: an SSH failure must not be read as a missing heartbeat (G1)"
reset_month
mkdir -p "$SB/remote/data/.claims/2021-04"
printf '%s' "hpc:99999:$(date +%s)" > "$SB/remote/data/.claims/2021-04/owner"
touch "$SB/remote/data/.claims/2021-04/heartbeat"      # LIVE: zero seconds old
# fail exactly ONE probe, the `ls` at line 248. One blip is sufficient; the
# original three-failure story is over-specified.
SB_SSH_FAIL="ls 'data/.claims/2021-04/heartbeat'" run_pipeline_local >"$SB/g1.log" 2>&1
[ -d "$SB/remote/data/.claims/2021-04" ] \
  && ok "live claim survived an SSH failure on the heartbeat probe" \
  || bad "reaper DELETED a live claim because one ctl() returned 255"
grep -q "claim has no heartbeat at all" "$SB/g1.log" \
  && bad "reaper reported 'no heartbeat' for a heartbeat it never managed to read" \
  || ok "an unreachable coordination host was not mistaken for a missing heartbeat"
# and the reaper must still work when the link is healthy: a genuinely stale
# claim is still reclaimed, so the fix cannot be "never reclaim anything".
touch -t "$(ago_stamp 120)" "$SB/remote/data/.claims/2021-04/heartbeat"
run_pipeline_local >"$SB/g1b.log" 2>&1
grep -q "reclaiming 2021-04" "$SB/g1b.log" \
  && ok "a genuinely stale heartbeat is still reclaimed over a healthy link" \
  || bad "the 255 guard broke the stale-heartbeat reaper"
```

### 7.3 G2: a brand-new ownerless claim must get the grace period

```bash
head_ "Scenario 8: an ownerless claim gets the same grace as a stale one (G2)"
reset_month
mkdir -p "$SB/data/.claims/2021-04"          # exactly the line 202-to-212 window
run_pipeline >"$SB/g2.log" 2>&1
[ -d "$SB/data/.claims/2021-04" ] \
  && ok "a seconds-old ownerless claim was NOT reaped (a live worker's claim survives)" \
  || bad "ownerless claim reaped with zero grace; this destroys a live worker's claim"
reset_month
mkdir -p "$SB/data/.claims/2021-04"
touch -t "$(ago_stamp 120)" "$SB/data/.claims/2021-04"   # backdate the DIRECTORY
run_pipeline >"$SB/g2b.log" 2>&1
grep -q "reclaiming 2021-04" "$SB/g2b.log" \
  && ok "an ownerless claim older than STALE_MIN is still reclaimed" \
  || bad "the grace period disabled the ownerless-claim reaper entirely"
```

### 7.4 G4: an interrupted sync must never publish a marker over a partial month

```bash
head_ "Scenario 9: an interrupted results sync leaves no .corpus_done (G4)"
reset_month
rm -rf "$SB/remote/data/processed_games/2021-04"
# stand-in for rsync: copies in rsync's own sorted order and dies after N files.
cat > "$SB/bin/rsync" <<'RS'
#!/usr/bin/env bash
src=""; dst=""
for a in "$@"; do
  case "$a" in */) [ -z "$src" ] && src="$a" ;; *:*) dst="${a#*:}" ;; esac
done
mkdir -p "$dst"; n=0
for f in $(cd "$src" && ls -A | sort); do
  [ "$n" -ge "${SB_RSYNC_STOP_AFTER:-999999}" ] && exit 20
  cp -R "$src$f" "$dst/$f"; n=$((n + 1))
done
RS
chmod +x "$SB/bin/rsync"

SB_RSYNC_STOP_AFTER=2 run_pipeline_local >"$SB/g4.log" 2>&1
[ -f "$SB/remote/data/processed_games/2021-04/.corpus_done" ] \
  && bad "interrupted sync published .corpus_done over a partial month" \
  || ok "interrupted sync left no completion marker on the coordination host"
run_pipeline_local >"$SB/g4b.log" 2>&1        # link healthy again, same command
grep -q "already done, skipping" "$SB/g4b.log" \
  && bad "a failed sync is never retried; the month is skipped forever" \
  || ok "a failed sync is retried on the next run"
n=$(ls "$SB/remote/data/processed_games/2021-04"/game_*.pkl 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "5" ] && ok "retry delivered the complete month ($n/5 games)" \
               || bad "retry left the month short ($n/5 games) under a .corpus_done"
rm -f "$SB/bin/rsync"
```

### 7.5 G5: a worker that lost its claim must not publish over the winner

```bash
head_ "Scenario 10: a worker that lost its claim while frozen must not publish (G5)"
reset_month
export FAKE_PREPROCESS_SLEEP=40
( run_pipeline >"$SB/g5.log" 2>&1 ) &
for _ in $(seq 1 120); do [ -f "$SB/data/.claims/2021-04/heartbeat" ] && break; sleep 0.25; done
wpid=$(worker_pid_of 2021-04)
# SIGSTOP is a faithful stand-in for suspend: it freezes the heartbeat subshell too.
kill -STOP "$wpid" 2>/dev/null
pkill -STOP -P "$wpid" 2>/dev/null
touch -t "$(ago_stamp 120)" "$SB/data/.claims/2021-04/heartbeat"
hb_before=$(mtime_of "$SB/data/.claims/2021-04/heartbeat")
# another host reaps the stale claim and takes the month
rm -rf "$SB/data/.claims/2021-04"; mkdir -p "$SB/data/.claims/2021-04"
printf '%s' "otherhost:12345:$(date +%s)" > "$SB/data/.claims/2021-04/owner"
touch "$SB/data/.claims/2021-04/heartbeat"
hb_taken=$(mtime_of "$SB/data/.claims/2021-04/heartbeat")
kill -CONT "$wpid" 2>/dev/null; pkill -CONT -P "$wpid" 2>/dev/null
sleep 5
[ "$(mtime_of "$SB/data/.claims/2021-04/heartbeat")" = "$hb_taken" ] \
  && ok "the resumed worker did not refresh a heartbeat it no longer owns" \
  || bad "the resumed worker is keeping ANOTHER host's claim alive (defeats the stale reaper)"
wait 2>/dev/null
grep -q "LOST CLAIM" "$SB/g5.log" \
  && ok "the resumed worker aborted at the publish gate instead of overwriting" \
  || bad "the resumed worker published with no ownership check"
[ -f "$SB/data/.claims/2021-04/owner" ] \
  && ok "the second worker's claim survived the first worker finishing" \
  || bad "the first worker's release_claim deleted the SECOND worker's live claim"
unset FAKE_PREPROCESS_SLEEP
```

### 7.6 G6: the heartbeat must die with its worker

```bash
head_ "Scenario 11: the heartbeat does not outlive an ungracefully killed worker (G6)"
reset_month
export FAKE_PREPROCESS_SLEEP=40
( run_pipeline >"$SB/g6.log" 2>&1 ) &
for _ in $(seq 1 120); do [ -f "$SB/data/.claims/2021-04/heartbeat" ] && break; sleep 0.25; done
# NOT hard_kill: it matches the sandbox basename, which is in the subshell's
# inherited argv, so it would kill the heartbeat too and hide the whole defect.
kill -9 "$(worker_pid_of 2021-04)" 2>/dev/null
hb1=$(mtime_of "$SB/data/.claims/2021-04/heartbeat")
sleep 4                                        # HEARTBEAT_S=1 in testconf, so 4 beats
hb2=$(mtime_of "$SB/data/.claims/2021-04/heartbeat")
[ "$hb1" = "$hb2" ] \
  && ok "heartbeat stopped when its worker was SIGKILLed" \
  || bad "orphaned heartbeat still refreshing a dead worker's claim ($hb1 -> $hb2)"
hard_kill
unset FAKE_PREPROCESS_SLEEP
```

### 7.7 G7: a claim stranded after the run started is recovered inside the run

```bash
head_ "Scenario 12: a claim stranded mid-run is recovered without a human (G7)"
reset_month
sed -i.bak 's/^MONTHS=.*/MONTHS="2021-04 2021-05"/' "$SB/testconf.sh"
# 2021-05 is claimed by a dead worker on another host, with a stale heartbeat,
# and the claim is planted BEFORE the run so the run's own startup reap is the
# only thing that could catch it today. The assertion is that the run finishes
# the month anyway, which requires a SECOND reap after the first pass drains.
mkdir -p "$SB/data/.claims/2021-05"
printf '%s' "otherhost:4242:$(date +%s)" > "$SB/data/.claims/2021-05/owner"
touch "$SB/data/.claims/2021-05/heartbeat"     # fresh at start, so the first reap skips it
( sleep 3; touch -t "$(ago_stamp 120)" "$SB/data/.claims/2021-05/heartbeat" ) &
run_pipeline >"$SB/g7.log" 2>&1; rc=$?
[ -f "$SB/data/processed_games/2021-05/.corpus_done" ] \
  && ok "a month stranded after the startup reap was still completed in the same run" \
  || bad "a month stranded mid-run waits for a human (single-shot reaper at line 573)"
[ "$rc" -eq 0 ] && [ ! -f "$SB/data/processed_games/2021-05/.corpus_done" ] \
  && bad "the run exited 0 while a month in its own list was never processed" \
  || ok "exit status reflects whether the list was actually finished"
sed -i.bak 's/^MONTHS=.*/MONTHS="2021-04"/' "$SB/testconf.sh"
```

### 7.8 G9: a run that could do nothing must not report success

```bash
head_ "Scenario 13: an unreachable coordination host is not reported as success (G9)"
reset_month
SB_SSH_FAIL="data" run_pipeline_local >"$SB/g9.log" 2>&1; rc=$?
[ "$rc" -ne 0 ] \
  && ok "a run that processed nothing exited nonzero ($rc)" \
  || bad "a run that processed nothing exited 0; the header at line 18 says 0 means done"
grep -q "coordination host unreachable" "$SB/g9.log" \
  && ok "the unreachable host is named per month in the log" \
  || bad "the outage was not reported at all"
grep -qE "unreachable for [0-9]+ month" "$SB/g9.log" \
  && ok "the run summarises how many months it could not attempt" \
  || bad "dropped months are indistinguishable from months that never started"
```

### 7.9 G10: a reused PID must not hide a stale claim

```bash
head_ "Scenario 14: a live but unrelated PID must not veto the heartbeat check (G10)"
reset_month
sleep 600 & squat=$!
mkdir -p "$SB/data/.claims/2021-04"
printf '%s' "hpc:$squat:$(( $(date +%s) - 10800 ))" > "$SB/data/.claims/2021-04/owner"
touch -t "$(ago_stamp 120)" "$SB/data/.claims/2021-04/heartbeat"
run_pipeline >"$SB/g10.log" 2>&1
kill "$squat" 2>/dev/null
grep -q "reclaiming 2021-04" "$SB/g10.log" \
  && ok "a 120m-stale claim is reclaimed even though its recorded PID is alive (reuse)" \
  || bad "the live-PID branch at line 242 hid a claim 120m past STALE_MIN"
```

### 7.10 G11: a month finished on the coordination host is not redone

```bash
head_ "Scenario 15: a month already done on the coordination host is skipped (G11)"
reset_month
mkdir -p "$SB/remote/data/processed_games/2021-04"
touch "$SB/remote/data/processed_games/2021-04/.corpus_done"
# the attempts counter is never reset on success, so a month the HPC needed all
# 3 attempts to finish arrives at line 409 already at the cap.
mkdir -p "$SB/remote/data/.state/attempts"
printf '3' > "$SB/remote/data/.state/attempts/2021-04"
run_pipeline_local >"$SB/g11.log" 2>&1
grep -q ": claimed (attempt" "$SB/g11.log" \
  && bad "re-downloaded a month the coordination host had already finished" \
  || ok "a month complete on the coordination host was skipped without re-work"
[ -f "$SB/remote/data/.state/failed/2021-04" ] \
  && bad "a COMPLETED month was marked FAILED because attempts was already at the cap" \
  || ok "a completed month was not marked FAILED by a stale attempts counter"
rm -f "$SB/remote/data/.state/attempts/2021-04"
```

### 7.11 G12: abandoned staging directories are pruned

```bash
head_ "Scenario 16: abandoned staging dirs are pruned by a later run (G12)"
reset_month
mkdir -p "$SB/data/processed_games/2021-04.staging.999999"
touch "$SB/data/processed_games/2021-04.staging.999999/game_0000000.pkl"
run_pipeline >"$SB/g12.log" 2>&1
[ -d "$SB/data/processed_games/2021-04.staging.999999" ] \
  && bad "an abandoned staging dir survived a full successful run of the same month" \
  || ok "abandoned staging dir pruned"
n=$(ls "$SB/data/processed_games"/*/game_*.pkl 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "5" ] \
  && ok "the flatten glob sees exactly the real month's games ($n)" \
  || bad "the flatten glob would pick up $n files, including an aborted attempt's"
```

### 7.12 One assertion in the existing suite that should be tightened

`test_pipeline_recovery.sh:180-182` currently calls `ok()` on **both** branches, so it passes
whether or not a staging directory is left behind and cannot detect either a leak or a
regression in the isolation it is named after. Split it in two:

```bash
staging=$(find "$SB/data/processed_games" -maxdepth 1 -name '2021-04.staging.*' 2>/dev/null | head -1)
[ -n "$staging" ] && ok "half-built output left in a staging dir, not in the real one" \
                  || bad "no staging dir: the isolation this scenario is named for did not engage"
```

with the "gone after the recovery run" half asserted by scenario 16 above.

---

## 8. What to hand firstmate for HPC validation

Everything in this document was verified locally or on a Mac. Named explicitly so nobody reads
it as HPC-validated:

1. **The 255 behaviour itself.** Every claim about ssh returning 255 for its own failures comes
   from the script's comment at 179-182 and from `ssh(1)`, plus a shim that *injects* 255. It
   was never measured against `<hpc-host>`. A real 30-second link drop against the real
   host, with the reaper running, is the check that matters for G1.
2. **rsync ordering on GNU rsync 3.x.** G4's transfer order and its discard-partials behaviour
   were verified on macOS openrsync (protocol 29). The captain's PC most likely runs GNU rsync
   3.x. The property is the same by design and `corpus-pipeline-fix-plan.md:320` asserts it
   independently, but it is untested there.
3. **Two concurrent `curl -C -` writers on one raw path.** The tail of G1 and G2 predicts
   overshoot thrashing into terminal FAILED at 338-339 and 456-464. Not reproduced.
4. **PID reuse rates and `pid_max`** on the HPC and on the captain's machine, which is what
   sets G10's actual likelihood. I could not determine the local machine's OS from this repo.
5. **Whether `clkscan` streams `.selected.pgn.tmp` during pass 1**, which decides whether G12's
   abandoned staging directories are empty or hold real bytes on the `--scanner rust` path.
   Reading `analysis/scanner-rs/**` was out of scope here.
6. **`STALE_MIN=45` against a real month's wall time.** `hardening.md:270-272` already flags it
   as an untuned guess. With G5's publish gate in place the cost of reclaiming too eagerly
   drops sharply, which may be a reason to lower it rather than raise it.

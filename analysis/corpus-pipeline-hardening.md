# Corpus pipeline hardening: status command and crash recovery

Task: `CLAUDE_TASK.md` Part D. Written 2026-08-17. Companion to
`analysis/corpus-parallel-pipeline-review.md`, which is the Part A review of the
script this replaces. Findings referenced as F1, F3 and so on come from that review.

Deliverable: `corpus_stream_parallel.sh` at the repo root, plus
`analysis/scripts/test_pipeline_recovery.sh` which proves the recovery claims rather
than asserting them.

**Not yet deployed.** Everything here was verified locally against a stand-in
Lichess server and a stub preprocessor. It has never run on the HPC. See
"Before deploying" at the end.

---

## The simple guide

Start it, on either machine:

    bash corpus_stream_parallel.sh --host hpc         # on the HPC
    bash corpus_stream_parallel.sh --host local       # on your PC

Check on it, any time, from anywhere:

    bash corpus_stream_parallel.sh --host hpc --status

If your PC turns off, or anything crashes, just run the same start command again.
It works out what was interrupted and picks it up. There is nothing to clean up by
hand and no special flag to remember.

If a month goes FAILED, that one needs you. `--status` names it and says why.

---

## One script instead of two

The old design had `hpc_corpus_stream_parallel.sh` and
`local_corpus_stream_parallel.sh`: the same concurrency-critical logic, twice, kept in
sync by hand. The only real difference is the month list and where the claim
lives. Two copies of tricky bash drift, and drift here means two workers on one month.

Now there is one file with `--host hpc|local`. The month list is a `case`, and every
claim/state mutation goes through one `ctl()` helper that runs the command locally on
the HPC or over SSH from the local machine.

A side effect worth naming: the review could not fully assess the local copy because
that file is not in the repo (F1b stayed uncertain for exactly this reason). One file
removes that blind spot.

## Why the claim needed rebuilding

`mkdir` is atomic, so the old `mkdir data/.claims/<month>/` really did stop two
workers taking the same month. That was never the problem. The problem is that a bare
directory answers only "is this month spoken for", and the pipeline needs to
distinguish three states:

1. a worker is actively processing this month,
2. a worker was SIGKILLed or lost power an hour ago and is never coming back,
3. this month finished successfully.

The old script could not tell these apart, and released the claim on neither success
nor death (F3), so case 2 stranded a month permanently and case 3 left a claim that
looked identical to case 1 forever.

The claim now carries its own liveness:

    data/.claims/<month>/owner       "<host>:<pid>:<epoch>"
    data/.claims/<month>/heartbeat   touched every 60s while work proceeds

and it is released on success, on clean exit, and on SIGINT/SIGTERM via a `trap`.
Completion is recorded separately by `<outdir>/.corpus_done`. So a claim that still
exists means case 1 or case 2, and the heartbeat separates those.

## Detecting a dead worker

Two mechanisms, because neither covers both cases.

**Same host, dead pid (instant).** On startup the script reads each claim's `owner`.
If the claim belongs to this host and `kill -0 <pid>` fails, the worker is gone and
the claim is reclaimed immediately. This is the captain's named scenario: the PC is
switched off mid-month and switched back on, the old pids no longer exist, and the
next run reclaims with no waiting.

**Stale heartbeat (after 45 minutes).** A crash on the *other* host cannot be checked
this way, since neither machine can inspect the other's process table. The reaper
falls back to the heartbeat's mtime, evaluated on the coordination host's own
filesystem, so it needs no clock agreement between the two machines.

A claim with no heartbeat file at all is also reclaimed: that is a worker that died in
the window between `mkdir` and the first touch.

`STALE_MIN=45` is deliberately much longer than a 60s heartbeat. The cost of waiting
is a delayed retry. The cost of reclaiming too eagerly is two workers on one month,
so the asymmetry is on purpose.

### Two workers on one month is now survivable anyway

The heartbeat can produce a false positive: if the local machine's SSH is down for 45
minutes while it is genuinely still working, its claim looks stale. Belt and braces,
preprocessing now builds into `<outdir>.staging.<pid>` and `mv`s the finished
directory into place. Two workers cannot corrupt each other's output, and a crash
mid-preprocess never leaves a half-built directory where the kept-count check would
misread it. The old script's `rm -rf "$outdir"` followed by a slow rebuild was
exposed to both.

## Retries, and knowing when to stop

The old script had zero preprocess retries where the serial version had three (F4),
and `xargs` feeds each month exactly once, so a single transient failure silently
dropped a month for the whole multi-week run.

Now `data/.state/attempts/<month>` counts attempts across runs and hosts. Under
`MAX_ATTEMPTS=3` the month is retried. On exhaustion it goes to
`data/.state/failed/<month>` with a reason, and:

- it is never retried automatically, so a genuinely broken month cannot retry-loop,
- it is reported loudly at the time and by `--status` afterwards,
- its raw archive is deleted, reclaiming about 30 GB (F6b),
- the run's exit status becomes 1 (the old script always exited 0 and always printed
  "all months processed", F5).

Clearing one is a documented one-liner printed with the failure.

## Status in one screen

    $ bash corpus_stream_parallel.sh --host hpc --status

      corpus pipeline status  (hpc, 2026-08-17 20:37:35)
      ------------------------------------------------------------
      done 12/34   in progress 5   not started 16   FAILED 1

      in progress:
        2022-03  on=hpc  running=41m  scanned=1240000  attempt=1
        ...

      FAILED (needs a human, will not retry on its own):
        2021-09 [failed on attempt 3 (see logs/preprocess_2021-09.log)]

      not started: 2022-08 2022-09 ...

      average 2h14m per month observed; rough ETA for the rest: 9h (at concurrency 5)
      scanner: python      logs: logs/preprocess_<month>.log

The ETA is derived from this host's own observed per-month wall time, not a guess. The
per-worker scan rate is read out of the preprocess logs so the operator never needs to
know a log path or a grep incantation, which was the original complaint.

## Other Part A findings addressed

| Finding | What changed |
|---|---|
| F1 truncated download silently accepted | Exact expected-size match is mandatory. `validate_months` refuses at startup any month with no recorded size, restoring the serial version's assertion. The `--scanner rust` path additionally refuses to process a truncated archive outright. |
| F2 `curl` lost `-f` | Restored, so an HTTP error body is never written into the archive. |
| F5 always exits 0 | Exit status now reflects FAILED months. |
| F6 peak staging ~153 GB unchecked | `df` check before each download; refuses to start a month that cannot fit, and says to free space or lower `--concurrency`. |
| F7 `$0` resolved after `cd` | Workers are launched from an absolute `$SELF`. |
| F9 `-C -` resume defeated by `rm -f` | Partial downloads are kept and resumed. Verified by asserting the server actually received a `Range:` request. |
| F10 startup validations dropped | Restored, plus a `PROTECTED_MONTHS` guard so the 2024-02..07 subset can never be clobbered. |
| F12 preprocess exit status discarded | Checked, and a crash is reported as a crash rather than as "0/30000 kept". |
| F13 `cd` lost its guard | Restored. |
| F15 observability | Everything tees to `logs/corpus_parallel.log`; `--status` replaces grepping. |
| F16 orphaned processes | `trap ... kill 0` tears down the worker process group. |
| F18 inert `export -f` | Removed. |
| F19 worker-mode `mkdir` ordering | Workers create their own directories. |
| F20 string compare on sizes | Numeric `-eq`. |

**F14 (editing the deployed script mid-run) is documented, not fixed.** Bash reads a
script incrementally, so an in-place edit during a live run corrupts the parked
parent's byte offset. Editing via atomic rename (`mv`) is safe. The comment in the
script says so at the point where it matters.

### One more, found while testing

The old script's `declare -A EXPECTED_SIZES` requires bash 4+. macOS still ships bash
3.2, where `declare -A` fails and `${EXPECTED_SIZES[2021-04]}` then degrades into an
*arithmetic* subscript: `2021-04` evaluates to `2017`, so every lookup silently
returns the wrong (empty) entry instead of erroring. A silent wrong size on a 30 GB
download is about the worst failure mode available here. The size table is now a
function, which removes the bash 4 requirement entirely.

That also retires the stated reason for the `bash "$0" --worker <month>` pattern. The
review asked whether that re-invocation works (Part A item 3); it does, but its
rationale was that associative arrays cannot be exported to a subshell. With a
function there is nothing to export. The re-invocation is kept, now purely so each
month gets its own process and one month's failure cannot take down its siblings.

## Verification

`analysis/scripts/test_pipeline_recovery.sh` runs the real script against a throttled,
`Range`-aware stand-in for `database.lichess.org`
(`analysis/scripts/slow_file_server.py`) and a stub preprocessor, so the claim,
staleness, resume and retry logic is exercised for real without 30 GB per month.

The kills are `SIGKILL` to every process in the sandbox, so no trap runs and nothing
gets a chance to clean up. That is the point: a graceful Ctrl-C would prove nothing
about a power cut.

    $ bash analysis/scripts/test_pipeline_recovery.sh

    == Scenario 1: SIGKILL mid-DOWNLOAD, then re-run the same command ==
      PASS download interrupted mid-flight at 131072B of 2000000B
      PASS claim survived the kill (correctly stranded, nothing cleaned up)
      PASS month correctly not marked done
      PASS re-run detected and reclaimed the stranded claim
      PASS re-run completed the month with no manual cleanup
      PASS download RESUMED via HTTP Range, partial bytes not discarded

    == Scenario 2: SIGKILL mid-PREPROCESS, then re-run the same command ==
      PASS preprocess reached and running
      PASS claim stranded by the mid-preprocess kill
      PASS half-built output left in a staging dir, not in the real one
      PASS real output dir untouched by the crash
      PASS re-run reclaimed the mid-preprocess claim
      PASS re-run completed the month unaided
      PASS output complete and correct (5/5 games)

    == Scenario 3: a genuinely broken month stops after MAX_ATTEMPTS, loudly ==
      PASS month moved to a terminal FAILED state
      PASS failure is surfaced loudly, not silently
      PASS claim released after terminal failure
      PASS later runs skip the failed month instead of looping on it

    == Scenario 4: --status is readable without knowing any log paths ==
      PASS status reports done/total
      PASS status surfaces the FAILED month

    == Scenario 5: two concurrent runs never process the same month twice ==
      PASS each of the 2 months claimed exactly once across 2 concurrent runs
      PASS 2021-04 completed
      PASS 2021-05 completed

Scenarios 1 and 2 together are the captain's named requirement: a hard kill mid-download
and mid-preprocess, recovered by re-running the identical command with no flags and no
manual cleanup.

Two bugs in the hardened script were found by this suite and fixed, which is the
argument for having written it: the claim's `owner` timestamp was being stored as the
literal text `$(date +%s)` (a quoting error inside the `ctl` command string), and
`show_status` then failed on the non-numeric value.

## The faster scanner (Part D item 3)

Wired in, but **not** as the default. `--scanner rust` uses `clkscan` plus
`analysis/scripts/preprocess_fast.py` when both are present, and falls back to the
Python path with a log line when they are not. `--scanner python` is the default.

The Part C prototype is 161x faster and agrees with the current scanner on 626,765 of
626,765 real games and on a 36-case adversarial suite, and `preprocess_fast.py`
produces content-identical output to `preprocess_lichess.py` on a 60,000-game sample.
That is good evidence, and it is still not evidence gathered on the HPC or on a
complete 30 GB month. Until it is, forcing it on would be exactly the trade the task
warned against, so the flag exists and the default does not move. Conditions for
promoting it are in `analysis/faster-scan-research.md` section 10.

## Before deploying

1. **Nothing here has run on the HPC.** Test on one month, ideally one already done,
   with the output directory pointed somewhere harmless.
2. **Set the local month list.** `MONTHS` for `--host local` is deliberately empty; the
   extension months (2024-09 onward) need their sizes added to `expected_size()` first.
   The script refuses months with no recorded size, so this fails loudly rather than
   downloading something unverified.
3. **Check `HPC_SSH`, `HPC_SSH_KEY` and `REMOTE_ROOT`** at the top of the script.
4. **Decide what happens to the run in flight.** This script's state layout
   (`owner`/`heartbeat` files, `data/.state/`) is not what the currently running
   pipeline uses. Months already carrying `.corpus_done` are recognised and skipped, so
   the safe sequence is: let the current run finish or stop it, clear
   `data/.claims/`, then start this one.
5. **`STALE_MIN=45`, `MAX_ATTEMPTS=3`, `CONCURRENCY=5`** are guesses that look sane, not
   tuned values. The one to think about is `STALE_MIN` against how long a month
   legitimately takes.

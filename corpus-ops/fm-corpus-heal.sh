#!/usr/bin/env bash
# fm-corpus-heal.sh - single idempotent self-healing pass for the corpus pipeline.
#
# Consolidates and extends fm-corpus-zst-cleanup.sh and fm-corpus-autopreprocess.sh
# with the one piece that was still missing: resuming a download that died
# mid-transfer. Designed to be invoked repeatedly and briefly (not as a long-lived
# background loop - those were shown 2026-08-20 to die from the same cause that
# kills everything else in the WSL2 VM). Meant to be driven by a Windows
# Scheduled Task calling into WSL every few minutes, which survives power-state
# events that kill anything living purely inside the VM.
#
# For each month name found in MONTHS below (extend this list as new months get
# assigned), exactly one of five things is true, and this script does the right
# next thing for whichever is true:
#   1. Already fully done (.zst gone, 30000 pkls) -> nothing to do.
#   2. .zst complete, not yet fully preprocessed, nothing running -> launch preprocessing
#      (resuming from scratch if a prior partial pass2 died mid-write - the partial
#      output isn't a valid sample and is safer to redo than to patch).
#   3. .zst complete, preprocessing genuinely running -> leave alone.
#   4. .zst partial/missing, nothing downloading -> (re)launch the download with -C -.
#   5. .zst partial, download genuinely running -> leave alone.
# Plus the original cleanup duty: once a month hits 30000 pkls, delete its .zst.
#
# --- 2026-08-21: why case 2 is no longer unconditional ------------------------
# Case 2 used to mean "no process + not done => it was killed, relaunch it".
# That conflated two very different deaths:
#   (a) the pass was killed externally mid-run (the VM power event) - relaunching
#       is exactly right, and is the entire reason this script exists;
#   (b) preprocess_onepass.py/clkscan ran to completion and *exited with an
#       error* - relaunching is pure waste, because the same input will produce
#       the same error forever.
# On 2025-04 the .zst was truncated, clkscan refused it, and (b) was handled as
# (a): 20 relaunches over ~19.5h, each one a full ~70min scan of a 29GB archive
# that could never succeed. Nothing in $HEALLOG said anything was wrong.
# Preprocessing is now launched through a wrapper that records the real exit
# status to $STATE_DIR/<month>.pp.rc, so a later pass can tell (a) from (b):
#   no rc file            -> killed externally      -> relaunch (old behaviour)
#   rc != 0, truncated    -> bad input archive      -> re-download once, then block
#   rc != 0, anything else-> real tool failure      -> block, flag for a human
#   rc == 0 but short     -> tool disagrees with us -> block, flag for a human
# "block" = write $STATE_DIR/<month>.blocked and stop touching that month until
# a human deletes that file. It is loud in $HEALLOG once, then re-reminds every
# BLOCK_REMIND_MINUTES so a blocked month can't be silently forgotten.
set -u

BASE="/mnt/d/firstmate/projects/cs_thesis_2/local-run"
# D: was hitting 80%+ from the transient ~30GB-per-month raw archives stacking
# up. 2026-08-21: raw downloads move to C: (213GB free vs D:'s ~50GB), while
# PROCESSED_DIR (the permanent ~6.1GB/month output) stays on D: - it's the only
# location anything downstream (training, HPC rsync) needs to know about, so
# splitting by STAGE (raw vs processed) instead of by MONTH avoids needing any
# per-month drive bookkeeping. RAW_DIR_OLD is kept only so a month whose
# archive already exists there (e.g. one mid-download at the time of this
# change) finishes in place instead of being silently orphaned - see the
# zst-path selection in the per-month loop below.
RAW_DIR_OLD="$BASE/data/raw_zst"
RAW_DIR="/mnt/c/Users/bacsa/Downloads/fm-corpus-raw"
# 2026-08-22: decided NOT to migrate existing D: processed output (the
# D:->C: move attempt caused real data loss from an overlap between two
# uncoordinated move mechanisms - see learnings.md). Existing months stay on
# D: untouched forever; only NEW months (never processed before) land on C:.
# Same "prefer wherever it already exists" pattern as RAW_DIR above.
PROCESSED_DIR_OLD="$BASE/data/processed_games"
PROCESSED_DIR="/mnt/c/Users/bacsa/Downloads/fm-corpus-processed"
LOG_DIR="$BASE/logs"
CLKSCAN="$BASE/bin/clkscan"
PYTHON="/mnt/d/firstmate/projects/cs_thesis_2/prototype/.venv/bin/python"
PREPROCESS_SCRIPT="/mnt/d/firstmate/projects/cs_thesis_2/analysis/scripts/preprocess_onepass.py"
HEALLOG="/mnt/d/firstmate/data/corpus-heal.log"

# Every month any worker has ever been assigned to, past or present - extend
# this list by hand when a new month gets assigned to a worker.
#
# 2026-08-22 correction: this list used to cover only 9 scattered months and
# missed a full contiguous block (2024-09..2026-01) that a DIFFERENT, earlier
# pipeline (local-run/corpus_stream_parallel.sh, --host local) had already
# completed before this heal script existed. Verified against real pkl counts
# on disk, not assumed - all months below except 2026-05 are already done.
# Kept in the list anyway so future re-corruption/partial-state on any of them
# gets caught the same way 2024-12 and 2025-04's did.
#
# Before adding a NEW month beyond this range, check it doesn't collide with
# the OTHER, separate claim-locked pipeline: HPC's claimed range is
# 2021-04..2024-01 + 2024-08 (see corpus_stream_parallel.sh's
# `case "$HOST_ROLE"` block), and 2024-02..2024-07 are PROTECTED (already
# built by an earlier 170k-subset run, never touch). ALWAYS verify real pkl
# counts on disk before assuming a month is an open gap - a static month-list
# in a script file is not the same as ground truth.
MONTHS="2019-07 2019-08 2019-09 2019-10 2019-11 2019-12 2020-01 2020-02 2020-03 2020-04 2020-05 2020-06 2020-07 2020-08 2020-09 2020-10 2020-11 2020-12 2021-01 2021-02 2021-03 2021-04 2021-05 2021-06 2021-07 2021-08 2021-09 2021-10 2021-11 2021-12 2022-01 2022-02 2022-03 2022-04 2022-05 2022-06 2022-07 2022-08 2022-09 2022-10 2022-11 2022-12 2023-01 2023-02 2023-03 2023-04 2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 2024-01 2024-08 2024-09 2024-10 2024-11 2024-12 2025-01 2025-02 2025-03 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06 2026-07"
# 2020-01..2020-06 added 2026-08-22: continuing backward, verified available,
# no collision, output lands on C: (new-month rule above).
# 2020-07..2020-08 added 2026-08-22: D: headroom getting tight (~48GB free) -
# smaller 2-month batch this round, not 4. Verified available, no collision.
# 2020-09..2020-12 added 2026-08-22: continuing backward before 2021-01,
# verified available on lichess, no collision (still before HPC's range).
# 2021-01..2021-03 added 2026-08-22: right before HPC's claimed range starts
# (2021-04), verified available on lichess, no collision.
# 2026-08 checked and NOT added: database.lichess.org returns a 146-byte
# placeholder for it (current month, archive not published yet).
#
# 2023-09..2024-01 + 2024-08 added 2026-08-23, deliberate deviation from the
# "never touch HPC's claimed range" rule above, per explicit captain
# instruction: HPC has been fully unreachable since before this script's
# monitor log even starts (2026-08-18, 100% UNREACHABLE ever since, zero
# entries where it was ever seen alive - data/hpc-monitor/monitor.log) with
# no ETA, every month outside its claimed range is already exhausted, and
# the captain explicitly chose to risk possible duplicate compute over
# sitting idle. There is NO telemetry on which of HPC's 35 assigned months
# it actually finished before going dark - captain.md's "35 months claimed"
# note is HPC's own scope assignment, not a completion report. Picked the
# TAIL of HPC's range (2024-08 backward through 2023-09) on the unverified
# guess that if HPC's run got interrupted mid-sweep, later-dispatched months
# are the likeliest to still be unstarted - a heuristic, not a fact. No
# filesystem collision risk either way: HPC writes to its own remote disk,
# entirely separate from this machine's C:/D:. Worst case is wasted
# redundant bandwidth/compute here, not corrupted data. If HPC resurfaces
# and its own claim-lock shows these months already done, that's a resolved
# duplicate, not a conflict - the claim system was never load-bearing for
# this ad-hoc list to begin with (see the top-of-file note on that).
#
# 2023-06..2023-08 added 2026-08-23, continuing backward through the same
# HPC-range deviation above, to keep parallel downloads running instead of
# the queue going idle after 2023-09..2024-08 lands. Kept to a 3-month batch
# this round (not 6): the previous 6-month batch drove C: from 149GB free to
# 1.3GB free and crashed 5 of 6 in-flight preprocessing jobs on
# `OSError: [Errno 5] Input/output error` mid pickle.dump - all recovered
# without data loss (each had already finished pass 1 into
# `.selected.pgn.tmp` before crashing, so pass 2 was rerun standalone from
# that staged file with the now-unneeded 30GB source archive deleted first
# to free space; see local-run/logs/pp_*_pass2resume.log). Smaller batches
# keep the transient per-month ~28-32GB download spike from stacking past
# whatever C: actually has free at launch time - check `df -h /mnt/c` before
# growing this list further, not just before adding a month.
#
# 2021-04..2023-05 added 2026-08-24: fills out the REST of HPC's claimed
# range (see the 2023-09..2024-01+2024-08 comment above for the full
# reasoning - same deviation, same captain instruction, same "no telemetry
# on HPC, worst case is wasted redundant compute not corrupted data"
# argument). This exhausts HPC's entire 35-month range in this list (all of
# 2021-04..2024-01 + 2024-08 now present). Added alongside the self-pacing
# auto-launch logic above specifically so the captain can leave this running
# 10+ hours unattended - the queue won't run dry partway through the night
# even at 4-way concurrency. If it DOES exhaust this too, there is no more
# "safe" (non-PROTECTED, non-currently-in-scope) territory left; the next
# extension would need a fresh captain call, not another silent add.
#
# 2019-07..2019-12 added 2026-08-24 (fresh captain call, per the note above):
# the entire April2021-present window (minus PROTECTED) plus all of 2020 is
# now complete - 73/73 months, ~2.19M games, already past the adviser's ~2M
# target. Captain asked to keep growing the corpus anyway: more data directly
# feeds the pre-registered stage-2 "double the corpus until not overfitting,
# then deepen the CNN" plan (Omori's ruling, chapter3-4-process.md), so this
# isn't idle scope creep. Continuing backward before 2020 since it's the only
# remaining untouched territory (everything else is either done or the
# PROTECTED 2024-02..07 block). Kept to 6 months this round, matching the
# established "check df -h before growing this list further" discipline.

# 2026-08-24: self-originating new downloads, so this can run genuinely
# unattended overnight (captain going to sleep, wants 10+ hours of
# self-paced progress with no manual curl launches). Until now this script
# only *resumed* an already-started month (case 4/5); a month with no .zst
# at all was explicitly "not this script's job". That meant the MONTHS list
# alone was never enough - someone had to manually curl each new month by
# hand. Now a month with no archive yet gets an ORIGINATED download IF
# there's concurrency and disk headroom, gated by real numbers from what
# actually broke earlier tonight: 6 simultaneous fresh curls got one 429'd
# by Lichess (5 simultaneous did not), and 6 simultaneous ~30GB downloads
# drove C: from 149GB free to 1.3GB free and crashed 5 of 6 in-flight
# preprocessing jobs on an I/O error mid pickle.dump (recovered without data
# loss, but avoid repeating it unattended). AUTOLAUNCH_CONCURRENCY caps
# active curl+preprocess processes; AUTOLAUNCH_MIN_FREE_GB blocks
# originating a new ~30GB transient download without comfortable margin.
#
# ARCHIVE_DIR is read-only here (recognizes a month archived by the separate
# one-off compression pass as done, so it isn't re-downloaded) - this script
# does NOT launch archiving itself. That was tried the same night and
# reverted: the captain's actual ask was a one-time compress of the already-
# complete backlog, not a standing pipeline behavior, and the auto-archive
# launch was also the direct cause of two real bugs (unbounded concurrent
# launches racing an already-running manual batch job, and false
# "unarchived" positives against split-directory months) - see git history
# on this file for the full incident if reintroducing this idea later.
ARCHIVE_DIR="$BASE/data/processed_games_archive"
AUTOLAUNCH_CONCURRENCY=4
AUTOLAUNCH_MIN_FREE_GB=40

# Small persistent per-month state (exit codes, block markers, re-download
# counters). Deliberately NOT in $LOG_DIR: nothing here is a log, and the log
# paths are part of this script's contract with the humans reading them.
STATE_DIR="$BASE/state/heal"

# How many times a month may have its archive automatically deleted and
# re-downloaded after clkscan reports a truncated/corrupt file. Kept at 1 on
# purpose: one fresh 29GB download covers the likely cause (a bad resume seam
# or an unflushed write when the VM was killed). A second identical failure
# means the problem isn't transient, and spending another 29GB + ~70min scan to
# learn that again is the same waste this script is supposed to prevent.
MAX_AUTO_REDOWNLOAD=1

# 2026-08-22: a process that is genuinely RUNNING but silently STALLED (a
# curl transfer wedged with no --speed-limit, or clkscan hung) was previously
# invisible to this script forever: "pgrep finds it -> leave it alone" never
# distinguished a live worker from a wedged one. Deliberately deterministic
# bash, not a monitoring LLM crewmate - captain's own agy-hpc-monitor.sh
# incident (2026-08-18, see learnings.md) showed a cheap model given an
# open-ended "watch and report" job fabricated a plausible-looking failure
# report instead of admitting it couldn't tell if something had genuinely
# hung; a trust-critical stall check needs to fail loud or not at all, which
# only a fixed mtime threshold on a specific file guarantees. Chosen value:
# comfortably longer than any observed healthy gap between clkscan's
# --log-every progress lines or a curl byte-count tick, short enough that a
# real hang costs one extra ~STALL_MINUTES wait, not hours.
STALL_MINUTES=20

# How often to re-announce a blocked month in $HEALLOG (minutes).
BLOCK_REMIND_MINUTES=720

# A lock older than this whose owner is gone is assumed dead and broken.
LOCK_STALE_MINUTES=30

# The lock lives in the VM's /tmp, not on $BASE: /tmp is tmpfs, so mkdir is
# reliably atomic there, and the lock evaporates on VM restart instead of
# outliving the process that held it.
LOCK_DIR="/tmp/fm-corpus-heal.lock"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$RAW_DIR" "$PROCESSED_DIR" 2>/dev/null
mkdir -p "$(dirname "$HEALLOG")" 2>/dev/null

log() {
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] $*" >> "$HEALLOG"
}

# stalled <progress-file> - true if the file exists and its mtime is older
# than STALL_MINUTES. A missing file is NOT stalled (nothing to judge yet -
# e.g. a curl or clkscan that hasn't written its first byte/line).
stalled() {
  [ -e "$1" ] && [ -n "$(find "$1" -maxdepth 0 -mmin +"$STALL_MINUTES" 2>/dev/null)" ]
}

# --- one pass at a time -------------------------------------------------------
# Every "is something already running?" test in this script is a check-then-act
# with a wide window (the HEAD request below sits right in the middle of it), so
# two overlapping passes could both decide nothing was running and both launch.
# Two curls appending to the same -o file with -C - is a plausible way to end up
# with an archive of exactly the right length and the wrong bytes, which is
# precisely the state 2025-04 was in. One pass at a time removes the window.
# The Scheduled Task can still fire as often as it likes; extra invocations
# simply exit 0.
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  lock_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
  if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
    exit 0  # a pass is genuinely in flight
  fi
  # Owner is gone - almost certainly killed by the same event that kills
  # everything else here. Only break the lock once it is also old, so we can't
  # steal it from a pass that is still mid-launch, and so a recycled PID that
  # merely looks alive resolves itself within LOCK_STALE_MINUTES.
  if [ -n "$(find "$LOCK_DIR" -maxdepth 0 -mmin +"$LOCK_STALE_MINUTES" 2>/dev/null)" ]; then
    log "stale lock (owner pid ${lock_pid:-?} gone, >${LOCK_STALE_MINUTES}m old) - breaking it"
    rm -rf "$LOCK_DIR"
    mkdir "$LOCK_DIR" 2>/dev/null || exit 0
  else
    exit 0
  fi
fi
echo "$$" > "$LOCK_DIR/pid"
# Children are setsid+disowned, so releasing the lock never touches them.
trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM

# block <month> <reason> - stop working this month until a human clears it.
block() {
  printf '%s\n' "$2" > "$STATE_DIR/${1}.blocked"
  log "$1: BLOCKED - $2 -- human action needed; delete $STATE_DIR/${1}.blocked to retry"
}

# classify_pass <rcfile> <pplog> - what happened to the last preprocessing run?
# Prints exactly one of: none killed ok fail-truncated fail-other
classify_pass() {
  _rcf="$1"; _plog="$2"
  if [ ! -e "$_rcf" ]; then
    # Launched but never recorded an exit status => it did not reach the end of
    # the wrapper => something killed it. (Also the state of every month
    # processed before this script started recording rc files.)
    if [ -e "$_plog" ]; then echo "killed"; else echo "none"; fi
    return
  fi
  _rc=$(cat "$_rcf" 2>/dev/null || echo "")
  case "$_rc" in ''|*[!0-9]*) echo "killed"; return ;; esac
  if [ "$_rc" -eq 0 ]; then echo "ok"; return; fi
  # Exact wording taken from pp_2025-04_heal.log, not paraphrased. Both lines
  # are matched because clkscan prints the "note:" line even in cases where the
  # final error line might be truncated by a killed writer.
  if grep -qF \
       -e 'Refusing to write a sample from a truncated archive' \
       -e 'input ended early (Data corruption detected)' \
       "$_plog" 2>/dev/null; then
    echo "fail-truncated"
    return
  fi
  echo "fail-other"
}

for month in $MONTHS; do
  # Prefer wherever this month's archive already lives: a file that started
  # downloading before the 2026-08-21 RAW_DIR move finishes in place on D:
  # instead of being orphaned (this script would otherwise see nothing at the
  # new C: path and treat an in-flight month as brand-new / not its job).
  # Every month with no archive yet - anything from here on - lands on C:.
  zst_old="$RAW_DIR_OLD/lichess_db_standard_rated_${month}.pgn.zst"
  zst_new="$RAW_DIR/lichess_db_standard_rated_${month}.pgn.zst"
  if [ -e "$zst_old" ]; then zst="$zst_old"; else zst="$zst_new"; fi
  # Same prefer-existing rule for output: a month already processed (or
  # partially processed) on D: stays on D: forever; only a month with no
  # output anywhere yet gets its outdir created fresh on C:.
  outdir_old="$PROCESSED_DIR_OLD/$month"
  outdir_new="$PROCESSED_DIR/$month"
  if [ -d "$outdir_old" ]; then outdir="$outdir_old"; else outdir="$outdir_new"; fi
  pplog="$LOG_DIR/pp_${month}_heal.log"
  rcfile="$STATE_DIR/${month}.pp.rc"
  blocked="$STATE_DIR/${month}.blocked"
  redlfile="$STATE_DIR/${month}.redownloads"
  archive_tar="$ARCHIVE_DIR/${month}.tar.gz"
  # 2026-08-24 CRITICAL: pkl_count MUST count both possible locations, not
  # just whichever one "prefer existing" picked as $outdir. Five months
  # (2020-07/09/11/12, 2021-01) have their 30000 pkls genuinely SPLIT across
  # both D: and C: from an earlier interrupted relief-move - counting only
  # one side made them look permanently ~23% done and triggered a full
  # redundant re-download+re-scan of an already-complete month. Also: an
  # ARCHIVED month has pkl_count=0 in both places (files live inside the
  # .tar.gz now, loose dirs are removed) - that must be checked SEPARATELY
  # from pkl_count, not inferred from it, or an archived month looks
  # identical to a never-started one and gets redone from scratch too. Both
  # of these actually happened in the first 45 minutes this logic was live.
  pkl_count=$(( $(find "$outdir_old" -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l | tr -d " ") + $(find "$outdir_new" -maxdepth 1 -name "*.pkl" 2>/dev/null | wc -l | tr -d " ") ))

  # already fully done (loose pkls OR already archived - see the note above)
  # -ge, not -eq: a month sitting at 30001 pkls for any reason used to miss this
  # branch forever and get wiped-and-relaunched on every single pass.
  if [ "$pkl_count" -ge 30000 ] || [ -e "$archive_tar" ]; then
    if [ -e "$zst" ]; then
      log "$month: done ($pkl_count pkls / archived=$([ -e "$archive_tar" ] && echo yes || echo no)), deleting leftover zst"
      rm -f "$zst"
    fi
    rm -f "$rcfile" "$blocked" "$redlfile"
    # 2026-08-24: auto-launching archiving from inside this script was tried
    # and reverted same night - captain's actual ask was a one-off compress
    # of the already-complete backlog, not a standing pipeline behavior, and
    # it was also the direct cause of both bugs found tonight (unbounded
    # concurrent launches, and false "unarchived" positives against split/
    # already-archived months). The $archive_tar check above stays (a month
    # already archived by the separate one-off pass must still read as done,
    # not get re-downloaded) but nothing here originates new archiving.
    continue
  fi

  # Checked after the done-test, so a month a human fixes by hand unblocks itself.
  if [ -e "$blocked" ]; then
    if [ -n "$(find "$blocked" -maxdepth 0 -mmin +"$BLOCK_REMIND_MINUTES" 2>/dev/null)" ]; then
      _why=$(tail -n 1 "$blocked" 2>/dev/null)
      log "$month: still BLOCKED - ${_why:-reason unrecorded} -- delete $blocked to retry"
      printf '%s\n' "${_why:-reason unrecorded}" > "$blocked"  # bump mtime = reminder timer
    fi
    continue
  fi

  if [ ! -e "$zst" ]; then
    # No archive at all yet. Originate a fresh download IF there's real
    # concurrency and disk headroom (see the 2026-08-24 comment block above
    # for why these two gates and these exact numbers).
    active_downloads=$(pgrep -f "curl.*lichess_db_standard_rated" 2>/dev/null | wc -l | tr -d " ")
    active_preprocess=$(pgrep -f "preprocess_onepass.py --input" 2>/dev/null | wc -l | tr -d " ")
    active_total=$((active_downloads + active_preprocess))
    avail_kb=$(df --output=avail "$RAW_DIR" 2>/dev/null | tail -1 | tr -d " ")
    avail_gb=$(( ${avail_kb:-0} / 1024 / 1024 ))
    if [ "$active_total" -lt "$AUTOLAUNCH_CONCURRENCY" ] && [ "$avail_gb" -ge "$AUTOLAUNCH_MIN_FREE_GB" ]; then
      log "$month: originating new download (active=$active_total/$AUTOLAUNCH_CONCURRENCY, free=${avail_gb}GB)"
      setsid nohup curl -fsSL -C - --connect-timeout 20 --retry 8 --retry-delay 15 --retry-all-errors \
        --speed-limit 1 --speed-time 30 -o "$zst" \
        "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" \
        > "$LOG_DIR/dl_${month}_heal.log" 2>&1 < /dev/null &
      disown
    fi
    continue
  fi

  local_len=$(stat -c '%s' "$zst" 2>/dev/null || echo "0")
  # --connect-timeout/--max-time: an unbounded HEAD used to be able to stall a
  # whole pass past the Scheduled Task's next firing, which is how two passes
  # end up overlapping in the first place.
  remote_len=$(curl -fsSL -I --connect-timeout 10 --max-time 30 "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" 2>/dev/null | grep -i '^content-length:' | tr -d '\r' | awk '{print $2}')

  case "${remote_len:-}" in
    ''|*[!0-9]*)
      # No usable Content-Length (DNS/network/CDN blip, or a non-numeric header).
      # This used to fall through to the else-branch and relaunch a download for
      # a month whose archive may well be complete. Skip instead: nothing here
      # is time-critical and the next pass is minutes away.
      log "$month: could not read remote Content-Length - skipping this pass (local=${local_len} bytes)"
      continue
      ;;
  esac

  if [ "$local_len" = "$remote_len" ]; then
    # NOTE: length equality is NOT an integrity check. 2025-04 matched the remote
    # Content-Length to the byte and was still corrupt ~74.3M games in. Verifying
    # for real would need to decompress the frame (zstd) or a published checksum,
    # neither of which is available here, so the truncation handling below is the
    # backstop: clkscan is effectively the integrity check, and its verdict is now
    # acted on instead of discarded.

    # download complete - is preprocessing running?
    if pgrep -f "preprocess_onepass.py.*${month}" >/dev/null 2>&1; then
      if stalled "$pplog"; then
        pkill -f "fm-corpus-heal-pp.*${month}" 2>/dev/null
        pkill -f "preprocess_onepass.py.*${month}" 2>/dev/null
        pkill -f "clkscan.*${month}" 2>/dev/null
        log "$month: STALLED - preprocessing log untouched for >${STALL_MINUTES}m, killed it; next pass relaunches via the normal killed-process path"
      fi
      continue
    fi

    # Nothing running. Before wiping or relaunching anything, find out how the
    # previous attempt actually ended.
    verdict=$(classify_pass "$rcfile" "$pplog")

    case "$verdict" in
      fail-truncated)
        n=$(cat "$redlfile" 2>/dev/null || echo 0)
        case "$n" in ''|*[!0-9]*) n=0 ;; esac
        if [ "$n" -lt "$MAX_AUTO_REDOWNLOAD" ]; then
          # Re-download rather than only flagging: this script already owns the
          # download branch, "delete the archive and fetch it again" is exactly
          # what the human did by hand at 15:24, and the failure is
          # self-diagnosing - clkscan told us the input is bad, not the pipeline.
          # Bounded by MAX_AUTO_REDOWNLOAD so this can't become the same
          # unbounded retry loop with a bigger cycle time.
          cp -f "$pplog" "${pplog}.truncated" 2>/dev/null
          # Truncate to zero rather than rm: -C - onto a file that is already
          # corrupt just rebuilds the same corrupt file, so the bad bytes have to
          # go - but deleting the .zst outright makes this month invisible to the
          # "no archive at all yet -> not this script's job" guard above, and it
          # would never be re-downloaded at all. A 0-byte file is case 4
          # (partial), and -C - from offset 0 is a plain full download.
          : > "$zst"
          rm -f "$pplog" "$rcfile"
          rm -rf "$outdir"
          echo $((n + 1)) > "$redlfile"
          log "$month: CORRUPT ARCHIVE - clkscan refused a truncated .zst (${local_len} bytes, matched remote length). Emptied it and queued a fresh full re-download, attempt $((n + 1))/${MAX_AUTO_REDOWNLOAD}. Evidence kept at ${pplog}.truncated"
        else
          block "$month" "archive still truncated after ${n} automatic re-download(s); clkscan refuses to sample it. Source archive needs to be fetched/verified by hand (see ${pplog}.truncated)"
        fi
        continue
        ;;
      fail-other)
        _last=$(tail -n 1 "$pplog" 2>/dev/null | cut -c1-200)
        block "$month" "preprocessing exited non-zero for a reason that is not truncation and will not fix itself (bad args, missing dependency, tool crash). Last log line: ${_last:-<empty log>}. Full log: ${pplog}"
        continue
        ;;
      ok)
        # Clean exit but short output. Retrying reproduces it exactly, so this is
        # another infinite loop waiting to happen - stop and let a human look.
        block "$month" "preprocessing exited 0 but only ${pkl_count}/30000 pkls exist - the tool thinks it succeeded and this script does not. Retrying would repeat forever. Full log: ${pplog}"
        continue
        ;;
      killed|none)
        : # externally killed, or never run - relaunching is the correct response
        ;;
    esac

    # complete zst, nothing processing it, not yet done, last attempt did not
    # fail on its own terms - (re)launch
    # 2026-08-24: pkl_count sums BOTH outdir_old and outdir_new (see the
    # combined-count fix above), so the wipe must clear both too - wiping only
    # $outdir (whichever "prefer existing" picked) left stale pkls in the
    # other location for a month with split storage, which then get
    # miscounted as progress on the NEXT pass and can trigger a premature
    # "done" detection on a restart that never actually finished. Caught in
    # code review before it could bite a real month.
    if [ "$pkl_count" -gt 0 ]; then
      log "$month: found $pkl_count/30000 pkls with no process running - partial sample is unusable, wiping and restarting preprocessing"
      rm -rf "$outdir_old" "$outdir_new"
    fi
    if ! mkdir -p "$outdir" 2>/dev/null; then
      log "$month: cannot create $outdir - skipping this pass"
      continue
    fi
    # Stale rc from the previous attempt must go before the new one starts, or
    # a pass that gets killed early would be read as the old attempt's failure.
    rm -f "$rcfile"
    # The wrapper exists only to capture the real exit status; the python
    # invocation and its log path are unchanged. Written as one line so the
    # cmdline stays greppable by the pgrep above.
    setsid nohup bash -c 'rc=0; "$1" "$2" --input "$3" --output-dir "$4" --clkscan "$5" --max-games 30000 > "$6" 2>&1 < /dev/null || rc=$?; printf "%s\n" "$rc" > "$7"' \
      fm-corpus-heal-pp "$PYTHON" "$PREPROCESS_SCRIPT" "$zst" "$outdir" "$CLKSCAN" "$pplog" "$rcfile" \
      > /dev/null 2>&1 < /dev/null &
    disown
    log "$month: launched preprocessing (zst verified complete, $local_len bytes)"
  else
    # download not complete - is a download already running?
    # Matched on the -o output path, not the bare filename: the bare filename
    # also appears in the preprocessing command line (and in a concurrent pass's
    # HEAD request URL), so this test used to report "downloading" for a month
    # that was actually being preprocessed.
    if pgrep -f "curl .*-o .*lichess_db_standard_rated_${month}\.pgn\.zst" >/dev/null 2>&1; then
      if stalled "$zst"; then
        pkill -f "curl .*-o .*lichess_db_standard_rated_${month}\.pgn\.zst" 2>/dev/null
        log "$month: STALLED - download made no byte progress for >${STALL_MINUTES}m, killed it; next pass resumes via -C - as usual"
      fi
      continue
    fi
    # This replaces the earlier --speed-limit/--speed-time hesitation (2026-08-22):
    # that flag kills the curl PROCESS itself on a stall, same as the check
    # above, but ALSO fires on brief legitimate slow patches mid-transfer,
    # forcing a resume seam every time - and a resume seam is one of the
    # candidate causes of the 2025-04 corruption. The stall check above only
    # acts between passes (>=STALL_MINUTES of true silence), so a normal
    # slow-but-moving transfer is never touched.
    setsid nohup curl -fsSL -C - --connect-timeout 20 --retry 8 --retry-delay 15 --retry-all-errors \
      -o "$zst" "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" \
      > "$LOG_DIR/dl_${month}_heal.log" 2>&1 < /dev/null &
    disown
    log "$month: (re)launched download, was at ${local_len:-0}/${remote_len:-?} bytes"
  fi
done

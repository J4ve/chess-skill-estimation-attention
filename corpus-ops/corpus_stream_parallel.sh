#!/usr/bin/env bash
# corpus_stream_parallel.sh — hardened parallel corpus downloader/preprocessor.
#
# ONE script for BOTH hosts. The previous design shipped two near-identical files
# (hpc_corpus_stream_parallel.sh and local_corpus_stream_parallel.sh) that had to
# be kept in sync by hand; the month lists are the only thing that actually
# differs, so they live in a case statement below and the shared logic exists
# once. Two copies of concurrency-critical bash WILL drift.
#
#   bash corpus_stream_parallel.sh --host hpc      # run on the HPC
#   bash corpus_stream_parallel.sh --host local    # run on the captain's PC
#   bash corpus_stream_parallel.sh --host hpc --status
#
# Safe to re-run at any time, including after a hard power-off mid-month. It
# reclaims its own stranded work immediately and other hosts' stranded work
# after a grace period. See analysis/corpus-pipeline-hardening.md.
#
# Exit codes: 0 = this host's list is fully done or claimed elsewhere,
#             1 = finished but some months are in a terminal FAILED state.

set -uo pipefail

# ------------------------------------------------------------------ config

HOST_ROLE=""
DO_STATUS=0
SCANNER="python"          # python | rust  (see --scanner)
CONCURRENCY_OVERRIDE=""

# The HPC is the coordination point: the claim/state directory lives there and
# only there, on both hosts' behalf.
HPC_SSH="<user>@<hpc-host>"
HPC_SSH_KEY="$HOME/.ssh/<key>"
REMOTE_ROOT="~/thesis2"

RAW_DIR="data/raw_zst"
OUT_DIR="data/processed_games"
CLAIM_DIR="data/.claims"
STATE_DIR="data/.state"
LOG_DIR="logs"

MAX_GAMES=30000
# Two separate counters, because "we started work on this month" and "this month
# was tried and judged to have failed" are different events and only the second
# should burn a retry. See analysis/corpus-resilience-review.md G3: charging the
# attempt at claim time meant three ordinary interruptions (Ctrl-C, power cut,
# laptop sleep) put a perfectly healthy month into a terminal FAILED state.
MAX_ATTEMPTS=3            # judged failures per month, across all runs and hosts, then FAILED
MAX_STARTS=20             # claim-time starts per month; a much looser crash-loop backstop
STALE_MIN=45              # a claim with no heartbeat for this long is stale
HEARTBEAT_S=60
DOWNLOAD_ATTEMPTS=6
RETRY_SLEEP=60            # pause between download attempts

PREPROCESS_SCRIPT="src/preprocess_lichess.py"
# --scanner rust prefers the one-pass extractor and falls back to the two-pass
# replay script, so a deployment that has only ever synced the older
# preprocess_fast.py keeps working instead of silently reverting to pure Python.
ONEPASS_PREPROCESS_SCRIPT="analysis/scripts/preprocess_onepass.py"
FAST_PREPROCESS_SCRIPT="analysis/scripts/preprocess_fast.py"
CLKSCAN_BIN="bin/clkscan"
PYTHON_BIN="$HOME/miniconda3/envs/ratingnet2/bin/python"

BASE_URL="https://database.lichess.org/standard"

# Expected sizes verified via Content-Range. A month with no entry here is NOT
# silently trusted (that was the old behavior and it let truncated downloads
# through); it is refused up front by validate_months.
#
# Deliberately a function over a `declare -A` associative array. Three reasons,
# all of which bit the previous version:
#   1. Associative arrays need bash 4+. macOS still ships bash 3.2, where
#      `declare -A` fails and `${SIZES[2021-04]}` then degrades into an
#      ARITHMETIC subscript: 2021-04 evaluates to 2017, so every lookup silently
#      returns the wrong entry (empty) instead of erroring. A silent wrong size
#      on a 30GB download is the worst possible failure mode here.
#   2. Associative arrays cannot be exported to a subshell, which is the entire
#      reason the old script re-invoked itself as `bash "$0" --worker <month>`.
#      A plain function has no such limit.
#   3. It keeps one source of truth that both the parent and workers read the
#      same way.
expected_size() {
  case "$1" in
    2021-04) echo 32128332169 ;; 2021-05) echo 32692123854 ;; 2021-06) echo 29730565470 ;;
    2021-07) echo 29809494467 ;; 2021-08) echo 30392855649 ;; 2021-09) echo 28460843524 ;;
    2021-10) echo 28478996133 ;; 2021-11) echo 28207869558 ;; 2021-12) echo 31107096955 ;;
    2022-01) echo 33232239428 ;; 2022-02) echo 27971810913 ;; 2022-03) echo 29397132411 ;;
    2022-04) echo 28144915874 ;; 2022-05) echo 28784698676 ;; 2022-06) echo 28331039141 ;;
    2022-07) echo 29807104208 ;; 2022-08) echo 29984806278 ;; 2022-09) echo 28887786014 ;;
    2022-10) echo 29973814533 ;; 2022-11) echo 28851245687 ;; 2022-12) echo 30161508176 ;;
    2023-01) echo 33465852948 ;; 2023-02) echo 31838284336 ;; 2023-03) echo 34864029436 ;;
    2023-04) echo 32870230722 ;; 2023-05) echo 33724525615 ;; 2023-06) echo 31246059602 ;;
    2023-07) echo 30849136644 ;; 2023-08) echo 31248072774 ;; 2023-09) echo 30254169325 ;;
    2023-10) echo 30813798999 ;; 2023-11) echo 30047560534 ;; 2023-12) echo 31654288404 ;;
    2024-01) echo 32379325271 ;; 2024-08) echo 30038633172 ;;
    *) echo "" ;;
  esac
}

# Months already built by the earlier 170k subset run. Touching these would
# destroy existing work, so they are refused outright.
PROTECTED_MONTHS="2024-02 2024-03 2024-04 2024-05 2024-06 2024-07"

# ------------------------------------------------------------------ args

while [ $# -gt 0 ]; do
  case "$1" in
    --host)        HOST_ROLE="${2:-}"; shift 2 ;;
    --status)      DO_STATUS=1; shift ;;
    --scanner)     SCANNER="${2:-}"; shift 2 ;;
    --concurrency) CONCURRENCY_OVERRIDE="${2:-}"; shift 2 ;;
    --worker)      WORKER_MONTH="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$HOST_ROLE" in
  hpc)
    CONCURRENCY=5
    MONTHS="2021-04 2021-05 2021-06 2021-07 2021-08 2021-09 2021-10 2021-11 2021-12 \
2022-01 2022-02 2022-03 2022-04 2022-05 2022-06 2022-07 2022-08 2022-09 2022-10 2022-11 2022-12 \
2023-01 2023-02 2023-03 2023-04 2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 \
2024-01 2024-08"
    ;;
  local)
    CONCURRENCY=3
    MONTHS=""     # extension months; empty until the captain assigns them
    ;;
  *)
    echo "error: --host must be 'hpc' or 'local'" >&2
    echo "  bash $0 --host hpc" >&2
    exit 2 ;;
esac
[ -n "$CONCURRENCY_OVERRIDE" ] && CONCURRENCY="$CONCURRENCY_OVERRIDE"

case "$SCANNER" in
  python|rust) ;;
  *) echo "error: --scanner must be 'python' or 'rust'" >&2; exit 2 ;;
esac

# Test hook. analysis/scripts/test_pipeline_recovery.sh sets this to exercise the
# claim, staleness and crash-recovery paths for real (including SIGKILL) without
# downloading 30GB per month. Unset in production, and the script is unaffected.
if [ -n "${CORPUS_STREAM_TEST_CONFIG:-}" ] && [ -f "$CORPUS_STREAM_TEST_CONFIG" ]; then
  . "$CORPUS_STREAM_TEST_CONFIG"
  log_test_hook=1
fi

cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" || exit 1

# ------------------------------------------------------------------ helpers

log() { echo "$(date '+%F %T') [$HOST_ROLE] $*" | tee -a "$LOG_DIR/corpus_parallel.log" >&2; }

# Portable file size (GNU stat on Linux, BSD stat on macOS). The old script used
# GNU-only `stat -c%s`, which silently returns 0 on macOS and would have made
# every size check fail on a Mac local machine.
file_size() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0; }

# Free bytes on the filesystem holding a path. -P forces POSIX single-line output
# so the awk column is stable across GNU and BSD df.
free_bytes() { df -Pk "${1:-.}" 2>/dev/null | awk 'NR==2 {printf "%.0f", $4 * 1024}'; }

# Space needed for one month: the rest of the archive, plus room for the ~6GB of
# pickles it produces, plus a margin. At CONCURRENCY=5 five of these run at once,
# which is how the serial design's documented ~30GB peak became ~150GB without
# anyone re-checking a /home that CLAUDE.md puts at ~92% full.
OUTPUT_HEADROOM=8000000000

# ---- coordination-host command execution -----------------------------------
# All CLAIM AND STATE mutations go through ctl() so they happen in exactly one
# place, the HPC, regardless of which host is doing the actual work.
ctl() {
  if [ "$HOST_ROLE" = "hpc" ]; then
    bash -c "$1"
  else
    ssh -i "$HPC_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=15 \
        -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
        "$HPC_SSH" "cd $REMOTE_ROOT && $1"
  fi
}

# ssh exits 255 for its OWN failures (host unreachable, auth, timeout). Remote
# `mkdir` on an existing directory exits 1. Conflating these is the bug that
# makes a network blip look like "another worker owns this month" and silently
# skip it. Every ctl() caller that cares must distinguish 255.
CTL_INFRA_RC=255

# ------------------------------------------------------------------ claims
#
# A claim is a directory (atomic mkdir) that additionally carries liveness
# information, because "the directory exists" alone cannot distinguish:
#   (a) a worker actively processing the month,
#   (b) a worker that was SIGKILLed / lost power an hour ago,
#   (c) a month that finished successfully.
# The old script could not tell these apart, so (b) stranded a month forever.
#
#   $CLAIM_DIR/<month>/owner      "<host>:<pid>:<epoch>"
#   $CLAIM_DIR/<month>/heartbeat  touched every $HEARTBEAT_S while work proceeds
#
# (c) is settled separately by $OUT_DIR/<month>/.corpus_done, and the claim is
# released on success, so a lingering claim always means (a) or (b).

try_claim() {
  local m="$1" rc
  ctl "mkdir '$CLAIM_DIR/$m' 2>/dev/null" >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq "$CTL_INFRA_RC" ] && [ "$HOST_ROLE" = "local" ]; then
    return 2   # coordination host unreachable: NOT a lost claim
  fi
  [ "$rc" -eq 0 ] || return 1
  # The timestamp must be evaluated by the shell that WRITES the file (the
  # coordination host), not by this one, so it is consistent with the heartbeat
  # mtimes the reaper compares against. Hence the escaped command substitution
  # inside double quotes: single quotes here would store the literal text.
  ctl "printf '%s' \"$HOST_ROLE:$$:\$(date +%s)\" > '$CLAIM_DIR/$m/owner'; touch '$CLAIM_DIR/$m/heartbeat'" >/dev/null 2>&1
  return 0
}

release_claim() { ctl "rm -rf '$CLAIM_DIR/$1'" >/dev/null 2>&1; }

# Reclaim stranded claims. Two independent mechanisms, because neither alone
# covers both cases:
#
#   1. SAME-HOST DEAD PID (instant). Covers the captain's named scenario: the PC
#      is switched off mid-month and switched back on. On the next run the old
#      pids are gone, so those claims are reclaimed immediately with no waiting.
#   2. STALE HEARTBEAT (after $STALE_MIN). Covers a crash on the OTHER host,
#      whose pids this host cannot inspect. Uses the coordination host's own
#      filesystem mtime, so it needs no clock agreement between the machines.
reap_stale_claims() {
  local claims m owner ohost opid reclaimed=0
  claims=$(ctl "ls -1 '$CLAIM_DIR' 2>/dev/null" 2>/dev/null) || return 0

  for m in $claims; do
    [ -z "$m" ] && continue
    owner=$(ctl "cat '$CLAIM_DIR/$m/owner' 2>/dev/null" 2>/dev/null)
    ohost="${owner%%:*}"
    opid=$(echo "$owner" | cut -d: -f2 | tr -dc '0-9')

    if [ "$ohost" = "$HOST_ROLE" ] && [ -n "$opid" ]; then
      if ! kill -0 "$opid" 2>/dev/null; then
        log "reclaiming $m: owned by this host, pid $opid is gone"
        release_claim "$m"; reclaimed=$((reclaimed + 1))
      fi
      continue
    fi

    if [ -n "$(ctl "find '$CLAIM_DIR/$m' -maxdepth 1 -name heartbeat -mmin +$STALE_MIN -print 2>/dev/null" 2>/dev/null)" ]; then
      log "reclaiming $m: heartbeat stale (>${STALE_MIN}m, owner ${owner:-unknown})"
      release_claim "$m"; reclaimed=$((reclaimed + 1))
    elif [ -z "$(ctl "ls '$CLAIM_DIR/$m/heartbeat' 2>/dev/null" 2>/dev/null)" ]; then
      # Claimed but never heartbeated: the worker died between mkdir and the
      # first touch. Rare, but it strands the month permanently if ignored.
      log "reclaiming $m: claim has no heartbeat at all"
      release_claim "$m"; reclaimed=$((reclaimed + 1))
    fi
  done
  [ "$reclaimed" -gt 0 ] && log "reclaimed $reclaimed stranded claim(s)"
  return 0
}

heartbeat_pid=""
start_heartbeat() {
  local m="$1"
  ( while :; do
      ctl "touch '$CLAIM_DIR/$m/heartbeat' 2>/dev/null" >/dev/null 2>&1
      sleep "$HEARTBEAT_S"
    done ) &
  heartbeat_pid=$!
}
stop_heartbeat() {
  [ -n "$heartbeat_pid" ] && kill "$heartbeat_pid" 2>/dev/null
  heartbeat_pid=""
}

# ------------------------------------------------------------------ attempts

attempts_get() { ctl "cat '$STATE_DIR/attempts/$1' 2>/dev/null" 2>/dev/null | tr -dc '0-9'; }
attempts_inc() {
  ctl "mkdir -p '$STATE_DIR/attempts'; n=\$(cat '$STATE_DIR/attempts/$1' 2>/dev/null || echo 0); echo \$((n+1)) > '$STATE_DIR/attempts/$1'" >/dev/null 2>&1
}

# Starts are charged when a worker takes the claim, before any work happens.
# They exist only to stop a month that dies instantly and repeatedly (an
# OOM-killed preprocessor, a corrupt archive that crashes the parser) from
# retry-looping forever, which is the property analysis/corpus-pipeline-hardening.md
# lines 113-117 asked for. They are NOT the retry budget: MAX_STARTS is loose
# enough that ordinary interruptions never reach it.
starts_get() { ctl "cat '$STATE_DIR/starts/$1' 2>/dev/null" 2>/dev/null | tr -dc '0-9'; }
starts_inc() {
  ctl "mkdir -p '$STATE_DIR/starts'; n=\$(cat '$STATE_DIR/starts/$1' 2>/dev/null || echo 0); echo \$((n+1)) > '$STATE_DIR/starts/$1'" >/dev/null 2>&1
}

mark_failed() {
  ctl "mkdir -p '$STATE_DIR/failed'; printf '%s\n' '$2' > '$STATE_DIR/failed/$1'" >/dev/null 2>&1
  log "!!! MONTH $1 IS NOW IN A TERMINAL FAILED STATE: $2"
  log "!!! it will NOT be retried. clear it with: rm -f $STATE_DIR/failed/$1 $STATE_DIR/attempts/$1 $STATE_DIR/starts/$1"
}
is_failed() { [ -n "$(ctl "ls '$STATE_DIR/failed/$1' 2>/dev/null" 2>/dev/null)" ]; }

# Nothing will retry a terminally failed month, so its raw archive is dead
# weight on a disk that is already tight (CLAUDE.md: /home is ~92% full).
# Reclaim it; a human clearing the FAILED marker can re-download.
# analysis/corpus-pipeline-hardening.md:118 states this invariant, and it must
# hold on EVERY path that marks a month failed. It previously held on only one
# of them (analysis/corpus-resilience-review.md G3).
reclaim_archive() {
  local m="$1" zst="$2"
  [ -f "$zst" ] || return 0
  log "$m: reclaiming $(( $(file_size "$zst") / 1000000000 ))GB of raw archive from the failed month"
  rm -f "$zst"
}

# ------------------------------------------------------------------ validation

validate_months() {
  local m bad=0
  for m in $MONTHS; do
    if echo "$PROTECTED_MONTHS" | grep -qw -- "$m"; then
      echo "FATAL: $m is a protected month (already built by the 170k subset run)" >&2; bad=1
    fi
    if [ -z "$(expected_size "$m")" ]; then
      echo "FATAL: no expected size recorded for $m; refusing to download it unverified" >&2; bad=1
    fi
  done
  [ "$bad" -eq 0 ] || exit 1
}

# ------------------------------------------------------------------ work

download_month() {
  local m="$1" zst="$2" url="$BASE_URL/lichess_db_standard_rated_${m}.pgn.zst"
  local exp attempt sz rc
  exp="$(expected_size "$m")"

  local avail need
  for attempt in $(seq 1 "$DOWNLOAD_ATTEMPTS"); do
    sz=$(file_size "$zst")
    [ "$sz" -eq "$exp" ] 2>/dev/null && { log "$m: download complete ($sz B)"; return 0; }

    # Refuse to start a 30GB download that cannot finish. Filling a shared /home
    # does not just fail this month, it takes down other users' jobs too.
    avail=$(free_bytes "$RAW_DIR")
    need=$(( exp - sz + OUTPUT_HEADROOM ))
    if [ -n "$avail" ] && [ "$avail" -lt "$need" ] 2>/dev/null; then
      log "$m: ABORT, only $((avail / 1000000000))GB free but this month needs about $((need / 1000000000))GB"
      log "$m: (at concurrency $CONCURRENCY several months stage at once; free space or lower --concurrency)"
      return 1
    fi
    if [ "$sz" -gt "$exp" ] 2>/dev/null; then
      log "$m: local file oversized ($sz > $exp), restarting from scratch"; rm -f "$zst"
    fi

    # -C - RESUMES from the partial bytes already on disk. The previous version
    # deleted the partial file on every mismatch, which made -C - dead code and
    # meant a connection dropping at 29GB of a 30GB file threw away all 29GB.
    # -f so an HTTP 404/5xx body is never written into the archive as if it were
    # data (curl without -f saves the error page and still exits 0).
    curl -fsSL -C - --retry 3 --retry-delay 30 -o "$zst" "$url"; rc=$?
    sz=$(file_size "$zst")
    [ "$sz" -eq "$exp" ] 2>/dev/null && { log "$m: download complete ($sz B)"; return 0; }

    # Overshoot means the resume appended a duplicate range (curl's own --retry
    # can re-append onto -o output). The partial bytes are worthless now, so
    # discard and restart immediately rather than sleeping first.
    if [ "$sz" -gt "$exp" ] 2>/dev/null; then
      log "$m: download overshot ($sz > $exp), discarding and restarting"
      rm -f "$zst"; continue
    fi

    log "$m: download attempt $attempt/$DOWNLOAD_ATTEMPTS rc=$rc size=$sz/$exp, retrying in ${RETRY_SLEEP}s"
    sleep "$RETRY_SLEEP"
  done
  log "$m: download FAILED after $DOWNLOAD_ATTEMPTS attempts (size=$(file_size "$zst")/$exp)"
  return 1
}

preprocess_month() {
  local m="$1" zst="$2" outdir="$3" staging kept rc

  # Build into a private staging directory and move it into place only once it
  # is complete. If two workers ever process the same month concurrently (a
  # heartbeat false-positive, say), they cannot corrupt each other's output, and
  # a crash mid-preprocess never leaves a half-built directory that a later
  # kept-count check would misread.
  staging="${outdir}.staging.$$"
  rm -rf "$staging"; mkdir -p "$staging" "$LOG_DIR"

  if [ "$SCANNER" = "rust" ] && [ -x "$CLKSCAN_BIN" ] && [ -f "$ONEPASS_PREPROCESS_SCRIPT" ]; then
    log "$m: preprocessing with the Rust one-pass extractor"
    "$PYTHON_BIN" -u "$ONEPASS_PREPROCESS_SCRIPT" --input "$zst" --output-dir "$staging" \
      --max-games "$MAX_GAMES" --clkscan "$CLKSCAN_BIN" \
      --log-every 200000 > "$LOG_DIR/preprocess_${m}.log" 2>&1
    rc=$?
  elif [ "$SCANNER" = "rust" ] && [ -x "$CLKSCAN_BIN" ] && [ -f "$FAST_PREPROCESS_SCRIPT" ]; then
    log "$m: preprocessing with the Rust scan fast path (two-pass; $ONEPASS_PREPROCESS_SCRIPT not present)"
    "$PYTHON_BIN" -u "$FAST_PREPROCESS_SCRIPT" --input "$zst" --output-dir "$staging" \
      --max-games "$MAX_GAMES" --clkscan "$CLKSCAN_BIN" > "$LOG_DIR/preprocess_${m}.log" 2>&1
    rc=$?
  else
    [ "$SCANNER" = "rust" ] && log "$m: --scanner rust requested but $CLKSCAN_BIN, $ONEPASS_PREPROCESS_SCRIPT and $FAST_PREPROCESS_SCRIPT are not all present, falling back to Python"
    "$PYTHON_BIN" -u "$PREPROCESS_SCRIPT" --input "$zst" --output-dir "$staging" \
      --max-games "$MAX_GAMES" --log-every 200000 > "$LOG_DIR/preprocess_${m}.log" 2>&1
    rc=$?
  fi

  if [ "$rc" -ne 0 ]; then
    log "$m: preprocess exited $rc (see $LOG_DIR/preprocess_${m}.log)"
    rm -rf "$staging"; return 1
  fi

  kept=$(find "$staging" -maxdepth 1 -name 'game_*.pkl' | wc -l | tr -d ' ')
  if [ "$kept" -lt "$MAX_GAMES" ]; then
    log "$m: preprocess kept only $kept/$MAX_GAMES games"
    rm -rf "$staging"; return 1
  fi

  rm -rf "$outdir"
  mv "$staging" "$outdir"
  touch "$outdir/.corpus_done"
  log "$m: preprocess OK ($kept games kept)"
  return 0
}

process_one_month() {
  local m="$1" zst outdir attempts
  zst="$RAW_DIR/lichess_db_standard_rated_${m}.pgn.zst"
  outdir="$OUT_DIR/$m"

  [ -f "$outdir/.corpus_done" ] && { log "$m: already done, skipping"; return 0; }
  if is_failed "$m"; then
    log "$m: in terminal FAILED state, skipping (clear $STATE_DIR/failed/$m to retry)"
    return 0
  fi

  attempts=$(attempts_get "$m"); attempts="${attempts:-0}"
  if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
    mark_failed "$m" "exhausted $MAX_ATTEMPTS attempts"
    reclaim_archive "$m" "$zst"
    return 1
  fi

  local starts
  starts=$(starts_get "$m"); starts="${starts:-0}"
  if [ "$starts" -ge "$MAX_STARTS" ]; then
    mark_failed "$m" "started $MAX_STARTS times without ever reaching a verdict (crash loop?)"
    reclaim_archive "$m" "$zst"
    return 1
  fi

  case "$(try_claim "$m"; echo $?)" in
    1) log "$m: claimed by another worker, skipping"; return 0 ;;
    2) log "$m: coordination host unreachable, NOT skipping, will retry later"; return 3 ;;
  esac

  # Re-check completion AFTER winning the claim: another worker may have
  # finished this month between the check above and the claim below.
  if [ -f "$outdir/.corpus_done" ]; then
    log "$m: completed by another worker while claiming, skipping"
    release_claim "$m"; return 0
  fi

  # Release the claim on ANY exit from here, including Ctrl-C and SIGTERM. This
  # covers every death except SIGKILL and power loss, which the staleness reaper
  # handles on the next run.
  trap 'stop_heartbeat; release_claim "'"$m"'"; exit 130' INT TERM
  start_heartbeat "$m"
  # Charge a START, not an attempt. The attempt is charged below, only if the
  # work actually runs to a verdict and that verdict is failure. An interruption
  # between here and there costs a start and nothing else, so the month stays
  # retryable however many times the captain's PC goes to sleep.
  starts_inc "$m"
  log "$m: claimed (attempt $((attempts + 1))/$MAX_ATTEMPTS, start $((starts + 1))/$MAX_STARTS), starting"

  mkdir -p "$RAW_DIR" "$LOG_DIR"
  local ok=1
  if download_month "$m" "$zst"; then
    if preprocess_month "$m" "$zst" "$outdir"; then ok=0; fi
  fi

  stop_heartbeat
  trap - INT TERM

  if [ "$ok" -eq 0 ]; then
    rm -f "$zst"
    if [ "$HOST_ROLE" = "local" ]; then
      log "$m: syncing results to the coordination host"
      rsync -a -e "ssh -i $HPC_SSH_KEY -o BatchMode=yes" "$outdir/" "$HPC_SSH:$REMOTE_ROOT/$OUT_DIR/$m/" \
        || { log "$m: rsync back to HPC FAILED; local copy kept at $outdir"; release_claim "$m"; return 1; }
    fi
    release_claim "$m"
    log "$m: DONE"
    return 0
  fi

  # The work ran and was judged to have failed. THIS is what burns a retry.
  attempts_inc "$m"
  attempts=$(attempts_get "$m"); attempts="${attempts:-0}"
  if [ "$attempts" -ge "$MAX_ATTEMPTS" ]; then
    mark_failed "$m" "failed on attempt $attempts (see $LOG_DIR/preprocess_${m}.log)"
    reclaim_archive "$m" "$zst"
  else
    # Deliberately KEEP the raw archive here. It was size-verified, so the retry
    # skips the download entirely instead of re-fetching 30GB.
    log "$m: attempt $attempts failed, releasing claim so it can be retried (raw archive kept for the retry)"
  fi
  release_claim "$m"
  return 1
}

# ------------------------------------------------------------------ status

show_status() {
  local m done_c=0 active_c=0 todo_c=0 failed_c=0 owner ohost started now age rate eta_s a
  local -a done_l=() active_l=() todo_l=() failed_l=()
  now=$(date +%s)

  for m in $MONTHS; do
    if [ -f "$OUT_DIR/$m/.corpus_done" ] || [ -n "$(ctl "ls '$OUT_DIR/$m/.corpus_done' 2>/dev/null" 2>/dev/null)" ]; then
      done_c=$((done_c + 1)); done_l+=("$m"); continue
    fi
    if is_failed "$m"; then
      failed_c=$((failed_c + 1)); failed_l+=("$m [$(ctl "cat '$STATE_DIR/failed/$m' 2>/dev/null" 2>/dev/null)]"); continue
    fi
    owner=$(ctl "cat '$CLAIM_DIR/$m/owner' 2>/dev/null" 2>/dev/null)
    if [ -n "$owner" ]; then
      ohost="${owner%%:*}"
      started=$(echo "$owner" | cut -d: -f3 | tr -dc '0-9')
      [ -n "$started" ] || started="$now"
      age=$(( (now - started) / 60 ))
      rate=$(grep -o 'scanned=[0-9]*' "$LOG_DIR/preprocess_${m}.log" 2>/dev/null | tail -1 | cut -d= -f2)
      active_c=$((active_c + 1))
      active_l+=("$m  on=$ohost  running=${age}m  scanned=${rate:-0}  attempt=$(attempts_get "$m")/$MAX_ATTEMPTS  start=$(starts_get "$m")/$MAX_STARTS")
      continue
    fi
    # Surface a non-zero attempt count on unclaimed months too. Previously the
    # counter was only ever printed for months holding a live claim, so a month
    # sitting at 2/3 after two failures looked identical to an untouched one
    # right up until the run that made it terminal.
    todo_c=$((todo_c + 1))
    a=$(attempts_get "$m"); a="${a:-0}"
    if [ "${a:-0}" -gt 0 ] 2>/dev/null; then
      todo_l+=("$m(attempt=$a/$MAX_ATTEMPTS)")
    else
      todo_l+=("$m")
    fi
  done

  local total=$((done_c + active_c + todo_c + failed_c))
  echo
  echo "  corpus pipeline status  ($HOST_ROLE, $(date '+%F %T'))"
  echo "  ------------------------------------------------------------"
  printf "  done %d/%d   in progress %d   not started %d   FAILED %d\n" \
         "$done_c" "$total" "$active_c" "$todo_c" "$failed_c"
  echo

  if [ "$active_c" -gt 0 ]; then
    echo "  in progress:"; printf "    %s\n" "${active_l[@]}"; echo
  fi
  if [ "$failed_c" -gt 0 ]; then
    echo "  FAILED (needs a human, will not retry on its own):"
    printf "    %s\n" "${failed_l[@]}"; echo
  fi
  if [ "$todo_c" -gt 0 ]; then
    echo "  not started: ${todo_l[*]}"; echo
  fi

  # ETA from this host's own observed per-month wall time, not a guess.
  if [ "$done_c" -gt 0 ] && [ -f "$LOG_DIR/corpus_parallel.log" ]; then
    local first_done avg
    first_done=$(grep -m1 ': DONE' "$LOG_DIR/corpus_parallel.log" 2>/dev/null | cut -d' ' -f1-2)
    if [ -n "$first_done" ]; then
      local t0 elapsed
      t0=$(date -d "$first_done" +%s 2>/dev/null || date -j -f '%Y-%m-%d %H:%M:%S' "$first_done" +%s 2>/dev/null)
      if [ -n "$t0" ]; then
        elapsed=$((now - t0)); avg=$((elapsed / done_c))
        eta_s=$(( (todo_c + active_c) * avg / (CONCURRENCY > 0 ? CONCURRENCY : 1) ))
        printf "  average %dh%02dm per month observed; rough ETA for the rest: %dh (at concurrency %d)\n" \
               $((avg / 3600)) $(((avg % 3600) / 60)) $((eta_s / 3600)) "$CONCURRENCY"
      fi
    fi
  fi
  echo "  scanner: $SCANNER      logs: $LOG_DIR/preprocess_<month>.log"
  echo
  [ "$failed_c" -gt 0 ] && return 1
  return 0
}

# ------------------------------------------------------------------ main

mkdir -p "$LOG_DIR"

# Worker mode: one month, re-invoked by xargs. The re-invocation now exists only
# to give each month its own process (so one month's failure cannot take down
# its siblings); the size table is a function, so nothing needs re-declaring.
# $SELF is an absolute path, so this is safe regardless of the caller's working
# directory, and it re-reads the script from disk, which means EDITING THIS FILE
# WHILE A RUN IS IN FLIGHT changes the behavior of workers launched afterwards.
# Copy it aside and edit the copy if you need to change it mid-run.
if [ -n "${WORKER_MONTH:-}" ]; then
  process_one_month "$WORKER_MONTH"
  exit $?
fi

if [ "$DO_STATUS" -eq 1 ]; then
  show_status
  exit $?
fi

validate_months
[ -z "$MONTHS" ] && { log "no months assigned to host '$HOST_ROLE'; nothing to do"; exit 0; }

ctl "mkdir -p '$CLAIM_DIR' '$STATE_DIR/attempts' '$STATE_DIR/failed' '$OUT_DIR'" >/dev/null 2>&1
mkdir -p "$RAW_DIR" "$OUT_DIR" "$LOG_DIR"

# Recover anything stranded by a previous run BEFORE claiming new work. This is
# what makes "turn the PC off, turn it back on, run the same command" work with
# no flags and no manual cleanup.
log "checking for stranded claims from previous runs"
reap_stale_claims

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
log "starting: $(echo $MONTHS | wc -w | tr -d ' ') months, concurrency=$CONCURRENCY, scanner=$SCANNER"

# Kill the whole worker process group on Ctrl-C rather than leaving orphaned
# curl/python children running against the shared box.
trap 'log "interrupted, stopping workers"; kill 0 2>/dev/null; exit 130' INT TERM

printf '%s\n' $MONTHS | xargs -P "$CONCURRENCY" -I{} \
  bash "$SELF" --host "$HOST_ROLE" --scanner "$SCANNER" --worker {}

log "pass complete"
show_status
exit $?

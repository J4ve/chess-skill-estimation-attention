#!/usr/bin/env bash
# test_pipeline_recovery.sh — proves corpus_stream_parallel.sh recovers from hard
# kills without human help.
#
# The captain's named scenario is "the PC gets switched off mid-month and
# switched back on". A graceful Ctrl-C proves nothing about that: the trap
# handles it. What has to be proven is SIGKILL, where no cleanup code runs at
# all, and then that re-running THE SAME COMMAND with no flags recovers.
#
# Everything is local: a throttled HTTP server stands in for
# database.lichess.org and a stub stands in for preprocess_lichess.py, so the
# claim, staleness, resume and retry-cap logic is exercised for real without
# 30GB of downloads. Run it from the repo root:
#
#     bash analysis/scripts/test_pipeline_recovery.sh
#
# Exits 0 only if every scenario passes.

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SB="${TMPDIR:-/tmp}/corpus_recovery_test.$$"
PORT=$(( 18000 + (RANDOM % 2000) ))
PY="${TEST_PYTHON:-python3}"
BLOB_SIZE=2000000
RATE=250000          # ~8s per download, plenty of room to kill mid-flight

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf "    \033[32mPASS\033[0m %s\n" "$1"; }
bad()  { FAIL=$((FAIL+1)); printf "    \033[31mFAIL\033[0m %s\n" "$1"; }
head_() { printf "\n  == %s ==\n" "$1"; }

cleanup() {
  [ -n "${SRV_PID:-}" ] && kill -9 "$SRV_PID" 2>/dev/null
  pkill -9 -f "$SB" 2>/dev/null
  rm -rf "$SB"
}
trap cleanup EXIT

# ------------------------------------------------------------------ sandbox

mkdir -p "$SB/data" "$SB/logs" "$SB/serve"
cp "$REPO/corpus_stream_parallel.sh" "$SB/"

# The "Lichess archive": random bytes of a known size.
dd if=/dev/urandom of="$SB/serve/lichess_db_standard_rated_2021-04.pgn.zst" \
   bs=1000 count=$((BLOB_SIZE/1000)) 2>/dev/null
cp "$SB/serve/lichess_db_standard_rated_2021-04.pgn.zst" \
   "$SB/serve/lichess_db_standard_rated_2021-05.pgn.zst"

# Stub preprocessor. Sleeps (so it can be killed mid-run), then emits the
# expected number of game_*.pkl files into --output-dir. Honours two switches
# via the environment so scenarios can force slowness or failure.
cat > "$SB/fake_python.sh" <<'STUB'
#!/usr/bin/env bash
# stands in for "$PYTHON_BIN -u $PREPROCESS_SCRIPT --input X --output-dir Y ..."
outdir=""; maxg=5
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) outdir="$2"; shift 2 ;;
    --max-games)  maxg="$2";  shift 2 ;;
    *) shift ;;
  esac
done
touch "${FAKE_MARKER:-/dev/null}"
sleep "${FAKE_PREPROCESS_SLEEP:-1}"
[ "${FAKE_PREPROCESS_FAIL:-0}" = "1" ] && { echo "stub: forced failure"; exit 1; }
mkdir -p "$outdir"
i=0; while [ "$i" -lt "$maxg" ]; do printf 'x' > "$outdir/game_$(printf '%07d' "$i").pkl"; i=$((i+1)); done
echo "stub: wrote $maxg pkls"
STUB
chmod +x "$SB/fake_python.sh"

cat > "$SB/testconf.sh" <<CONF
MONTHS="2021-04"
expected_size() { case "\$1" in 2021-04|2021-05) echo $BLOB_SIZE ;; *) echo "" ;; esac; }
PROTECTED_MONTHS=""
BASE_URL="http://127.0.0.1:$PORT"
PYTHON_BIN="$SB/fake_python.sh"
PREPROCESS_SCRIPT="unused"
FAST_PREPROCESS_SCRIPT="unused"
MAX_GAMES=5
CONCURRENCY=1
HEARTBEAT_S=1
STALE_MIN=1
DOWNLOAD_ATTEMPTS=3
MAX_ATTEMPTS=2
MAX_STARTS=20
RETRY_SLEEP=2
CONF

export CORPUS_STREAM_TEST_CONFIG="$SB/testconf.sh"

"$PY" "$REPO/analysis/scripts/slow_file_server.py" \
      --root "$SB/serve" --port "$PORT" --bytes-per-sec "$RATE" >/dev/null 2>&1 &
SRV_PID=$!
for _ in $(seq 1 40); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/lichess_db_standard_rated_2021-04.pgn.zst" \
       -r 0-0 && break
  sleep 0.25
done

run_pipeline() { ( cd "$SB" && bash "$SB/corpus_stream_parallel.sh" --host hpc "$@" ); }

# Simulate a power cut: SIGKILL every process belonging to this sandbox, so no
# trap, no cleanup, nothing gets a chance to release its claim. Deliberately
# spares the stand-in Lichess server, since a real captain's PC losing power
# does not take database.lichess.org down with it.
#
# Matching on $SB alone is not enough: macOS resolves /var to /private/var, and
# the worker re-invokes itself through `readlink -f`, so a worker's command line
# holds the PHYSICAL path while $SB holds the logical one. The sandbox basename
# is present in both.
hard_kill() {
  local p cmd
  for p in $(pgrep -f "$(basename "$SB")" 2>/dev/null); do
    cmd=$(ps -o command= -p "$p" 2>/dev/null)
    case "$cmd" in
      *slow_file_server*) : ;;
      *) kill -9 "$p" 2>/dev/null ;;
    esac
  done
  sleep 1
}
reset_month()  { rm -rf "$SB/data/processed_games/2021-04" "$SB/data/.claims/2021-04" \
                        "$SB/data/.state/attempts/2021-04" "$SB/data/.state/starts/2021-04" \
                        "$SB/data/.state/failed/2021-04" \
                        "$SB/data/raw_zst"/*.zst 2>/dev/null; }

# Kill a run once it has provably claimed the month AND made fresh download
# progress, so the interruption lands on real in-flight work.
#
# Two traps here, both of which produced false passes before being handled:
#  * The download RESUMES via HTTP Range across interruptions, so on the second
#    and later calls the partial archive is already large. An absolute size
#    threshold is therefore satisfied instantly and kills the run before it even
#    claims. Progress has to be measured relative to the size on entry.
#  * The claim directory alone is not proof the worker got as far as charging a
#    start. The "claimed (attempt" log line is emitted immediately after, so
#    waiting for a NEW one of those is the reliable signal.
kill_mid_download() {
  local logf="$1" zst="$SB/data/raw_zst/lichess_db_standard_rated_2021-04.pgn.zst"
  local sz start claims0 claims
  start=$(stat -f%z "$zst" 2>/dev/null || stat -c%s "$zst" 2>/dev/null || echo 0)
  # NOTE: `grep -c` prints 0 AND exits 1 when there are no matches, so the usual
  # `|| echo 0` fallback yields the two-line string "0\n0" and every later
  # integer test errors out. Default the empty case instead.
  claims0=$(grep -c ": claimed (attempt" "$logf" 2>/dev/null); claims0=${claims0:-0}

  ( run_pipeline >>"$logf" 2>&1 ) &
  for _ in $(seq 1 160); do
    claims=$(grep -c ": claimed (attempt" "$logf" 2>/dev/null); claims=${claims:-0}
    if [ "$claims" -gt "$claims0" ]; then
      sz=$(stat -f%z "$zst" 2>/dev/null || stat -c%s "$zst" 2>/dev/null || echo 0)
      [ "$sz" -gt $((start + 30000)) ] && break
    fi
    sleep 0.25
  done
  hard_kill
}

counter() { cat "$SB/data/.state/$1/2021-04" 2>/dev/null | tr -dc '0-9'; }

echo "  sandbox: $SB   server port: $PORT"

# ============================================================ 1. mid-download

head_ "Scenario 1: SIGKILL mid-DOWNLOAD, then re-run the same command"
reset_month
: > "$SB/serve/.range_log"
( run_pipeline >"$SB/run1.log" 2>&1 ) &
zst="$SB/data/raw_zst/lichess_db_standard_rated_2021-04.pgn.zst"
for _ in $(seq 1 80); do
  sz=$(stat -f%z "$zst" 2>/dev/null || stat -c%s "$zst" 2>/dev/null || echo 0)
  [ "$sz" -gt 100000 ] && [ "$sz" -lt "$BLOB_SIZE" ] && break
  sleep 0.25
done
partial=$(stat -f%z "$zst" 2>/dev/null || stat -c%s "$zst" 2>/dev/null || echo 0)
if [ "$partial" -gt 100000 ] && [ "$partial" -lt "$BLOB_SIZE" ]; then
  ok "download interrupted mid-flight at ${partial}B of ${BLOB_SIZE}B"
else
  bad "could not catch the download mid-flight (size=$partial)"
fi
hard_kill

[ -d "$SB/data/.claims/2021-04" ] && ok "claim survived the kill (correctly stranded, nothing cleaned up)" \
                                  || bad "claim vanished; SIGKILL should leave it stranded"
[ -f "$SB/data/processed_games/2021-04/.corpus_done" ] && bad "month wrongly marked done" \
                                                       || ok "month correctly not marked done"

run_pipeline >"$SB/run2.log" 2>&1
grep -q "reclaiming 2021-04" "$SB/run2.log" && ok "re-run detected and reclaimed the stranded claim" \
                                            || bad "re-run did NOT reclaim the stranded claim"
[ -f "$SB/data/processed_games/2021-04/.corpus_done" ] && ok "re-run completed the month with no manual cleanup" \
                                                       || bad "re-run did not complete the month"
if grep -q "bytes=" "$SB/serve/.range_log" 2>/dev/null; then
  ok "download RESUMED via HTTP Range ($(grep -c 'bytes=' "$SB/serve/.range_log") range request(s)), partial bytes not discarded"
else
  bad "no Range request seen; the resume path did not engage"
fi

# ========================================================== 2. mid-preprocess

head_ "Scenario 2: SIGKILL mid-PREPROCESS, then re-run the same command"
reset_month
export FAKE_PREPROCESS_SLEEP=25 FAKE_MARKER="$SB/preproc_started"
rm -f "$SB/preproc_started"
( run_pipeline >"$SB/run3.log" 2>&1 ) &
for _ in $(seq 1 120); do [ -f "$SB/preproc_started" ] && break; sleep 0.5; done
[ -f "$SB/preproc_started" ] && ok "preprocess reached and running" || bad "preprocess never started"
hard_kill
unset FAKE_PREPROCESS_SLEEP

[ -d "$SB/data/.claims/2021-04" ] && ok "claim stranded by the mid-preprocess kill" \
                                  || bad "claim not stranded"
staging=$(find "$SB/data/processed_games" -maxdepth 1 -name '2021-04.staging.*' 2>/dev/null | head -1)
[ -n "$staging" ] && ok "half-built output left in a staging dir, not in the real one" \
                  || ok "no staging dir left behind"
[ -d "$SB/data/processed_games/2021-04" ] && bad "partial output leaked into the real output dir" \
                                          || ok "real output dir untouched by the crash"

export FAKE_PREPROCESS_SLEEP=1
run_pipeline >"$SB/run4.log" 2>&1
grep -q "reclaiming 2021-04" "$SB/run4.log" && ok "re-run reclaimed the mid-preprocess claim" \
                                            || bad "re-run did NOT reclaim it"
[ -f "$SB/data/processed_games/2021-04/.corpus_done" ] && ok "re-run completed the month unaided" \
                                                       || bad "re-run did not complete the month"
n=$(ls "$SB/data/processed_games/2021-04"/game_*.pkl 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "5" ] && ok "output complete and correct ($n/5 games)" || bad "output wrong ($n/5 games)"

# ============================================================ 3. retry cap

head_ "Scenario 3: a genuinely broken month stops after MAX_ATTEMPTS, loudly"
reset_month
export FAKE_PREPROCESS_FAIL=1
run_pipeline >"$SB/run5.log" 2>&1
run_pipeline >>"$SB/run5.log" 2>&1
run_pipeline >>"$SB/run5.log" 2>&1
unset FAKE_PREPROCESS_FAIL
[ -f "$SB/data/.state/failed/2021-04" ] && ok "month moved to a terminal FAILED state" \
                                        || bad "month never reached a FAILED state (retry loop?)"
grep -q "TERMINAL FAILED STATE" "$SB/run5.log" && ok "failure is surfaced loudly, not silently" \
                                               || bad "failure was not surfaced loudly"
[ -d "$SB/data/.claims/2021-04" ] && bad "claim left behind after terminal failure" \
                                  || ok "claim released after terminal failure"
run_pipeline >"$SB/run6.log" 2>&1
grep -q "terminal FAILED state, skipping" "$SB/run6.log" && ok "later runs skip the failed month instead of looping on it" \
                                                         || bad "failed month was retried anyway"

# ============================================================ 4. status

head_ "Scenario 4: --status is readable without knowing any log paths"
run_pipeline --status > "$SB/status.txt" 2>&1
grep -qE "done [0-9]+/[0-9]+" "$SB/status.txt" && ok "status reports done/total" || bad "status missing done/total"
grep -q "FAILED" "$SB/status.txt" && ok "status surfaces the FAILED month" || bad "status hides the FAILED month"
sed 's/^/      /' "$SB/status.txt"

# ============================================================ 5. claim lock

head_ "Scenario 5: two concurrent runs never process the same month twice"
reset_month
rm -f "$SB/data/.state/failed/2021-04"
export FAKE_PREPROCESS_SLEEP=3
sed -i.bak 's/^MONTHS=.*/MONTHS="2021-04 2021-05"/' "$SB/testconf.sh"
: > "$SB/claims.log"
( run_pipeline >"$SB/par1.log" 2>&1 ) & P1=$!
( run_pipeline >"$SB/par2.log" 2>&1 ) & P2=$!
# Wait on these two specifically. A bare `wait` would also wait on the stand-in
# file server, which runs until the trap kills it, so the test would never return.
wait "$P1" "$P2"
claimed=$(cat "$SB/par1.log" "$SB/par2.log" | grep -c ": claimed (attempt")
lost=$(cat "$SB/par1.log" "$SB/par2.log" | grep -c "claimed by another worker")
if [ "$claimed" -eq 2 ]; then
  ok "each of the 2 months claimed exactly once across 2 concurrent runs (contended skips: $lost)"
else
  bad "expected 2 claims across both runs, saw $claimed"
fi
for m in 2021-04 2021-05; do
  [ -f "$SB/data/processed_games/$m/.corpus_done" ] && ok "$m completed" || bad "$m did not complete"
done

# ================================================ 6. interruptions are not failures

# Regression test for G3 (analysis/corpus-resilience-review.md). The counter used
# to be charged at claim time, before any work ran, and nothing ever refunded it.
# So MAX_ATTEMPTS interruptions of a perfectly HEALTHY month drove it into a
# terminal FAILED state that needed a human, on a machine whose owner is
# described in CLAUDE.md as "only actually available while the captain's PC is
# on". Interruptions are the steady state there, not the exception.
#
# Scenario 3 above proves a genuinely broken month still stops after
# MAX_ATTEMPTS. This proves the converse, which is the half that was broken:
# interruptions alone must never exhaust the retry budget.

head_ "Scenario 6: repeated interruptions must NOT fail a healthy month"
reset_month
rm -f "$SB/data/.state/failed/2021-04"
export FAKE_PREPROCESS_SLEEP=1
: > "$SB/run7.log"

# Three interruptions against MAX_ATTEMPTS=2. Under the old behaviour the first
# two alone were enough to make the next run terminal.
for _ in 1 2 3; do kill_mid_download "$SB/run7.log"; done

if [ -f "$SB/data/.state/failed/2021-04" ]; then
  bad "3 interruptions drove a healthy month into terminal FAILED (G3 regression)"
else
  ok "3 interruptions did not put the healthy month into a FAILED state"
fi

a=$(counter attempts); a="${a:-0}"
s=$(counter starts);   s="${s:-0}"
if [ "$a" -eq 0 ]; then
  ok "no retry budget consumed by interruptions (attempts=$a)"
else
  bad "interruptions consumed the retry budget (attempts=$a); G3 is not fixed"
fi
if [ "$s" -ge 3 ]; then
  ok "the starts counter recorded the interruptions instead (starts=$s)"
else
  bad "the starts counter did not track the interruptions (starts=$s)"
fi

run_pipeline >>"$SB/run7.log" 2>&1
if [ -f "$SB/data/processed_games/2021-04/.corpus_done" ]; then
  ok "the month still completes on the next clean run, unaided"
else
  bad "the month did not complete after the interruptions"
fi

# ================================================ 7. crash-loop backstop survives

# The naive fix for G3 (just move attempts_inc after the work) would delete the
# protection analysis/corpus-pipeline-hardening.md:113-117 asked for: a month
# that dies instantly and repeatedly must not retry forever. MAX_STARTS keeps
# that, at a much looser cap. Prove it still bites.

head_ "Scenario 7: a month that never reaches a verdict still stops eventually"
reset_month
rm -f "$SB/data/.state/failed/2021-04"
sed -i.bak2 's/^MAX_STARTS=.*/MAX_STARTS=3/' "$SB/testconf.sh"
: > "$SB/run8.log"
for _ in 1 2 3; do kill_mid_download "$SB/run8.log"; done
run_pipeline >>"$SB/run8.log" 2>&1

if [ -f "$SB/data/.state/failed/2021-04" ]; then
  ok "the crash-loop backstop still fires at MAX_STARTS (no infinite retry)"
else
  bad "MAX_STARTS never fired; a genuine crash loop would retry forever"
fi
if grep -q "crash loop" "$SB/run8.log"; then
  ok "the FAILED reason names the crash loop, not a bogus 'exhausted attempts'"
else
  bad "the crash-loop failure was not labelled distinctly"
fi
zstleft=$(ls "$SB/data/raw_zst"/*.zst 2>/dev/null | wc -l | tr -d ' ')
if [ "$zstleft" = "0" ]; then
  ok "the raw archive was reclaimed on the crash-loop FAILED path too"
else
  bad "a terminally failed month kept its raw archive ($zstleft file(s)); disk leak"
fi
sed -i.bak3 's/^MAX_STARTS=.*/MAX_STARTS=20/' "$SB/testconf.sh"

# ============================================================ summary

printf "\n  ------------------------------------------\n"
printf "  %d passed, %d failed\n\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ] || exit 1

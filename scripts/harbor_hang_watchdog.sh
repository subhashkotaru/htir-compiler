#!/usr/bin/env bash
# Harbor hang watchdog.
#
# The terminus-2 agent occasionally enters a non-terminating loop that keeps
# re-dumping its trajectory and calling the LLM for hours, past harbor's own
# 900s timeout, without being cancelled (a known harbor/litellm bug). Left
# alone it burns real API budget and blocks the run's batch from completing.
#
# This watchdog polls running harbor task containers and kills any whose
# trial.log shows the runaway-loop signature (many repeated "Trajectory dumped"
# lines). It uses the loop count -- NOT cpu% or uptime -- so genuinely slow,
# CPU-bound tasks (e.g. make-mips-interpreter at 99% CPU) are never killed.
#
# Usage:  scripts/harbor_hang_watchdog.sh [JOBS_GLOB_ROOT] [LOOP_THRESHOLD] [POLL_SECONDS]
#   JOBS_GLOB_ROOT   dir to search for trial.log files (default: data)
#   LOOP_THRESHOLD   kill if >= this many "Trajectory dumped" lines (default 120)
#   POLL_SECONDS     seconds between scans (default 120)

set -uo pipefail
ROOT="${1:-data}"
THRESHOLD="${2:-120}"
POLL="${3:-120}"

echo "[watchdog] start root=$ROOT threshold=$THRESHOLD poll=${POLL}s $(date)"
while true; do
  # For each running harbor task container, resolve its trial.log by the
  # container name's task id, count the loop signature, kill if runaway.
  docker ps --format '{{.Names}}' 2>/dev/null | while read -r cname; do
    # container names look like:  <task>__<id>__env-main-1
    task_id="${cname%%__env*}"           # <task>__<id>
    short="${task_id##*__}"              # <id>
    log=$(find "$ROOT" -iname 'trial.log' -path "*${short}*" 2>/dev/null | head -1)
    [ -z "$log" ] && continue
    total=$(wc -l < "$log" 2>/dev/null | tr -d ' ')
    loops=$(grep -c 'Trajectory dumped' "$log" 2>/dev/null || echo 0)
    [ "$total" -eq 0 ] && continue
    # Hang signature: the log is (almost) NOTHING but repeated trajectory dumps
    # -- a runaway loop with no real work. A genuinely-working task interleaves
    # other log lines, so its loop:total ratio stays low even at high CPU.
    pure=$(( loops * 100 / total ))
    if [ "$loops" -ge "$THRESHOLD" ] && [ "$pure" -ge 90 ]; then
      echo "[watchdog] KILL $cname (loop_lines=$loops/$total ${pure}% pure) $(date)"
      docker stop "$cname" >/dev/null 2>&1
    fi
  done
  sleep "$POLL"
done

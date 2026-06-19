#!/usr/bin/env bash
set -o pipefail

# Factory config.
WORKERS="${WORKERS:-8}"
MERGE_EVERY_SECS="${MERGE_EVERY_SECS:-2400}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

PY="$REPO/.venv/bin/python"
FACTORY="$REPO/artifacts/factory"
LOGS="$FACTORY/logs"
DRIVER_LOCK="$FACTORY/driver.lock"
DRIVER_PID="$FACTORY/driver.pid"
DRIVER_EVENTS="$FACTORY/events.log"

cd "$REPO" || exit 1
mkdir -p "$FACTORY" "$LOGS"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$DRIVER_EVENTS"
}

if [ -d "$DRIVER_LOCK" ]; then
  old_pid=""
  [ -f "$DRIVER_PID" ] && old_pid="$(cat "$DRIVER_PID" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    log "driver already running pid=$old_pid"
    exit 1
  fi
  rmdir "$DRIVER_LOCK" 2>/dev/null || true
fi
if ! mkdir "$DRIVER_LOCK" 2>/dev/null; then
  log "unable to acquire driver lock"
  exit 1
fi
echo "$$" > "$DRIVER_PID"

PIDS=()
TASKS=()
STOP=0
LAST_MERGE_EPOCH="$(date +%s)"
LAST_SUBMIT_EPOCH=0
LAST_MERGE_STATUS="never"
LAST_SUBMIT_STATUS="never"

cleanup() {
  STOP=1
  log "stopping driver; terminating ${#PIDS[@]} active workers"
  i=0
  while [ "$i" -lt "${#PIDS[@]}" ]; do
    kill "${PIDS[$i]}" 2>/dev/null || true
    i=$((i + 1))
  done
  wait 2>/dev/null || true
  rm -f "$DRIVER_PID"
  rmdir "$DRIVER_LOCK" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

log "driver start repo=$REPO workers=$WORKERS merge_every=${MERGE_EVERY_SECS}s"
"$PY" scripts/factory/build_queue.py >> "$FACTORY/build_queue.log" 2>&1 || log "build_queue failed"
"$PY" scripts/factory/state.py init >> "$FACTORY/state.log" 2>&1 || log "state init failed"
PROMOTIONS_AT_LAST_MERGE="$("$PY" scripts/factory/state.py promotions 2>> "$FACTORY/state.log" || echo 0)"

reap_children() {
  new_pids=()
  new_tasks=()
  i=0
  while [ "$i" -lt "${#PIDS[@]}" ]; do
    pid="${PIDS[$i]}"
    task="${TASKS[$i]}"
    state="$(ps -p "$pid" -o state= 2>/dev/null | tr -d ' ' || true)"
    case "$state" in
      ""|Z*)
        wait "$pid" 2>/dev/null || true
        log "reaped worker pid=$pid task=$task"
        ;;
      *)
        new_pids+=("$pid")
        new_tasks+=("$task")
        ;;
    esac
    i=$((i + 1))
  done
  PIDS=("${new_pids[@]}")
  TASKS=("${new_tasks[@]}")
}

fill_workers() {
  while [ "${#PIDS[@]}" -lt "$WORKERS" ]; do
    slot=$(( ${#PIDS[@]} + 1 ))
    claim="$("$PY" scripts/factory/state.py claim --slot "$slot" --tsv 2>> "$FACTORY/state.log")"
    rc=$?
    if [ "$rc" -ne 0 ] || [ -z "$claim" ]; then
      break
    fi
    set -- $claim
    task="$1"
    hash="$2"
    cost="$3"
    log_file="$LOGS/task$(printf "%03d" "$task").log"
    scripts/factory/run_worker.sh "$task" "$hash" "$cost" >> "$log_file" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    TASKS+=("$task")
    "$PY" scripts/factory/state.py set-pid --task "$task" --pid "$pid" \
      >> "$FACTORY/state.log" 2>&1 || true
    log "launched task$(printf "%03d" "$task") pid=$pid cost=$cost"
  done
}

run_merge_and_submit() {
  promotions_now="$("$PY" scripts/factory/state.py promotions 2>> "$FACTORY/state.log" || echo 0)"
  log "merge start promotions=$promotions_now"
  if [ ! -d artifacts/optimized_pre_merge ] && [ -d artifacts/optimized ]; then
    cp -R artifacts/optimized artifacts/optimized_pre_merge
  fi

  "$PY" scripts/merge_external.py --tasks all --zip --source X=artifacts/handcrafted \
    >> "$FACTORY/merge.log" 2>&1
  merge_rc=$?
  LAST_MERGE_EPOCH="$(date +%s)"
  LAST_MERGE_STATUS="exit-$merge_rc"
  if [ "$merge_rc" -eq 0 ]; then
    "$PY" scripts/factory/submit_factory.py -m "factory: auto merge" \
      >> "$FACTORY/submit.log" 2>&1
    submit_rc=$?
    LAST_SUBMIT_EPOCH="$(date +%s)"
    LAST_SUBMIT_STATUS="exit-$submit_rc"
  else
    LAST_SUBMIT_STATUS="skipped-merge-exit-$merge_rc"
  fi
  PROMOTIONS_AT_LAST_MERGE="$promotions_now"
  log "merge end status=$LAST_MERGE_STATUS submit=$LAST_SUBMIT_STATUS"
}

write_status() {
  touch "$FACTORY/heartbeat"
  "$PY" scripts/factory/state.py status-json \
    --workers "$WORKERS" \
    --driver-pid "$$" \
    --last-merge-epoch "$LAST_MERGE_EPOCH" \
    --last-submit-epoch "$LAST_SUBMIT_EPOCH" \
    --last-merge-status "$LAST_MERGE_STATUS" \
    --last-submit-status "$LAST_SUBMIT_STATUS" \
    --promotions-at-last-merge "$PROMOTIONS_AT_LAST_MERGE" \
    >> "$FACTORY/status-write.log" 2>> "$FACTORY/state.log" || true
}

while [ "$STOP" -eq 0 ]; do
  reap_children
  fill_workers

  now_epoch="$(date +%s)"
  promotions_now="$("$PY" scripts/factory/state.py promotions 2>> "$FACTORY/state.log" || echo 0)"
  promotions_delta=$(( promotions_now - PROMOTIONS_AT_LAST_MERGE ))
  since_merge=$(( now_epoch - LAST_MERGE_EPOCH ))
  if [ "$since_merge" -ge "$MERGE_EVERY_SECS" ] || [ "$promotions_delta" -ge 5 ]; then
    run_merge_and_submit
  fi

  write_status
  sleep 15
done

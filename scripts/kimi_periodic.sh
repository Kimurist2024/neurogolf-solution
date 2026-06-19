#!/usr/bin/env bash
# Periodic Kimi adder: launch ONE new worker every INTERVAL seconds on the next
# best OPEN headroom target (next_periodic_target.py picks it from live costs).
# Workers run non-blocking with their own timeout, so they overlap into a rolling
# pool — this loop only governs the *add* cadence, not worker lifetime.
#
# Must be started from the MAIN session (seatbelt inheritance; see the factory
# Codex sandbox incident). Workers write to scratch_kimi/ and promote via
# try_candidate (monotonic, gold+fresh gated) — safe.
#
# Env: KIMI_INTERVAL (sec between adds, default 1200=20min)
#      KIMI_TIMEOUT  (sec per worker, default 1800)
#      KIMI_LO/KIMI_HI (target cost band, default 5000/30000)
#      KIMI_MAX      (max workers to add, 0=unlimited, default 0)
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
VENV=".venv/bin/python"
ASK_KIMI="$HOME/.claude/bin/ask-kimi"
LOG="artifacts/kimi_logs/periodic.log"
mkdir -p artifacts/kimi_logs
INTERVAL="${KIMI_INTERVAL:-1200}"
export KIMI_TIMEOUT="${KIMI_TIMEOUT:-1800}"
MAX="${KIMI_MAX:-0}"
log(){ echo "[$(date -u '+%H:%M:%SZ')] $*" >> "$LOG"; }

[ -x "$ASK_KIMI" ] || { echo "FATAL: ask-kimi missing" >&2; exit 2; }

killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }

mark(){  # add a task num to a json int-array file
  $VENV - "$1" "$2" <<'PY' 2>/dev/null
import json,sys
f,n=sys.argv[1],int(sys.argv[2])
try: cur=set(json.load(open(f)))
except Exception: cur=set()
cur.add(n); json.dump(sorted(cur),open(f,"w"),indent=2)
PY
}

launch_worker(){  # $1=task $2=hash $3=cost ; backgrounded, own timeout
  local TASK="$1" HASH="$2" COST="$3" T3 pf log_t
  T3=$(printf "%03d" "$TASK")
  mkdir -p "scripts/golf/scratch_kimi/task${T3}"
  pf="$(mktemp "${TMPDIR:-/tmp}/kimi_periodic_${TASK}_XXXXXX")"
  $VENV scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed 's/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; s/\bCodex\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g' > "$pf"
  log_t="artifacts/kimi_logs/task${T3}.log"
  (
    "$ASK_KIMI" < "$pf" > "$log_t" 2>&1 &
    wpid=$!
    ( sleep "$KIMI_TIMEOUT"; killtree "$wpid" ) & kpid=$!
    wait "$wpid" 2>/dev/null; rc=$?
    killtree "$kpid" 2>/dev/null
    rm -f "$pf"
    echo "[$(date -u '+%H:%M:%SZ')] task${T3} worker exit=$rc" >> "$LOG"
  ) &
}

log "================ KIMI periodic adder START (interval=${INTERVAL}s timeout=${KIMI_TIMEOUT}s band=${KIMI_LO:-5000}-${KIMI_HI:-30000} max=${MAX}) ================"
count=0
while :; do
  spec=$($VENV scripts/golf/next_periodic_target.py 2>/dev/null)
  if [ -z "$spec" ]; then
    log "no OPEN headroom target right now; sleeping ${INTERVAL}s"
  else
    IFS=: read -r TASK HASH COST <<< "$spec"
    mark docs/golf/periodic_launched.json "$TASK"
    mark docs/golf/kimi_attempted.json "$TASK"
    launch_worker "$TASK" "$HASH" "$COST"
    count=$((count+1))
    log "ADD #${count}: task$(printf '%03d' "$TASK") (hash $HASH, live cost $COST)"
    if [ "$MAX" -gt 0 ] && [ "$count" -ge "$MAX" ]; then
      log "reached KIMI_MAX=${MAX}; stopping adder (workers keep running)"
      break
    fi
  fi
  sleep "$INTERVAL"
done
log "================ KIMI periodic adder END (added=${count}) ================"
echo "periodic adder ended (added=${count})"

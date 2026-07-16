#!/usr/bin/env bash
# Kimi golf IMPROVEMENT loop — a SEPARATE lane from Codex (別枠).
# Isolation: scratch_kimi/ (own scratch+FAILURE_LOG), kimi_logs/ (own logs),
#   disjoint targets (kimi_exclude.json = factory_done + recent codex waves),
#   own process tree. Shared only the result store handcrafted/ via try_candidate
#   (monotonic + visible-gold gated = safe merge point).
# Each iteration: pick N OPEN highest-REAL-cost targets -> kimi wave -> mark
#   attempted. NO auto-submit (the LB judge is run manually via the real-cost
#   aggressive build). Stops when no OPEN targets remain or MAX_ITERS reached.
# Env: KIMI_BATCH (targets/iter, default 3), KIMI_MAX_ITERS (default 8),
#      KIMI_TIMEOUT (sec/worker, default 1500).
set -u
REPO="/Users/user/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
VENV=".venv/bin/python"
LOG="artifacts/kimi_logs/loop.log"
mkdir -p artifacts/kimi_logs
BATCH="${KIMI_BATCH:-3}"
MAX_ITERS="${KIMI_MAX_ITERS:-8}"
export KIMI_TIMEOUT="${KIMI_TIMEOUT:-1500}"
log(){ echo "[$(date -u '+%H:%M:%SZ')] $*" >> "$LOG"; }

log "================ KIMI improvement loop START (batch=$BATCH max_iters=$MAX_ITERS timeout=${KIMI_TIMEOUT}s) ================"
iter=0
while [ "$iter" -lt "$MAX_ITERS" ]; do
  iter=$((iter+1))

  targets=$($VENV scripts/golf/kimi_targets.py "$BATCH" 2>/dev/null)
  if [ -z "$targets" ]; then log "STOP: no more OPEN Kimi targets."; break; fi
  tasknums=$(echo "$targets" | tr ' ' '\n' | cut -d: -f1 | tr '\n' ' ')
  log "iter $iter targets: $targets"

  # mark attempted BEFORE running so a crash/timeout never re-picks the same set
  $VENV - "$tasknums" <<'PY' 2>/dev/null
import json,sys
nums=[int(x) for x in sys.argv[1].split()]
p="docs/golf/kimi_attempted.json"
cur=set(json.load(open(p))); cur|=set(nums)
json.dump(sorted(cur),open(p,"w"),indent=2)
PY

  # Kimi wave (blocks until all workers finish or per-task timeout)
  bash scripts/kimi_wave.sh $targets >/dev/null 2>&1
  log "iter $iter wave done"

  # record any promotions this iter (try_candidate writes PROMOTED: into task logs)
  for tn in $tasknums; do
    t3=$(printf "%03d" "$tn")
    p=$(grep -c '^PROMOTED:' "artifacts/kimi_logs/task${t3}.log" 2>/dev/null || echo 0)
    last=$(grep '^PROMOTED:' "artifacts/kimi_logs/task${t3}.log" 2>/dev/null | tail -1)
    [ "${p:-0}" -gt 0 ] && log "iter $iter task${t3} PROMOS=$p $last"
  done
done
log "================ KIMI improvement loop END (iter=$iter) ================"
echo "KIMI LOOP ENDED after $iter iterations"

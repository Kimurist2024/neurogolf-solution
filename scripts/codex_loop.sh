#!/usr/bin/env bash
# Fully autonomous Codex golf replenishment loop. Run from the MAIN session.
# Each iteration: quota-gate -> pick 3 OPEN medium-cost targets -> Codex wave
# -> fresh-gate(k=100) + build + promote -> submit. Stops when Codex weekly
# remaining < 20% (codex_quota.py exit 3) or no OPEN targets remain.
# Safe by construction: only adopts cheaper-AND-fresh-0-fail nets (no regression,
# no landmine, public floor preserved).
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
VENV=".venv/bin/python"
KAG="/opt/anaconda3/bin/python"
LOG="artifacts/codex_logs/loop.log"
mkdir -p artifacts/codex_logs
log(){ echo "[$(date -u '+%H:%M:%SZ')] $*" >> "$LOG"; }

log "================ codex autonomous loop START ================"
iter=0
while true; do
  iter=$((iter+1))

  # 1. quota read (informational only; 20% auto-stop DISABLED per user 2026-06-14)
  q=$($VENV scripts/codex_quota.py 2>/dev/null); rc=$?
  log "iter $iter quota: $q"
  # loop now stops only when no OPEN targets remain (see below).

  # 2. pick targets (empty -> no OPEN tasks left)
  targets=$($VENV scripts/golf/next_targets.py 8 2>/dev/null)
  if [ -z "$targets" ]; then log "STOP: no more OPEN targets."; break; fi
  tasknums=$(echo "$targets" | tr ' ' '\n' | cut -d: -f1 | tr '\n' ' ')
  log "iter $iter targets: $targets"
  for tn in $tasknums; do rm -f "artifacts/codex_logs/task$(printf %03d "$tn").log"; done

  # 3. Codex wave (blocks until all workers finish or 15-min timeout each)
  CODEX_TIMEOUT=600 bash scripts/codex_wave.sh $targets >/dev/null 2>&1

  # 4. fresh-gate + build + promote
  res=$($VENV scripts/golf/gate_build.py $tasknums 2>/dev/null); rc=$?
  log "iter $iter gate: $res"
  if [ "$rc" -eq 5 ]; then log "iter $iter: nothing adopted; skip submit."; continue; fi

  # 5. submit
  proj=$(echo "$res" | awk '{print $NF}')
  sub=$($KAG -c "
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
r=api.competition_submit('artifacts/final_submit/submission.zip','codex-loop iter $iter $res','neurogolf-2026')
print(r.get('ref') if isinstance(r,dict) else r)
" 2>/dev/null)
  log "iter $iter SUBMITTED proj=$proj ref=$sub"

  # 6. advance the tracked best score (projections have matched the grader exactly)
  $VENV -c "import json;json.dump({'score':float('$proj'),'run':'codex-loop-iter$iter','message':'$res'},open('artifacts/best_score.json','w'),indent=2)" 2>/dev/null
done
log "================ codex autonomous loop END (iter=$iter) ================"
echo "CODEX LOOP ENDED after $iter iterations"

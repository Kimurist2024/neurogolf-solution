#!/usr/bin/env bash
# Single-worker SEQUENTIAL deep-dive loop. Run from the MAIN session (never a
# daemon) so Codex children do NOT inherit a seatbelt sandbox.
#
# For each task scoring <= THR (hardest/highest-cost first): pin ONE Codex worker
# to it and run repeated rounds until the task crosses THR (PROMOTED) or a floor
# is reached (no_progress/maxrounds). Then fresh-gate-adopt, rebuild the zip,
# submit, record, and move to the next task. No OVERALL time limit; each Codex
# round still has a generous cap so a single hang cannot stall the campaign.
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
VENV=".venv/bin/python"
KAG="/opt/anaconda3/bin/python"
LOG="artifacts/codex_logs/solo.log"
mkdir -p artifacts/codex_logs

# Campaign tunables (also consumed by solo.py via the environment).
export SOLO_K="${SOLO_K:-500}"
export SOLO_MAXROUNDS="${SOLO_MAXROUNDS:-8}"
export SOLO_STUCK="${SOLO_STUCK:-3}"
export SOLO_THR="${SOLO_THR:-15.1}"
export SOLO_AB_THR="${SOLO_AB_THR:-0.05}"   # <= this fresh-fail rate -> A/B candidate
CODEX_TIMEOUT="${CODEX_TIMEOUT:-2400}"   # per-round cap (40 min); not a per-task cap
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"

log(){ echo "[$(date -u '+%H:%M:%SZ')] $*" | tee -a "$LOG"; }

# Preflight: a sandboxed env cannot write $HOME/.codex -> codex exec dies at init.
if ! touch "$HOME/.codex/.solo_probe" 2>/dev/null; then
  log "FATAL: \$HOME/.codex not writable (sandboxed). Launch from the MAIN session."
  exit 2
fi
rm -f "$HOME/.codex/.solo_probe"

# kill a process AND all descendants (codex exec spawns children).
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }

net_up(){ curl -sI --max-time 8 https://chatgpt.com >/dev/null 2>&1; }

wait_for_net(){   # block until codex's backend is reachable again
  local waited=0
  until net_up; do
    log "network DOWN (codex unreachable) — waiting 30s [waited ${waited}s]"
    sleep 30; waited=$((waited+30))
  done
}

# Run ONE codex round. Returns 0 = real round (caller should gate it),
# 1 = VOID round (network error, no work done) -> caller retries WITHOUT gating
# so a transient outage can never burn no_progress into a false floor.
run_codex_round(){   # $1=task $2=hash $3=cost
  local task="$1" hash="$2" cost="$3" t3 pf rlog wpid kpid before new target
  t3=$(printf "%03d" "$task")
  # target cost = the cost at which the task reaches SOLO_THR (cost = e^(25-THR)).
  # Derived from SOLO_THR so the goal text is correct for any cost band, not the
  # hardcoded 19700 that only matched the THR=15.1 high-cost campaign.
  target=$(awk "BEGIN{printf \"%.0f\", exp(25-$SOLO_THR)}")
  wait_for_net
  pf="$(mktemp "${TMPDIR:-/tmp}/solo_prompt_${task}_XXXXXX")"
  $VENV scripts/factory/worker_prompt.py "$task" "$hash" "$cost" > "$pf"
  {
    echo
    echo "Solo deep-dive goal (campaign override)"
    echo "- This task's champion cost is $cost (score $(awk "BEGIN{printf \"%.2f\",25-log($cost)}"))."
    echo "- Drive cost STRICTLY BELOW $target so the task scores >= ${SOLO_THR}. That needs"
    echo "  roughly $(awk "BEGIN{printf \"%.1f\",$cost/$target}")x reduction. If incremental shrinking"
    echo "  stalls, do a ground-up minimal rebuild straight from the generator spec."
    echo "- Promote every strictly-cheaper correct net via try_candidate. The campaign"
    echo "  fresh-gates (k=${SOLO_K}) before adopting, so visible-example overfit is"
    echo "  rejected: compile the spec, never fit examples."
  } >> "$pf"
  rlog="artifacts/codex_logs/task${t3}.log"
  before=$(wc -l < "$rlog" 2>/dev/null || echo 0)
  bash -c "codex exec --sandbox workspace-write --skip-git-repo-check - < '$pf' >> '$rlog' 2>&1" &
  wpid=$!
  ( sleep "$CODEX_TIMEOUT"; killtree "$wpid" ) &
  kpid=$!
  wait "$wpid" 2>/dev/null
  killtree "$kpid" 2>/dev/null
  rm -f "$pf"
  # inspect ONLY the lines this round appended.
  new=$(tail -n +"$((before+1))" "$rlog" 2>/dev/null)
  if echo "$new" | grep -q "PASS score:\|PROMOTED:\|COMPARE optimized"; then
    return 0   # codex did real work -> gate it
  fi
  if echo "$new" | grep -q "failed to connect to websocket\|stream disconnected before completion\|nodename nor servname\|error sending request"; then
    log "task$task round VOID (network error, no work) — not counted, will retry"
    return 1
  fi
  return 0   # no errors but no candidate either = genuine no-progress -> gate it
}

submit(){   # $1=zip $2=msg  -> prints ref on success, NOTHING on failure (e.g. 400)
  $KAG -c "
from kaggle.api.kaggle_api_extended import KaggleApi
try:
    api=KaggleApi(); api.authenticate()
    r=api.competition_submit('$1','$2','neurogolf-2026')
    print(r.get('ref') if isinstance(r,dict) else r)
except Exception:
    pass
" 2>/dev/null
}

# Verify pending <=AB_THR-fresh-fail candidates: submit each in isolation
# (champion + just that net). Kaggle 400 -> the net is structurally invalid,
# auto-drop it; success -> keep it for LB A/B measurement. The safe champion
# line is never touched, so a bad net can neither poison submissions nor lower
# the standing LB (Kaggle keeps the best submission).
ab_verify(){
  [ "$AUTO_SUBMIT" = "1" ] || return 0
  local abtask abzip ref
  for abtask in $($VENV scripts/golf/solo.py ab-pending); do
    wait_for_net
    abzip=$($VENV scripts/golf/solo.py build-ab-one "$abtask" 2>/dev/null)
    [ -z "$abzip" ] && continue
    ref=$(submit "$abzip" "AB-verify task$abtask (<=$(awk "BEGIN{printf \"%.0f\",$SOLO_AB_THR*100}")% fresh-fail)")
    if [ -n "$ref" ]; then
      $VENV scripts/golf/solo.py ab-mark "$abtask" submitted "$ref" >/dev/null 2>&1
      log "AB-SUBMIT task$abtask ref=$ref (check Kaggle LB vs champion to decide promotion)"
    else
      $VENV scripts/golf/solo.py ab-mark "$abtask" dropped >/dev/null 2>&1
      log "AB-DROP task$abtask (Kaggle 400 = structurally invalid net)"
    fi
  done
}

log "============== SOLO deep-dive START (k=$SOLO_K maxrounds=$SOLO_MAXROUNDS stuck=$SOLO_STUCK thr=$SOLO_THR timeout=${CODEX_TIMEOUT}s) =============="
# Idempotent init: if a state file already exists (e.g. pre-seeded + target-pruned
# by the operator), DO NOT re-init or the pruned target queue is clobbered.
if [ -f docs/golf/solo_state.json ]; then
  log "solo_state.json exists -> skip init (using existing pre-pruned target queue)"
else
  $VENV scripts/golf/solo.py init 2>&1 | tee -a "$LOG"
fi

while true; do
  read -r TASK HASH COST < <($VENV scripts/golf/solo.py next)
  if [ -z "${TASK:-}" ]; then log "STOP: no OPEN targets left."; break; fi
  log "---- task $TASK (hash $HASH, champion cost $COST) deep-dive START ----"
  : > "artifacts/codex_logs/task$(printf %03d "$TASK").log"
  while true; do
    if ! run_codex_round "$TASK" "$HASH" "$COST"; then
      # VOID round (network) — do NOT gate (no_progress untouched); wait + retry.
      wait_for_net
      continue
    fi
    out=$($VENV scripts/golf/solo.py gate "$TASK")
    log "gate task$TASK: $out"
    case "$out" in
      TERMINAL*)
        if echo "$out" | grep -q "zip=" && [ "$AUTO_SUBMIT" = "1" ]; then
          zip=$(echo "$out" | sed -n 's/.*zip=\([^ ]*\).*/\1/p')
          proj=$(echo "$out" | sed -n 's/.*proj=\([^ ]*\).*/\1/p')
          ref=$(submit "$zip" "solo deepdive task$TASK -> proj $proj")
          $VENV scripts/golf/solo.py bump-submit >/dev/null 2>&1
          log "SUBMITTED task$TASK ref=$ref proj=$proj"
        fi
        break ;;
      CONTINUE*)
        # task stays pinned; refresh COST in case an adoption lowered it.
        nc=$(echo "$out" | sed -n 's/.*best=\([0-9]*\).*/\1/p')
        [ -n "$nc" ] && COST="$nc" ;;
    esac
  done
  ab_verify   # isolated verify-submit of any <=AB_THR fresh-fail candidates found
  $VENV scripts/golf/solo.py status 2>&1 | tee -a "$LOG"
done
log "============== SOLO deep-dive END =============="
echo "SOLO LOOP ENDED"

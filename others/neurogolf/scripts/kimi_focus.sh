#!/usr/bin/env bash
# Kimi FOCUS campaign: concentrate 8 diverse-approach workers on ONE task per
# iteration, looping rounds until a FULL +1 (cost <= pre/e) is fresh-gated, or a
# round cap is hit (avoids infinite loops on tasks where +1 is structurally
# unreachable). Then bank the best and move on. Auto fresh-gate (k=30) + auto-submit.
# Must be started from the MAIN session.
#
# Env: KIMI_PAR (approaches/round, default 8) KIMI_TIMEOUT (sec, default 1200)
#      KIMI_HOURS (budget, default 4) KIMI_ROUNDS (max rounds/task, default 2)
#      KIMI_THRESHOLD (auto-submit, default 0.5) KIMI_K (fresh-gate, default 30)
set -u
REPO="/Users/user/Downloads/projects/Kaggle/Neurogolf"; cd "$REPO" || exit 1
V=".venv/bin/python"; ASK="$HOME/.claude/bin/ask-kimi"
LOG="artifacts/kimi_logs/focus.log"; mkdir -p artifacts/kimi_logs
PAR="${KIMI_PAR:-8}"; export KIMI_TIMEOUT="${KIMI_TIMEOUT:-1200}"
HOURS="${KIMI_HOURS:-4}"; ROUNDS="${KIMI_ROUNDS:-2}"; THRESH="${KIMI_THRESHOLD:-0.5}"; K="${KIMI_K:-30}"
log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] $*" >> "$LOG"; }
[ -x "$ASK" ] || { echo "FATAL ask-kimi"; exit 2; }
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }
[ -f docs/golf/campaign_best.txt ] || printf 'artifacts/_BEST_6804.22.zip\t6804.22' > docs/golf/campaign_best.txt

# 8 distinct approach directives (worker i gets APPROACH[i])
approach(){ case "$1" in
 0) echo "APPROACH: PORT. Study the sibling net at artifacts/focus_ref.onnx (onnx.load + print nodes) and reuse its cheap structure verbatim where the transform matches.";;
 1) echo "APPROACH: SMALLEST DTYPES. Re-type every temporary to int8/uint8/bool; keep only what truly needs int32/float. Output-as-Equal/paint-onto-input.";;
 2) echo "APPROACH: EARLY CROP. Crop to the generator MAX grid immediately, do all work on the small ROI, Pad back to [1,10,30,30] only at the very end.";;
 3) echo "APPROACH: FP16. Cast big float intermediates to FLOAT16 (keep input float32, margin>=1.0). Halve every large float temporary.";;
 4) echo "APPROACH: FULL REBUILD MINIMAL. Build a brand-new minimal static graph from the generator spec; fewest nodes, fold constants into Conv bias.";;
 5) echo "APPROACH: ALT ALGORITHM. Reformulate: replace CumSum/scan with Conv-stencil or integral-image+Gather, or Chebyshev flood-fill via MaxPool. Different decomposition.";;
 6) echo "APPROACH: FUSION/CSE. Collapse Max/Min/Add/Sum chains, common-subexpression-eliminate duplicate intermediates, delete redundant Cast/Where/Pad/Identity.";;
 *) echo "APPROACH: AGGRESSIVE COMBO. Apply crop + smallest dtypes + FP16 + fusion together; drive toward the few-hundred floor.";;
esac; }

launch_one(){ # $1=task $2=hash $3=cost $4=approach_idx $5=beat_cost(0=none)
  local TASK="$1" HASH="$2" COST="$3" IDX="$4" BEAT="$5"
  local T3=$(printf "%03d" "$TASK"); mkdir -p "scripts/golf/scratch_kimi/task${T3}"
  local pf=$(mktemp /tmp/kimi_focus_${TASK}_${IDX}_XXXXXX)
  $V scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed 's/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; s/\bCodex\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g' > "$pf"
  { echo; echo "===== FOCUS GOAL: drive cost from ${COST} to <= 19930 (score 15.1). Lower is better. ====="
    [ "$BEAT" != 0 ] && echo "Current best net is cost ${BEAT}; you MUST beat it, and the goal is cost <= 19930."
    approach "$IDX"
    echo "Promote via try_candidate. Only a strictly-cheaper, correct, in-margin net is kept."; } >> "$pf"
  local lg="artifacts/kimi_logs/task${T3}.log"
  ( "$ASK" < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$KIMI_TIMEOUT"; killtree "$w" ) & kk=$!; wait "$w" 2>/dev/null; kill -KILL "$kk" 2>/dev/null; rm -f "$pf" ) &
}

hand_cost(){ $V - "$1" <<'PY' 2>/dev/null
import onnx,sys,tempfile
sys.path.insert(0,"scripts"); from lib import scoring
t=int(sys.argv[1]); from pathlib import Path
p=Path(f"artifacts/handcrafted/task{t:03d}.onnx")
if not p.is_file(): print(""); raise SystemExit
with tempfile.TemporaryDirectory() as wd:
    s=scoring.score_and_verify(onnx.load(str(p)),t,wd,label="x",require_correct=True)
print(s["cost"] if s else "")
PY
}

END=$(( $(date +%s) + HOURS*3600 ))
TARGET=19930   # cost for score 15.1 — the crossing goal for every task
log "==== FOCUS START par=$PAR rounds=$ROUNDS timeout=${KIMI_TIMEOUT}s budget=${HOURS}h k=$K goal=score>=15.1(cost<=$TARGET) ===="
[ -f docs/golf/focus_attempted.json ] || echo '[]' > docs/golf/focus_attempted.json
done_tasks=0; passes=0
while [ "$(date +%s)" -lt "$END" ]; do
  SPEC=$($V scripts/golf/focus_target.py 2>/dev/null)
  if [ "$SPEC" = "RESET" ]; then
    passes=$((passes+1)); echo '[]' > docs/golf/focus_attempted.json
    log "pass $passes complete; uncrossed tasks remain -> REVISIT (forever until all >=15.1)"; continue
  fi
  [ -z "$SPEC" ] && { log "ALL TASKS >=15.1 — goal reached, stopping"; break; }
  IFS=: read -r TASK HASH COST TMPL <<< "$SPEC"
  # template ref for the PORT approach (approach 0)
  if [ -n "$TMPL" ] && [ "$TMPL" != "-" ]; then
    TT=$(printf "%03d" "$TMPL")
    cp -f "artifacts/handcrafted/task${TT}.onnx" artifacts/focus_ref.onnx 2>/dev/null || \
      $V -c "import zipfile;open('artifacts/focus_ref.onnx','wb').write(zipfile.ZipFile(open('docs/golf/campaign_best.txt').read().split(chr(9))[0]).read('task${TT}.onnx'))" 2>/dev/null
  fi
  $V - "$TASK" <<'PY'
import json,sys;f="docs/golf/focus_attempted.json"
try: cur=set(json.load(open(f)))
except: cur=set()
cur.add(int(sys.argv[1])); json.dump(sorted(cur),open(f,'w'))
PY
  log "FOCUS task${TASK} cost=${COST} goal<=${TARGET} (need $(( (COST-TARGET)*100/COST ))% cut; tmpl=$TMPL)"
  # SEED handcrafted with the current BEST version so try_candidate competes
  # against the real baseline (external merges may have advanced best beyond the
  # stale handcrafted store), keeping any cheaper pending gain.
  $V scripts/golf/seed_best.py "$TASK" >> "$LOG" 2>/dev/null
  r=0; best=0
  while [ "$r" -lt "$ROUNDS" ]; do
    r=$((r+1))
    for i in $(seq 0 $((PAR-1))); do launch_one "$TASK" "$HASH" "$COST" "$i" "$best"; done
    wait
    best=$(hand_cost "$TASK"); best=${best:-0}
    log "task${TASK} round $r -> best handcrafted cost=${best:-none} (target<$TARGET)"
    if [ -n "$best" ] && [ "$best" != 0 ] && [ "$best" -lt "$TARGET" ]; then
      log "task${TASK} FULL +1 reached at round $r (cost $best)"; break
    fi
  done
  # fresh-gate + bank + maybe auto-submit (campaign_step handles all)
  $V scripts/golf/campaign_step.py "$TASK" --threshold "$THRESH" --k "$K" >> "$LOG" 2>&1
  done_tasks=$((done_tasks+1))
done
log "==== FOCUS END (tasks=$done_tasks) ===="
echo "focus ended after $done_tasks tasks"

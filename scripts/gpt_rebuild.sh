#!/usr/bin/env bash
# GPT (Codex) REBUILD campaign for the cost>5000 tasks. Goal: drive each task's
# cost toward <=1000 via FULL spec-rebuild (NOT node-pruning). SLOT model: keep N
# codex workers busy; the instant one exits, refill its slot with the next
# cheapest-still-over-1000 unassigned target from docs/golf/gpt5000_targets.json.
# Workers promote strictly-cheaper-correct nets via try_candidate -> handcrafted/.
# Harvest (LB-judge merge) is done by the MAIN session. Launch from MAIN session
# only (codex daemon under a codex sandbox dies -- see factory-codex-sandbox-incident).
#
# Env: GR_ENGINE (codex|claude|kimi, default codex) GR_SLOTS (worker slots, default 6)
#      GR_TIMEOUT (sec/worker, default 3000) GR_HOURS (budget, default 8)
#      GR_MODEL (default: codex=gpt-5.5, claude=claude-opus-4-8, kimi=config default)
#      GR_GOAL (default 1000)
#      GR_MODE (rebuild|memshave, default rebuild). memshave = incumbent-preserving
#        memory-footprint golf (dtype narrowing / fusion / const folding); pair it
#        with GR_TARGETS_FILE=docs/golf/mem_targets.json. Hints default to
#        docs/golf/gpt_hints_memshave.md in this mode (override via GR_HINTS).
# Engines: codex -> GPT-5.5 (codex exec), claude -> Opus (claude -p), kimi -> Kimi (kimi -p).
#      All three read the generator, rebuild ground-up, promote via try_candidate, and
#      write scratch to scripts/golf/scratch_<engine>/. Same risk-free harvest path.
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"; cd "$REPO" || exit 1
V=".venv/bin/python"
LOGDIR="${GR_LOGDIR:-artifacts/gpt_rebuild_logs}"; mkdir -p "$LOGDIR"
N="${GR_SLOTS:-6}"; TIMEOUT="${GR_TIMEOUT:-3000}"; HOURS="${GR_HOURS:-8}"
GOAL="${GR_GOAL:-1000}"
ENGINE="${GR_ENGINE:-codex}"   # codex | claude | kimi
MODE="${GR_MODE:-rebuild}"     # rebuild | memshave
case "$MODE" in rebuild|memshave) ;; *) echo "FATAL bad GR_MODE=$MODE (rebuild|memshave)"; exit 2;; esac
case "$ENGINE" in
  codex)  MODEL="${GR_MODEL:-gpt-5.5}";;
  claude) MODEL="${GR_MODEL:-claude-opus-4-8}";;
  kimi)   MODEL="${GR_MODEL:-}";;   # empty -> kimi config.toml default_model
  *) echo "FATAL bad GR_ENGINE=$ENGINE (codex|claude|kimi)"; exit 2;;
esac
SCRATCH="scripts/golf/scratch_${ENGINE}"
LOG="$LOGDIR/gpt_rebuild.log"
log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] $*" >> "$LOG"; }
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }
command -v "$ENGINE" >/dev/null || { echo "FATAL $ENGINE missing"; exit 2; }
[ -f docs/golf/gpt_assigned.json ] || echo '[]' > docs/golf/gpt_assigned.json

build_prompt(){ # $1=task $2=hash $3=cost -> stdout
  local TASK="$1" HASH="$2" COST="$3"; local T3; T3=$(printf "%03d" "$TASK")
  mkdir -p "${SCRATCH}/task${T3}"
  $V scripts/golf/seed_best.py "$TASK" >/dev/null 2>&1
  $V scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed "s#scripts/golf/scratch/task#${SCRATCH}/task#g; s/detached Codex CLI/detached ${ENGINE} worker/g; s/Codex CLI/${ENGINE}/g"
  if [ "$MODE" = memshave ]; then
  cat <<EOF

===== MEMORY-SHAVE GOAL: cut task${T3} cost from ${COST} toward <= ${GOAL}. =====
The incumbent net for task${T3} is ALREADY FUNCTIONALLY CORRECT. Do NOT rebuild
the rule from scratch first. Apply incumbent-preserving MEMORY-FOOTPRINT
transformations only (dtype narrowing in place / node fusion / constant folding
to initializer / early crop) -- full guidance is appended below (MEMORY-SHAVE
MODE). Outputs must stay decision-identical with margin (official decode
wrong=0 on train+test+arc-gen, no raw value in (0,0.25), fresh k=30 zero fail).
ADOPTION: promote ONLY via  ${V} scripts/golf/try_candidate.py --task ${TASK} --onnx PATH
(keeps only strictly-cheaper, correct, in-margin nets -> shave is risk-free).
Keep ALL scratch under ${SCRATCH}/task${T3}/ . Do NOT touch any
orchestration/queue/submission files. LOOP POLICY (memshave): KEEP SHAVING the
same task after each successful adoption; exit (print 'MOVED_ON task${T3}')
only when cost <= ${GOAL}, when two consecutive transformation attempts fail to
produce an adopted strictly-cheaper net, or when the session nears timeout.
EOF
  else
  cat <<EOF

===== HARD STRETCH GOAL: drive task${T3} cost from ${COST} toward <= ${GOAL}. =====
Lightweighting/node-pruning the incumbent is EXHAUSTED and does NOT work here.
The ONLY accepted method is a FULL GROUND-UP REBUILD of a tiny rule-engine ONNX:
  1. IMAGE the generator inputs/arc-gen-repo/tasks/task_${HASH}.py (+ common.py)
     and INFER the exact ARC rule. Visible examples are debug fixtures only.
  2. Write a numpy SOLVER of that rule; verify EXACT match on all train+test+
     arc-gen AND >=1000 fresh generator instances (compile the SPEC, never fit
     examples -- a public-fit net already caused a ~-15pt private incident).
  3. Translate to the SMALLEST static-shape ONNX. DROP whole-grid [1,10,30,30]
     float tensors / Conv / Where / Scatter / Expand over the full grid (#1 cost
     bloat). Work on ROI / coordinates only, crop early, output = input-copy +
     local diff (paint-onto-input). Smallest dtypes. Direct Equal/Add/Slice.
Banned/dangerous ops (grader-reject): Loop/Scan/NonZero/Unique/Script/Function/
Compress/*Sequence*/nested graphs/dynamic shape/sparse_initializer. Avoid TopK if
a TopK-free formulation exists.
ADOPTION: promote ONLY via  ${V} scripts/golf/try_candidate.py --task ${TASK} --onnx PATH
(keeps only strictly-cheaper, correct, in-margin nets -> rebuild is risk-free).
Keep ALL scratch under ${SCRATCH}/task${T3}/ . Do NOT touch any
orchestration/queue/submission files. MOVE-ON POLICY: the moment try_candidate
ADOPTS your FIRST strictly-cheaper correct net for task${T3}, STOP and exit the
session immediately (print 'MOVED_ON task${T3}') -- do NOT keep shrinking it; the
orchestrator will hand your freed slot a fresh task. Only if you cannot achieve ANY
adoption should you keep trying smaller representations until the worker times out.
EOF
  fi
  # optional band-specific extra guidance appended verbatim (GR_HINTS=path);
  # memshave mode defaults to the memshave playbook when GR_HINTS is unset.
  local hints="${GR_HINTS:-}"
  [ -z "$hints" ] && [ "$MODE" = memshave ] && hints="docs/golf/gpt_hints_memshave.md"
  if [ -n "$hints" ] && [ -f "$hints" ]; then cat "$hints"; fi
}

LAST_PID=""
launch(){ # $1=task $2=hash $3=cost
  local TASK="$1" HASH="$2" COST="$3"; local T3; T3=$(printf "%03d" "$TASK")
  local pf; pf=$(mktemp "/tmp/gr_${TASK}_XXXXXX")
  build_prompt "$TASK" "$HASH" "$COST" > "$pf"
  local lg="$LOGDIR/${ENGINE}_task${T3}.log"
  case "$ENGINE" in
    codex)  ( codex exec -m "$MODEL" -s workspace-write --skip-git-repo-check -C "$REPO" - < "$pf" > "$lg" 2>&1 & \
              w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
    claude) ( claude -p --model "$MODEL" --dangerously-skip-permissions --verbose --output-format stream-json < "$pf" > "$lg" 2>&1 & \
              w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
    kimi)   ( cd "$REPO" && kimi ${MODEL:+-m "$MODEL"} -p "$(cat "$pf")" > "$lg" 2>&1 & \
              w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
  esac
  LAST_PID=$!
  log "launch ${ENGINE} task${T3} cost=${COST} pid=${LAST_PID}"
}

next_target(){
  local spec; spec=$($V scripts/golf/gpt_next_target.py 1 2>/dev/null)
  if [ -z "$spec" ]; then
    echo '[]' > docs/golf/gpt_assigned.json          # pass exhausted -> re-attempt
    spec=$($V scripts/golf/gpt_next_target.py 1 2>/dev/null)
  fi
  [ -z "$spec" ] && return 1
  local t="${spec%%:*}"
  $V - "$t" <<'PY' >/dev/null 2>&1
import json,sys
f="docs/golf/gpt_assigned.json"
try: cur=set(json.load(open(f)))
except: cur=set()
cur.add(int(sys.argv[1])); json.dump(sorted(cur),open(f,'w'))
PY
  echo "$spec"
}

refill(){
  local pf="$LOGDIR/.pids"; touch "$pf"
  local tmp; tmp=$(mktemp); local p alive=0
  while read -r p; do
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "$p" >> "$tmp"; alive=$((alive+1)); }
  done < "$pf"
  while [ "$alive" -lt "$N" ]; do
    # quota guard (codex only): stop refilling when codex weekly budget < 20%
    if [ "$ENGINE" = codex ]; then
      $V scripts/codex_quota.py >/dev/null 2>&1 || { log "quota STOP -- not refilling"; break; }
    fi
    local spec; spec=$(next_target) || break
    local T H C; IFS=: read -r T H C <<< "$spec"
    launch "$T" "$H" "$C"; echo "$LAST_PID" >> "$tmp"; alive=$((alive+1))
  done
  mv "$tmp" "$pf"
}

END=$(( $(date +%s) + HOURS*3600 ))
rm -f "$LOGDIR/.pids"
log "==== REBUILD START engine=${ENGINE} mode=${MODE} slots=${N} model=${MODEL} timeout=${TIMEOUT}s budget=${HOURS}h goal<=${GOAL} targets=${GR_TARGETS_FILE:-default} ===="
while [ "$(date +%s)" -lt "$END" ]; do
  refill
  sleep 20
done
log "==== GPT REBUILD END ===="
echo "gpt_rebuild ended"

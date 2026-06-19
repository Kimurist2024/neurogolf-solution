#!/usr/bin/env bash
# 12-engine specialization campaign (SLOT model): Codex x4 + Kimi x4 + Claude x4.
# Goal: drive every task's cost STRICTLY BELOW 8000 (score >= 16.01).
# SLOT model: each engine keeps N workers busy; the instant a worker exits, its
# slot is refilled with the next cheapest-over-8000 unassigned task (no wave
# barrier -> fast engines never idle). Workers promote cheaper-correct nets via
# try_candidate -> artifacts/handcrafted/. Harvest (fresh-gate + submit) is done
# by the main session. Must be started from the MAIN session.
#
# Env: SP_CODEX/SP_KIMI/SP_CLAUDE (slots/engine, default 4 each)
#      SP_TIMEOUT (sec/worker, default 2400) SP_HOURS (budget, default 8)
#      SP_CODEX_MODEL (default gpt-5.5) SP_CLAUDE_MODEL (default claude-opus-4-8)
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"; cd "$REPO" || exit 1
V=".venv/bin/python"; ASK="$HOME/.claude/bin/ask-kimi"
LOGDIR="artifacts/specialize_logs"; mkdir -p "$LOGDIR"
GOAL=8000
NC="${SP_CODEX:-4}"; NK="${SP_KIMI:-4}"; NL="${SP_CLAUDE:-4}"
TIMEOUT="${SP_TIMEOUT:-2400}"; export KIMI_TIMEOUT="$TIMEOUT"
HOURS="${SP_HOURS:-8}"
CODEX_MODEL="${SP_CODEX_MODEL:-gpt-5.5}"; CLAUDE_MODEL="${SP_CLAUDE_MODEL:-claude-opus-4-8}"
LOG="$LOGDIR/specialize.log"
log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] $*" >> "$LOG"; }
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }
[ -x "$ASK" ] || { echo "FATAL ask-kimi"; exit 2; }
command -v codex >/dev/null || { echo "FATAL codex"; exit 2; }
command -v claude >/dev/null || { echo "FATAL claude"; exit 2; }
[ -f docs/golf/specialize_assigned.json ] || echo '[]' > docs/golf/specialize_assigned.json

build_prompt(){ # $1=task $2=hash $3=cost $4=engine  -> stdout
  local TASK="$1" HASH="$2" COST="$3" ENG="$4"; local T3; T3=$(printf "%03d" "$TASK")
  mkdir -p "scripts/golf/scratch_${ENG}/task${T3}"
  $V scripts/golf/seed_best.py "$TASK" >/dev/null 2>&1
  $V scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed "s#scripts/golf/scratch/task#scripts/golf/scratch_${ENG}/task#g; s/detached Codex CLI/detached ${ENG} worker/g; s/Codex CLI/${ENG}/g; s/\bCodex\b/${ENG}/g"
  cat <<EOF

===== HARD GOAL: drive task${T3} cost STRICTLY BELOW ${GOAL} (score >= 16.01). Current ${COST}. =====
This is a SPECIALIZATION pass. Generic node-pruning is exhausted; the PROVEN
winning method is to REBUILD a tiny rule-engine ONNX, not shave the old net:

WINNING METHOD (follow this order):
  1. IMAGE the data: render train/test/arc-gen input->output grids and INFER the
     ARC rule by eye (color map, bbox move, template copy, landmark relocate,
     symmetry/flip, mode/rare-color, background removal, input-copy+local-diff).
  2. Write a small numpy/Python SOLVER of that rule. Verify EXACT match on ALL
     visible train+test+arc-gen, then >=1000 fresh generator instances.
  3. Translate to the SMALLEST static-shape ONNX.

COST is dominated by REPRESENTATION SIZE, not correctness logic. To get < ${GOAL}:
  - DROP whole-grid tensors. Never push big [1,10,30,30] float masks / Conv /
    Where / Scatter / Expand over the full grid -- that is the #1 cost bloat.
  - Work on ROI / COORDINATES only: target bbox, landmark colors, the few cells
    that change. Crop early; pad back at the very end.
  - Prefer OUTPUT = input-copy + local diff (paint-onto-input) over building the
    whole output. Reduce: target-cell count, candidate coords (fixed small ints),
    color branches, big constant tensors. Direct Equal/Add/Slice beats Conv.
  - task018-style: read mode color / a few landmarks / template->copy relative
    coords; Scatter ONLY the needed cells (keep Scatter count tiny).
AVOID (cost bloat or grader-reject): full-coord generation, all-cell compare,
all-candidate Scatter, per-color huge masks, 30x30 Where chains, big constants,
and banned/dangerous ops: Loop/Scan/NonZero/Unique/Script/Function/Compress/
*Sequence*/nested graphs/dynamic shape/sparse_initializer/huge Expand-Tile-Gather.
TopK: avoid if a TopK-free formulation exists.

ADOPTION: visible/public gold must match EXACTLY (all train+test+arc-gen). Promote
ONLY via:  ${V} scripts/golf/try_candidate.py --task ${TASK} --onnx PATH
Keep ALL scratch under scripts/golf/scratch_${ENG}/task${T3}/ . try_candidate keeps
only strictly-cheaper, correct, in-margin nets, so rebuilds are risk-free.
EOF
}

LAST_PID=""
launch(){ # $1=task $2=hash $3=cost $4=engine ; sets LAST_PID to the worker wrapper pid
  local TASK="$1" HASH="$2" COST="$3" ENG="$4"; local T3; T3=$(printf "%03d" "$TASK")
  local pf; pf=$(mktemp "/tmp/sp_${ENG}_${TASK}_XXXXXX")
  build_prompt "$TASK" "$HASH" "$COST" "$ENG" > "$pf"
  local lg="$LOGDIR/${ENG}_task${T3}.log"
  case "$ENG" in
    codex)  ( codex exec -m "$CODEX_MODEL" -s workspace-write --skip-git-repo-check -C "$REPO" - < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
    kimi)   ( "$ASK" < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
    claude) ( claude -p --model "$CLAUDE_MODEL" --dangerously-skip-permissions < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) & ;;
  esac
  LAST_PID=$!
  log "launch ${ENG} task${T3} cost=${COST} pid=${LAST_PID}"
}

# next_target: echo "task:hash:cost" for the cheapest unassigned >GOAL task and
# mark it assigned; empty if none. Resets assigned when the pass is exhausted.
next_target(){
  local spec; spec=$($V scripts/golf/specialize_targets.py 1 2>/dev/null)
  if [ -z "$spec" ]; then
    echo '[]' > docs/golf/specialize_assigned.json
    spec=$($V scripts/golf/specialize_targets.py 1 2>/dev/null)
  fi
  [ -z "$spec" ] && return 1
  local t="${spec%%:*}"
  $V - "$t" <<'PY' >/dev/null 2>&1
import json,sys
f="docs/golf/specialize_assigned.json"
try: cur=set(json.load(open(f)))
except: cur=set()
cur.add(int(sys.argv[1])); json.dump(sorted(cur),open(f,'w'))
PY
  echo "$spec"
}

refill(){ # $1=engine $2=N  ; file-based PID tracking (bash 3.2 safe, no arrays)
  local ENG="$1" N="$2"
  local pf="$LOGDIR/.pids_${ENG}"
  touch "$pf"
  local tmp; tmp=$(mktemp); local p alive=0
  while read -r p; do
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && { echo "$p" >> "$tmp"; alive=$((alive+1)); }
  done < "$pf"
  while [ "$alive" -lt "$N" ]; do
    local spec; spec=$(next_target) || break
    local T H C; IFS=: read -r T H C <<< "$spec"
    launch "$T" "$H" "$C" "$ENG"; echo "$LAST_PID" >> "$tmp"; alive=$((alive+1))
  done
  mv "$tmp" "$pf"
}

END=$(( $(date +%s) + HOURS*3600 ))
rm -f "$LOGDIR"/.pids_*
log "==== SPECIALIZE START (SLOT) codex=${NC} kimi=${NK} claude=${NL} timeout=${TIMEOUT}s budget=${HOURS}h goal<${GOAL} ===="
while [ "$(date +%s)" -lt "$END" ]; do
  refill codex "$NC"
  refill kimi "$NK"
  refill claude "$NL"
  sleep 20
done
log "==== SPECIALIZE END ===="
echo "specialize ended"

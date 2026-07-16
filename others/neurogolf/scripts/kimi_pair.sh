#!/usr/bin/env bash
# Kimi PAIR campaign: directed handcrafted attack on the genuine
# "magnification-reconstruction" archetype pair task101 <-> task133.
# Both: base sprite (idx=0) fully visible in input, magnified (bmag) copies
# placed elsewhere with most pixels hidden, output reconstructs full copies.
# Shared cheap structure: output is a SUPERSET of input (paint-onto-input) and
# the core graph = detect base sprite -> bmag block-expand -> stamp at offsets.
# Build the core on task101 (lead), port to task133 (sibling).
#
# One wave, N workers per task, detached with per-worker timeout. Re-runnable.
# Must be started from the MAIN session. Does NOT touch kimi_focus or queue state.
#
# Env: KP_T101 (workers on 101, default 3) KP_T133 (workers on 133, default 2)
#      KIMI_TIMEOUT (sec, default 1500)
set -u
REPO="/Users/user/Downloads/projects/Kaggle/Neurogolf"; cd "$REPO" || exit 1
V=".venv/bin/python"; ASK="$HOME/.claude/bin/ask-kimi"
mkdir -p artifacts/kimi_logs
N101="${KP_T101:-3}"; N133="${KP_T133:-2}"; export KIMI_TIMEOUT="${KIMI_TIMEOUT:-1500}"
LOG="artifacts/kimi_logs/pair.log"
log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] $*" >> "$LOG"; }
[ -x "$ASK" ] || { echo "FATAL ask-kimi missing"; exit 2; }
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }

# Shared archetype brief appended to every worker (both tasks).
shared_brief(){ cat <<'EOF'

===== ARCHETYPE: MAGNIFICATION-RECONSTRUCTION (task101 & task133 share this) =====
Read both generators to see the shared structure:
  - inputs/arc-gen-repo/tasks/task_447fd412.py  (task101)
  - inputs/arc-gen-repo/tasks/task_57aa92db.py  (task133)

Shared transform (verify from spec, do NOT fit examples):
  1. A base "continuous_creature" sprite is placed at idx=0 with magnifier bmag=1
     and is FULLY visible in the input. This is the shape key.
  2. Additional copies (idx>0) are the SAME base sprite, block-magnified by an
     integer bmag (each base pixel -> a bmag x bmag solid block), placed at
     (brow,bcol). In the INPUT most of each magnified copy is HIDDEN; only a few
     blocks are shown (task101: the red pixel blocks; task133: the signature
     pixel0 block + one 'show' pixel block).
  3. The OUTPUT reconstructs every magnified copy in full.

Two cost levers that make this cheap:
  A. OUTPUT IS A SUPERSET OF INPUT. Every on-cell of the input is an on-cell of
     the output. So build output = paint-onto-input: keep the input one-hot and
     only COMPUTE THE HIDDEN CELLS to add. Never recompute what is already given.
  B. bmag block-magnification is a STATIC stencil: detect the base sprite shape
     from idx=0, infer each copy's bmag (= size of its visible solid blocks) and
     offset, then block-expand via ConvTranspose / Resize-with-static-scale /
     a Conv stencil, and stamp. Carry per-channel int8/uint8/bool [1,1,H,W]
     masks, not full [1,10,30,30] float32, through the pipeline. Crop early.

Cost is dominated by intermediate tensor MEMORY, not file size. The incumbents
are ~75k-90k cost because they push big float32 tensors; the win is smaller
dtypes + paint-onto-input + a single block-expand stencil.

CROSS-PORT: task101 is the lead. Once a correct cheap core promotes on task101,
the SAME graph structure ports to task133 (only the visible-cue detection and
the per-copy color/pcolor assignment differ). In your REPORT.md, note explicitly
which parts of the core graph are reusable for the sibling task.
EOF
}

approach(){ case "$1" in
 0) echo "APPROACH: CONVTRANSPOSE BLOCK-EXPAND. Express bmag magnification as ConvTranspose (or Resize with static integer scale) on a per-channel int8 mask. Detect base sprite from idx=0, stamp expanded copies at inferred offsets. paint-onto-input for output.";;
 1) echo "APPROACH: INTEGRAL-IMAGE DETECT. Use CumSum integral images + corner Gather to find each visible block's size (=bmag) and offset cheaply, then a single Conv stencil to block-expand and stamp. Smallest dtypes throughout.";;
 2) echo "APPROACH: FULL MINIMAL REBUILD. Ignore the incumbent. Build a brand-new minimal static graph straight from the generator spec: per-channel bool masks, early crop to generator-max grid, output-as-paint-onto-input (OR the reconstructed hidden blocks into the input), Pad back at the end.";;
 3) echo "APPROACH: STENCIL FUSION. Collapse the detect/expand/stamp chain into the fewest nodes; common-subexpression-eliminate duplicate intermediates; delete redundant Cast/Where/Pad/Identity; keep only one large tensor alive at a time.";;
 *) echo "APPROACH: AGGRESSIVE COMBO. Crop + smallest dtypes + ConvTranspose block-expand + paint-onto-input + fusion together; drive toward the few-thousand floor.";;
esac; }

launch_one(){ # $1=task $2=hash $3=cost $4=approach_idx
  local TASK="$1" HASH="$2" COST="$3" IDX="$4"
  local T3; T3=$(printf "%03d" "$TASK"); mkdir -p "scripts/golf/scratch_kimi/task${T3}"
  local pf; pf=$(mktemp "/tmp/kimi_pair_${TASK}_${IDX}_XXXXXX")
  $V scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed 's/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; s/\bCodex\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g' > "$pf"
  { shared_brief
    echo
    echo "===== PAIR GOAL: drive task${T3} cost from ${COST} toward <= 19930 (score 15.1) and lower. ====="
    approach "$IDX"
    echo "Promote via: $V scripts/golf/try_candidate.py --task ${TASK} --onnx PATH"
    echo "Only a strictly-cheaper, correct, in-margin net is kept. Do at least one full ground-up rebuild before concluding."; } >> "$pf"
  local lg="artifacts/kimi_logs/pair_task${T3}_w${IDX}.log"
  ( "$ASK" < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$KIMI_TIMEOUT"; killtree "$w" ) & kk=$!; wait "$w" 2>/dev/null; kill -KILL "$kk" 2>/dev/null; rm -f "$pf" ) &
  log "launched task${TASK} approach=${IDX} -> ${lg}"
}

log "==== PAIR START 101x${N101} 133x${N133} timeout=${KIMI_TIMEOUT}s goal<=19930 ===="
i=0; while [ "$i" -lt "$N101" ]; do launch_one 101 447fd412 75581 "$i"; i=$((i+1)); done
i=0; while [ "$i" -lt "$N133" ]; do launch_one 133 57aa92db 89925 "$i"; i=$((i+1)); done
log "all ${N101}+${N133} workers launched (detached)"
echo "PAIR launched: task101 x${N101}, task133 x${N133}. Logs: artifacts/kimi_logs/pair_task*.log ; summary: $LOG"

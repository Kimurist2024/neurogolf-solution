#!/usr/bin/env bash
# Kimi comprehensive campaign: mode-routed waves over the achievable band
# (5k-25k), auto fresh-gate + auto-submit. Skips the proven-0-yield monsters.
# Must be started from the MAIN session (seatbelt; see factory Codex incident).
#
# Each cycle: campaign_targets.py picks N mode-routed targets -> launch N Kimi
# workers (port: sibling-template injected; shrink: shrink-the-existing-net) ->
# wait -> campaign_step.py fresh-gates promotions, banks ADOPTs, auto-submits
# when pending gain >= THRESHOLD (updates _BEST on improvement). Repeat until the
# band is exhausted, DEADLINE passes, or DRY consecutive empty cycles.
#
# Env: KIMI_PAR (workers/wave, default 8) KIMI_TIMEOUT (sec/worker, default 1200)
#      KIMI_HOURS (wall budget, default 4) KIMI_THRESHOLD (auto-submit, default 0.5)
set -u
REPO="/Users/kimura2003/Downloads/projects/Kaggle/Neurogolf"; cd "$REPO" || exit 1
V=".venv/bin/python"; ASK="$HOME/.claude/bin/ask-kimi"
LOG="artifacts/kimi_logs/campaign.log"; mkdir -p artifacts/kimi_logs artifacts/campaign_stage
PAR="${KIMI_PAR:-8}"; export KIMI_TIMEOUT="${KIMI_TIMEOUT:-1200}"
HOURS="${KIMI_HOURS:-4}"; THRESH="${KIMI_THRESHOLD:-0.5}"
DEADLINE_FILE="artifacts/kimi_logs/.campaign_deadline"
log(){ echo "[$(date -u '+%m-%d %H:%M:%SZ')] $*" >> "$LOG"; }
[ -x "$ASK" ] || { echo "FATAL ask-kimi missing"; exit 2; }
killtree(){ local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done; kill -KILL "$p" 2>/dev/null; }

# init best pointer if absent
[ -f docs/golf/campaign_best.txt ] || printf 'artifacts/_BEST_6795.72.zip\t6795.72' > docs/golf/campaign_best.txt
[ -f docs/golf/campaign_banked.json ] || echo '{}' > docs/golf/campaign_banked.json

launch(){  # $1=spec task:hash:cost:mode:template
  local spec="$1"; IFS=: read -r TASK HASH COST MODE TMPL <<< "$spec"
  local T3=$(printf "%03d" "$TASK"); mkdir -p "scripts/golf/scratch_kimi/task${T3}"
  local pf=$(mktemp /tmp/kimi_camp_${TASK}_XXXXXX)
  $V scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed 's/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; s/\bCodex\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g' > "$pf"
  if [ "$MODE" = "port" ]; then
    local TT=$(printf "%03d" "$TMPL")
    cp -f "artifacts/handcrafted/task${TT}.onnx" "artifacts/campaign_ref_task${TT}.onnx" 2>/dev/null || \
      $V -c "import zipfile;open('artifacts/campaign_ref_task${TT}.onnx','wb').write(zipfile.ZipFile(open(open('docs/golf/campaign_best.txt').read().split(chr(9))[0],'rb')).read('task${TT}.onnx'))" 2>/dev/null
    cat >> "$pf" <<HINT

============= ARCHETYPE-PORT (cost ${COST}; target < $((COST/3)) for +1) =============
This task shares a generator archetype with a MUCH cheaper SOLVED sibling task${TMPL}
(cost ${TMPL} net at artifacts/campaign_ref_task${TT}.onnx). Inspect it
(onnx.load + print nodes) and PORT its cheap technique. This task already has a
correct net but is over-built; do not re-solve from zero — reuse the sibling's
cheap structure (smallest dtypes, Slice/Pad/Conv instead of CumSum/full-grid).
Stay spec-derived. fresh-gate + margin>=1.0 as usual.
=================================================================================
HINT
  else
    cat >> "$pf" <<HINT

============= SHRINK MODE (cost ${COST}; target < $((COST/3)) for +1) =============
This task ALREADY has a correct net at cost ${COST}; it is OVER-BUILT. Your job is
to make the SAME function CHEAPER (target a 2.7x cut for +1), NOT to re-solve from
zero. Apply the proven cost cuts: smallest dtypes (int8/uint8/bool) for temporaries,
crop early to the generator max grid then Pad back, fold color constants into Conv
bias, assemble channels with Where, remove redundant Cast/Pad/Where, FLOAT16 big
float intermediates. fresh-gate + margin>=1.0 as usual.
=================================================================================
HINT
  fi
  local lg="artifacts/kimi_logs/task${T3}.log"
  ( "$ASK" < "$pf" > "$lg" 2>&1 & w=$!; ( sleep "$KIMI_TIMEOUT"; killtree "$w" ) & k=$!; wait "$w" 2>/dev/null; kill -KILL "$k" 2>/dev/null; rm -f "$pf" ) &
}

# compute deadline (epoch) without Date.now in python: use shell date
END=$(( $(date +%s) + HOURS*3600 ))
log "==== CAMPAIGN START par=$PAR timeout=${KIMI_TIMEOUT}s budget=${HOURS}h threshold=$THRESH ===="
cycle=0; dry=0
while [ "$(date +%s)" -lt "$END" ] && [ "$dry" -lt 2 ]; do
  cycle=$((cycle+1))
  SPECS=$($V scripts/golf/campaign_targets.py "$PAR" 2>/dev/null)
  if [ -z "$SPECS" ]; then dry=$((dry+1)); log "cycle $cycle: no targets (dry=$dry)"; sleep 30; continue; fi
  dry=0
  NUMS=$(echo "$SPECS" | tr ' ' '\n' | cut -d: -f1 | paste -sd, -)
  $V - "$NUMS" <<'PY'
import json,sys;f="docs/golf/campaign_attempted.json"
try: cur=set(json.load(open(f)))
except: cur=set()
cur|={int(x) for x in sys.argv[1].split(',') if x}; json.dump(sorted(cur),open(f,'w'))
PY
  log "cycle $cycle launch: $SPECS"
  for spec in $SPECS; do launch "$spec"; done
  wait
  log "cycle $cycle wave done; fresh-gating + maybe submit"
  $V scripts/golf/campaign_step.py "$NUMS" --threshold "$THRESH" --k "${KIMI_K:-30}" >> "$LOG" 2>&1
done
log "==== CAMPAIGN END (cycles=$cycle) ===="
echo "campaign ended after $cycle cycles"

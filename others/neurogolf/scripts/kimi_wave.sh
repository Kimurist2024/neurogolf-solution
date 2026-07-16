#!/usr/bin/env bash
# Kimi golf wave launcher (Codex 代替). Run from the MAIN session.
# Each worker: worker_prompt.py <task> <hash> <cost>  (Codex->Kimi 表記置換)
#   piped to ~/.claude/bin/ask-kimi (= kimi -p, 非対話・ツール実行=ファイル編集まで自動承認),
#   wrapped in a per-task timeout to cap runaway token burn.
# Args: one or more  TASK:HASH:COST  triples (3 推奨 = 3 セッション).
# Env: KIMI_TIMEOUT (seconds, default 900).
set -u
REPO="/Users/user/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
mkdir -p artifacts/kimi_logs
TIMEOUT="${KIMI_TIMEOUT:-480}"
ASK_KIMI="$HOME/.claude/bin/ask-kimi"

if [ ! -x "$ASK_KIMI" ]; then
  echo "FATAL: ask-kimi not found/executable at $ASK_KIMI" >&2
  exit 2
fi

# kill a process AND all its descendants (kimi spawns a detached-looking
# "kimi-code" child that a plain SIGALRM on the wrapper does NOT reap).
killtree() {
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done
  kill -KILL "$p" 2>/dev/null
}

echo "[$(date -u '+%H:%M:%SZ')] kimi wave start (timeout=${TIMEOUT}s): $*" > artifacts/kimi_logs/wave.log
for spec in "$@"; do
  IFS=: read -r TASK HASH COST <<< "$spec"
  T3=$(printf "%03d" "$TASK")
  pf="$(mktemp "${TMPDIR:-/tmp}/kimi_prompt_${TASK}_XXXXXX")"
  # 同じワーカープロンプトを使うが、(1) 自己認識を Codex->Kimi に置換し、
  # (2) scratch を scratch_kimi/ にリダイレクトして Codex の scratch/FAILURE_LOG/
  #     候補onnx と物理分離する(別枠化=汚染遮断)。結果ストア handcrafted/ と
  #     try_candidate は共有のまま(安全な合流点)。
  mkdir -p "scripts/golf/scratch_kimi/task${T3}"
  .venv/bin/python scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" \
    | sed 's/detached Codex CLI/detached Kimi/g; s/Codex CLI/Kimi/g; s/\bCodex\b/Kimi/g; s#scripts/golf/scratch/task#scripts/golf/scratch_kimi/task#g' > "$pf"
  log="artifacts/kimi_logs/task${T3}.log"
  (
    # Run the worker in the background and reap its WHOLE tree on timeout.
    "$ASK_KIMI" < "$pf" > "$log" 2>&1 &
    wpid=$!
    ( sleep "$TIMEOUT"; killtree "$wpid" ) &
    kpid=$!
    wait "$wpid" 2>/dev/null; rc=$?
    killtree "$kpid" 2>/dev/null   # worker finished first -> cancel the killer
    rm -f "$pf"
    [ "$rc" -ge 128 ] && note="TIMEOUT/killed(rc=$rc)" || note="exit=$rc"
    echo "[$(date -u '+%H:%M:%SZ')] task${T3} kimi_$note" >> artifacts/kimi_logs/wave.log
  ) &
done
wait
echo "[$(date -u '+%H:%M:%SZ')] ALL KIMI WORKERS DONE" >> artifacts/kimi_logs/wave.log
echo "ALL KIMI WORKERS DONE"

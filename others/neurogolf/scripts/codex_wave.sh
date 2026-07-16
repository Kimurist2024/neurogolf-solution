#!/usr/bin/env bash
# Codex golf wave launcher. Run from the MAIN session (NOT a daemon) so the
# Codex children do NOT inherit a seatbelt sandbox (factory-codex-sandbox-incident).
# Each worker: worker_prompt.py <task> <hash> <cost> | codex exec (workspace-write),
# wrapped in a per-task timeout (default 900s) to cap runaway quota burn.
# Args: one or more  TASK:HASH:COST  triples. Keep the count small (2-3) to limit
# simultaneous quota consumption. Env: CODEX_TIMEOUT (seconds, default 900).
set -u
REPO="/Users/user/Downloads/projects/Kaggle/Neurogolf"
cd "$REPO" || exit 1
mkdir -p artifacts/codex_logs
TIMEOUT="${CODEX_TIMEOUT:-480}"

# Preflight: a sandboxed env cannot write $HOME/.codex -> codex exec dies at init.
if ! touch "$HOME/.codex/.codex_wave_probe" 2>/dev/null; then
  echo "FATAL: cannot write \$HOME/.codex (sandboxed); aborting Codex wave." >&2
  exit 2
fi
rm -f "$HOME/.codex/.codex_wave_probe"

# kill a process AND all descendants. codex exec spawns children that a plain
# SIGALRM on the wrapper leaves orphaned; killtree reaps the whole tree.
killtree() {
  local p="$1" c
  for c in $(pgrep -P "$p" 2>/dev/null); do killtree "$c"; done
  kill -KILL "$p" 2>/dev/null
}

echo "[$(date -u '+%H:%M:%SZ')] codex wave start (timeout=${TIMEOUT}s): $*" > artifacts/codex_logs/wave.log
for spec in "$@"; do
  IFS=: read -r TASK HASH COST <<< "$spec"
  T3=$(printf "%03d" "$TASK")
  pf="$(mktemp "${TMPDIR:-/tmp}/codex_prompt_${TASK}_XXXXXX")"
  .venv/bin/python scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" > "$pf"
  log="artifacts/codex_logs/task${T3}.log"
  (
    # Run codex in the background; reap its WHOLE tree on timeout (no orphans).
    bash -c "codex exec --sandbox workspace-write --skip-git-repo-check - < '$pf' > '$log' 2>&1" &
    wpid=$!
    ( sleep "$TIMEOUT"; killtree "$wpid" ) &
    kpid=$!
    wait "$wpid" 2>/dev/null; rc=$?
    killtree "$kpid" 2>/dev/null   # codex finished first -> cancel the killer
    rm -f "$pf"
    [ "$rc" -ge 128 ] && note="TIMEOUT/killed(rc=$rc)" || note="exit=$rc"
    echo "[$(date -u '+%H:%M:%SZ')] task${T3} codex_$note" >> artifacts/codex_logs/wave.log
  ) &
done
wait
echo "[$(date -u '+%H:%M:%SZ')] ALL CODEX WORKERS DONE" >> artifacts/codex_logs/wave.log
echo "ALL CODEX WORKERS DONE"

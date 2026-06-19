#!/usr/bin/env bash
set -u -o pipefail

TASK="${1:?task required}"
HASH="${2:?hash required}"
COST="${3:?cost required}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
FACTORY="$REPO/artifacts/factory"
RESULTS="$FACTORY/results.log"
HAND="$REPO/artifacts/handcrafted/task$(printf "%03d" "$TASK").onnx"

cd "$REPO" || exit 1
mkdir -p "$FACTORY" "$FACTORY/logs"

before_sig=""
if [ -f "$HAND" ]; then
  before_sig="$(shasum -a 256 "$HAND" 2>/dev/null | awk '{print $1}' || true)"
fi

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] worker start task$(printf "%03d" "$TASK") hash=$HASH cost=$COST"

# Preflight probe: a seatbelt-inherited environment cannot write $HOME/.codex,
# which kills `codex exec` at init (factory-codex-sandbox-incident, 2026-06-13).
# Release the claim (no task consumed) and pause the whole factory immediately.
if ! touch "$HOME/.codex/.factory_probe" 2>/dev/null; then
  echo "FATAL: cannot write \$HOME/.codex - sandboxed environment; releasing task + pausing factory"
  "$PY" scripts/factory/state.py release --task "$TASK"
  "$PY" scripts/factory/state.py pause "Auto-paused: worker preflight probe failed (codex would die with EPERM)."
  exit 86
fi
rm -f "$HOME/.codex/.factory_probe"

prompt_file="$(mktemp "${TMPDIR:-/tmp}/neurogolf_prompt_${TASK}_XXXXXX")"
"$PY" scripts/factory/worker_prompt.py "$TASK" "$HASH" "$COST" > "$prompt_file"

exit_code=0
if command -v codex >/dev/null 2>&1; then
  codex exec --sandbox workspace-write --skip-git-repo-check - < "$prompt_file"
  exit_code=$?
else
  echo "ERROR: codex command not found"
  exit_code=127
fi
rm -f "$prompt_file"

"$PY" - "$TASK" "$COST" "$exit_code" "$before_sig" "$RESULTS" <<'PY'
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import onnx

task = int(sys.argv[1])
baseline_cost = int(sys.argv[2])
exit_code = int(sys.argv[3])
before_sig = sys.argv[4]
results_path = Path(sys.argv[5])
repo = Path.cwd()
scripts_dir = repo / "scripts"
sys.path.insert(0, str(scripts_dir))

from lib import scoring  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


hand_path = repo / "artifacts" / "handcrafted" / f"task{task:03d}.onnx"
present = hand_path.is_file()
new_cost = None
score_error = ""
after_mtime = ""
after_sig = ""
if present:
    after_mtime = str(int(hand_path.stat().st_mtime))
    after_sig = hashlib.sha256(hand_path.read_bytes()).hexdigest()
    try:
        model = onnx.load(str(hand_path))
        with tempfile.TemporaryDirectory(prefix="neurogolf_worker_score_") as workdir:
            scored = scoring.score_and_verify(
                model, task, workdir, label="factory_hand", require_correct=True
            )
        if scored is not None:
            new_cost = int(scored["cost"])
        else:
            score_error = "score_and_verify returned None"
    except Exception as exc:  # noqa: BLE001
        score_error = repr(exc)

changed = present and (not before_sig or after_sig != before_sig)
promoted = bool(changed and new_cost is not None and new_cost < baseline_cost)
record = {
    "time": now(),
    "task": task,
    "exit_code": exit_code,
    "handcrafted_present": present,
    "handcrafted_changed": changed,
    "promoted": promoted,
    "baseline_cost": baseline_cost,
    "new_cost": new_cost,
    "score_error": score_error,
    "before_sha256": before_sig,
    "after_sha256": after_sig,
}
results_path.parent.mkdir(parents=True, exist_ok=True)
with results_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

cmd = [
    sys.executable,
    "scripts/factory/state.py",
    "finish",
    "--task",
    str(task),
    "--exit-code",
    str(exit_code),
    "--promoted",
    "true" if promoted else "false",
]
if new_cost is not None:
    cmd.extend(["--cost", str(new_cost)])
subprocess.run(cmd, check=False)
print(json.dumps(record, sort_keys=True))
PY

echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] worker end task$(printf "%03d" "$TASK") codex_exit=$exit_code"
exit 0

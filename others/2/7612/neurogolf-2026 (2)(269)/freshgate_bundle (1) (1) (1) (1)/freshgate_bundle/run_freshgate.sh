#!/usr/bin/env bash
# Portable fresh-gate (overfit detector) for NeuroGolf ONNX candidates.
# Generates K fresh instances from the task's OWN generator (the same spec the
# organizer uses for the hidden set) and checks the net on them. Any fresh
# failure => overfit. Also re-checks visible gold via lib + the OFFICIAL utils.
#
# Usage:
#   ./run_freshgate.sh <task_number> <path_to_onnx> [K=500]
#   ./run_freshgate.sh --batch 158=a.onnx,280=b.onnx [K=500]
#
# Requires: python3 with  numpy onnx onnxruntime   (pip install numpy onnx onnxruntime)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
if [ "${1:-}" = "--batch" ]; then
  exec "$PY" "$DIR/scripts/verify_fix.py" --batch "$2" --k "${3:-500}"
else
  exec "$PY" "$DIR/scripts/verify_fix.py" --task "$1" --onnx "$2" --k "${3:-500}"
fi

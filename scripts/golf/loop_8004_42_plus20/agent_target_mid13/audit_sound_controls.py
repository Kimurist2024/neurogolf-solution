#!/usr/bin/env python3
"""Audit the nearest archived shape-truthful true-rule controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR_PATH = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
SPEC = importlib.util.spec_from_file_location("mid13_control_auditor", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


CONTROLS = {
    237: ROOT / "scripts/golf/loop_7999_13/lane_c23/candidates/task237_rebuild_542.onnx",
    238: ROOT / "scripts/golf/scratch_claude/task238/cand.onnx",
    354: ROOT / "scripts/golf/scratch/task354/task354.onnx",
    378: ROOT / "scripts/golf/loop_7999_13/lane_c23/candidates/task378_sound_k12_scaled.onnx",
}


def main() -> None:
    payload: dict[str, object] = {}
    for task, path in CONTROLS.items():
        row = AUDITOR.audit(f"sound_control_task{task}", task, path)
        trace = row.get("runtime_shape_trace") or {}
        row["truthful_static_runtime_shapes"] = not (
            trace.get("declared_actual_mismatches") or trace.get("error")
        )
        payload[str(task)] = row
        (HERE / "sound_controls_audit.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        score = row.get("official_like_score") or {}
        print(
            f"task={task} cost={score.get('cost')} "
            f"truthful={row['truthful_static_runtime_shapes']} "
            f"known_disable={row.get('known_disable_all', {}).get('total')} "
            f"known_default={row.get('known_default', {}).get('total')}",
            flush=True,
        )


if __name__ == "__main__":
    main()

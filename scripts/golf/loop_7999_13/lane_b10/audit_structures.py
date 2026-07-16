#!/usr/bin/env python3
"""Reuse the C11 strict structural/runtime audit on exact B10 members."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import audit  # noqa: E402


TASKS = (123, 134, 143, 162, 169, 184, 206)


def main() -> None:
    result = {}
    output = HERE / "baseline_structure_audit.json"
    for task in TASKS:
        label = f"base_task{task:03d}"
        result[label] = audit(label, task, HERE / "baseline" / f"task{task:03d}.onnx")
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        record = result[label]
        score = record.get("official_like_score", {})
        trace = record.get("runtime_shape_trace", {})
        print(
            label,
            score.get("cost"),
            score.get("correct"),
            len(trace.get("declared_actual_mismatches", [])),
            record.get("known_default", {}).get("total"),
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Checkpointed non-mutating audit of the eight 8005.16 baseline members."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (46, 157, 161, 189, 384, 193, 195, 281)
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from screen_all import known_dual, static_audit  # noqa: E402
from audit_candidates import runtime_shape_trace  # noqa: E402


def main() -> None:
    rows: dict[str, object] = {}
    output = HERE / "baseline_audit.json"
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        data = path.read_bytes()
        model = onnx.load_model_from_string(data)
        row: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "file_bytes": len(data),
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
            "static": static_audit(data, [str(path)], task),
            "known_dual": known_dual(task, data),
        }
        try:
            row["runtime_shape_trace"] = runtime_shape_trace(task, model)
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
        rows[f"{task:03d}"] = row
        output.write_text(json.dumps({"baseline": "submission_base_8005.16.zip", "tasks": rows}, indent=2) + "\n")
        print(f"task{task:03d} done", flush=True)


if __name__ == "__main__":
    main()

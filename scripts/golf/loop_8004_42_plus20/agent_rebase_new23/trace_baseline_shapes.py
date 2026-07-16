#!/usr/bin/env python3
"""Trace declared versus runtime shapes for the eight extracted incumbents."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(
    0,
    str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_private_exact15"),
)

from audit_exact import trace_shapes  # noqa: E402


def main() -> int:
    result = {}
    for task in (133, 145, 182, 187, 201, 204, 216, 233):
        model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        try:
            row = trace_shapes(model, task)
        except Exception as exc:  # noqa: BLE001
            row = {"trace_error": f"{type(exc).__name__}: {exc}"}
        result[str(task)] = row
        print(task, row.get("mismatch_count"), row.get("trace_error"), flush=True)
    (HERE / "baseline_runtime_shapes.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

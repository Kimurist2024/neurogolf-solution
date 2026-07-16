#!/usr/bin/env python3
"""Run both ORT modes over every known case for low42 incumbents."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "agent_new_low39" / "audit_lane.py"
SPEC = importlib.util.spec_from_file_location("low39_audit_lane", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TARGETS = (339, 126, 21, 171, 346, 227, 318, 332)


def main() -> None:
    rows = []
    for task in TARGETS:
        model = onnx.load(HERE / "baselines" / f"task{task:03d}.onnx")
        disable_all = MODULE.run_known(model, task, True)
        default = MODULE.run_known(model, task, False)
        rows.append({"task": task, "disable_all": disable_all, "default": default})
        print(
            f"task{task:03d}: disable={disable_all['right']}/{disable_all['total']} "
            f"errors={disable_all['errors']} default={default['right']}/{default['total']} "
            f"errors={default['errors']}",
            flush=True,
        )
    (HERE / "known_baseline_dual.json").write_text(json.dumps({
        "targets_completed": len(rows),
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

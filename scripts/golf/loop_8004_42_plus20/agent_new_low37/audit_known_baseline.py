#!/usr/bin/env python3
"""Record dual-ORT known correctness of the immutable latest members."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.agent_new_low34.audit_lane import run_known  # noqa: E402


TARGETS = (320, 154, 393, 290, 336, 3, 58, 72)


def main() -> None:
    rows = {}
    for task in TARGETS:
        model = onnx.load(HERE / "baselines" / f"task{task:03d}.onnx")
        rows[str(task)] = {
            "disable_all": run_known(model, task, True),
            "default": run_known(model, task, False),
        }
        print(f"task{task:03d}: {rows[str(task)]}", flush=True)
    (HERE / "known_baseline_dual.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

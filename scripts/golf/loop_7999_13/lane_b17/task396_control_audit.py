#!/usr/bin/env python3
"""Contrast the task396 exact baseline with the generator-derived sound control."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
sys.path.insert(0, str(HERE))

from fresh5000 import COUNT, SEED, encode, session  # noqa: E402


MODELS = {
    "exact_7999_13": HERE / "baseline" / "task396.onnx",
    "sound_control_1245": ROOT / "scripts/golf/scratch_codex/task396/agent_corner_micro.onnx",
}


def main() -> int:
    generator = importlib.import_module("task_fcb5c309")
    runners = []
    for label, path in MODELS.items():
        for mode in ("disabled", "default"):
            runners.append(
                {
                    "label": label,
                    "path": str(path.relative_to(ROOT)),
                    "mode": mode,
                    "session": session(path, mode),
                    "right": 0,
                    "wrong": 0,
                    "errors": 0,
                    "first_failure": None,
                }
            )
    random.seed(SEED)
    for index in range(COUNT):
        example = generator.generate()
        input_value = encode(example["input"])
        expected = encode(example["output"]).astype(bool)
        for runner in runners:
            try:
                raw = np.asarray(
                    runner["session"].run(["output"], {"input": input_value})[0]
                )
                if np.array_equal(raw > 0, expected):
                    runner["right"] += 1
                else:
                    runner["wrong"] += 1
                    runner["first_failure"] = runner["first_failure"] or {"index": index}
            except Exception as exc:  # noqa: BLE001
                runner["errors"] += 1
                runner["first_failure"] = runner["first_failure"] or {
                    "index": index,
                    "error": repr(exc),
                }
    rows = []
    for runner in runners:
        runner.pop("session")
        runner["pass"] = (
            runner["right"] == COUNT
            and runner["wrong"] == 0
            and runner["errors"] == 0
        )
        rows.append(runner)
        print(runner)
    (HERE / "task396_control_fresh5000.json").write_text(
        json.dumps({"count": COUNT, "seed": SEED, "rows": rows}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

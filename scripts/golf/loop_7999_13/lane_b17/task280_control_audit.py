#!/usr/bin/env python3
"""Fresh-5000 dual-ORT audit of the truthful task280 sound control."""

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


def main() -> int:
    generator = importlib.import_module("task_b527c5c6")
    path = HERE / "candidate_task280_truthful.onnx"
    runners = [
        {
            "mode": mode,
            "session": session(path, mode),
            "right": 0,
            "wrong": 0,
            "errors": 0,
            "first_failure": None,
        }
        for mode in ("disabled", "default")
    ]
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
    (HERE / "task280_control_fresh5000.json").write_text(
        json.dumps(
            {
                "count": COUNT,
                "seed": SEED,
                "path": str(path.relative_to(ROOT)),
                "actual_cost": 2161,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

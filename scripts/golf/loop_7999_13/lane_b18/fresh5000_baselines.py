#!/usr/bin/env python3
"""Dual-ORT fresh-5000 audit of the exact B18 baseline members."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))

from lib import scoring  # noqa: E402


COUNT = 5000
SEED = 181799913
GENERATORS = {89: "task_3e980e27", 255: "task_a64e4611"}


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def make_runner(task: int, mode: str) -> dict[str, Any]:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    row: dict[str, Any] = {
        "task": task,
        "mode": mode,
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "first_failure": None,
        "margin_min_positive": None,
        "margin_mid_count": 0,
    }
    if model is None:
        row["errors"] = COUNT
        row["first_failure"] = {"phase": "sanitize"}
        return row
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    try:
        row["session"] = ort.InferenceSession(model.SerializeToString(), options)
    except Exception as exc:  # noqa: BLE001
        row["errors"] = COUNT
        row["first_failure"] = {
            "phase": "session",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return row


def main() -> int:
    ort.set_default_logger_severity(4)
    generators = {
        task: importlib.import_module(module) for task, module in GENERATORS.items()
    }
    runners = [
        make_runner(task, mode)
        for task in GENERATORS
        for mode in ("disabled", "default")
    ]
    random.seed(SEED)
    for index in range(COUNT):
        examples = {task: generator.generate() for task, generator in generators.items()}
        encoded = {
            task: (encode(example["input"]), encode(example["output"]).astype(bool))
            for task, example in examples.items()
        }
        for runner in runners:
            if "session" not in runner:
                continue
            input_value, expected = encoded[runner["task"]]
            try:
                raw = np.asarray(
                    runner["session"].run(["output"], {"input": input_value})[0]
                )
                positive = raw[raw > 0]
                if positive.size:
                    current = float(positive.min())
                    prior = runner["margin_min_positive"]
                    runner["margin_min_positive"] = (
                        current if prior is None else min(prior, current)
                    )
                runner["margin_mid_count"] += int(
                    np.count_nonzero((raw > 0) & (raw < 0.25))
                )
                if np.array_equal(raw > 0, expected):
                    runner["right"] += 1
                else:
                    runner["wrong"] += 1
                    runner["first_failure"] = runner["first_failure"] or {
                        "index": index
                    }
            except Exception as exc:  # noqa: BLE001
                runner["errors"] += 1
                runner["first_failure"] = runner["first_failure"] or {
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if (index + 1) % 1000 == 0:
            print("progress", index + 1, flush=True)
    rows = []
    for runner in runners:
        runner.pop("session", None)
        runner["pass"] = (
            runner["right"] == COUNT
            and runner["wrong"] == 0
            and runner["errors"] == 0
            and runner["margin_mid_count"] == 0
        )
        rows.append(runner)
        print(runner, flush=True)
    report = {"count": COUNT, "seed": SEED, "rows": rows}
    (HERE / "fresh5000_baselines.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

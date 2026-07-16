#!/usr/bin/env python3
"""Fresh dual-ORT audit for the only structurally truthful target controls."""

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
BASE = HERE / "base"
TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
TARGETS = {
    80: {"count": 5000, "seeds": (80_800_001, 80_800_002)},
    101: {"count": 1000, "seeds": (101_800_001, 101_800_002)},
}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, optimized: bool) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(clean.SerializeToString(), options)


def generate(task: int, count: int, seed: int) -> tuple[list[dict[str, np.ndarray]], int]:
    module = importlib.import_module(f"task_{TASK_MAP[f'{task:03d}']}")
    random.seed(seed)
    rows: list[dict[str, np.ndarray]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        converted = scoring.convert_to_numpy(module.generate())
        if converted is not None:
            rows.append(converted)
    return rows, attempts


def audit(task: int, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    model = onnx.load(BASE / f"task{task:03d}.onnx")
    sessions = {
        "disable_all": make_session(model, False),
        "default": make_session(model, True),
    }
    result: dict[str, Any] = {
        "total": len(rows),
        "modes": {
            label: {"right": 0, "runtime_errors": 0}
            for label in sessions
        },
        "dual_raw_equal": 0,
        "first_failures": {},
    }
    for index, row in enumerate(rows):
        expected = row["output"] > 0
        raw_by_mode: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                raw = session.run(["output"], {"input": row["input"]})[0]
                raw_by_mode[label] = raw
                correct = raw.shape == expected.shape and np.array_equal(raw > 0, expected)
                result["modes"][label]["right"] += int(correct)
                if not correct and label not in result["first_failures"]:
                    result["first_failures"][label] = {
                        "index": index,
                        "wrong_cells": int(np.count_nonzero((raw > 0) != expected))
                        if raw.shape == expected.shape
                        else None,
                    }
            except Exception as exc:
                result["modes"][label]["runtime_errors"] += 1
                if label not in result["first_failures"]:
                    result["first_failures"][label] = {
                        "index": index,
                        "runtime_error": f"{type(exc).__name__}: {exc}",
                    }
        if len(raw_by_mode) == 2:
            result["dual_raw_equal"] += int(
                np.array_equal(
                    raw_by_mode["disable_all"], raw_by_mode["default"], equal_nan=True
                )
            )
    for row in result["modes"].values():
        row["rate"] = row["right"] / len(rows) if rows else None
    return result


def main() -> int:
    ort.set_default_logger_severity(4)
    output: dict[str, Any] = {}
    for task, config in TARGETS.items():
        task_rows = []
        for seed in config["seeds"]:
            rows, attempts = generate(task, config["count"], seed)
            row = audit(task, rows)
            row["seed"] = seed
            row["attempts"] = attempts
            task_rows.append(row)
            print(f"task{task:03d} seed={seed}: {row}", flush=True)
        output[f"task{task:03d}"] = task_rows
        (HERE / "fresh_audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

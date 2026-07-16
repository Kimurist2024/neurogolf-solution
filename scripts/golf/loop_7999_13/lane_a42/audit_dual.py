#!/usr/bin/env python3
"""Known and generator-fresh dual-ORT audit for task196 authority/history."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

MODELS = {
    "authority": HERE / "baseline_task196.onnx",
    "historical_968": ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task196_r07_static296.onnx",
}
MODES = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "default": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def make_session(path: Path, level: ort.GraphOptimizationLevel) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> None:
    sessions = {
        (mode, label): make_session(path, level)
        for mode, level in MODES.items()
        for label, path in MODELS.items()
    }
    rows = {
        (mode, label): {
            "mode": mode,
            "label": label,
            "known_correct": 0,
            "known_wrong": 0,
            "known_errors": 0,
            "known_raw_equal_authority": 0,
            "fresh_correct": 0,
            "fresh_wrong": 0,
            "fresh_errors": 0,
            "fresh_raw_equal_authority": 0,
            "first_fresh_failure": None,
        }
        for mode in MODES
        for label in MODELS
    }
    examples = scoring.load_examples(196)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    for example_index, example in enumerate(known):
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        want = benchmark["output"].astype(bool)
        for mode in MODES:
            authority_raw = sessions[(mode, "authority")].run(
                ["output"], {"input": benchmark["input"]}
            )[0]
            for label in MODELS:
                row = rows[(mode, label)]
                try:
                    raw = sessions[(mode, label)].run(
                        ["output"], {"input": benchmark["input"]}
                    )[0]
                    if np.array_equal(raw > 0, want):
                        row["known_correct"] += 1
                    else:
                        row["known_wrong"] += 1
                    row["known_raw_equal_authority"] += int(
                        np.array_equal(raw, authority_raw, equal_nan=True)
                    )
                except Exception:
                    row["known_errors"] += 1

    module = importlib.import_module("task_810b9b61")
    random.seed(196_800_263)
    started = time.monotonic()
    for case_index in range(5000):
        case = module.generate()
        benchmark = encode(case["input"])
        want = encode(case["output"]).astype(bool)
        for mode in MODES:
            authority_raw = sessions[(mode, "authority")].run(
                ["output"], {"input": benchmark}
            )[0]
            for label in MODELS:
                row = rows[(mode, label)]
                try:
                    raw = sessions[(mode, label)].run(
                        ["output"], {"input": benchmark}
                    )[0]
                    equal_gold = np.array_equal(raw > 0, want)
                    row["fresh_correct" if equal_gold else "fresh_wrong"] += 1
                    row["fresh_raw_equal_authority"] += int(
                        np.array_equal(raw, authority_raw, equal_nan=True)
                    )
                    if not equal_gold and row["first_fresh_failure"] is None:
                        difference = np.argwhere((raw > 0) != want)
                        row["first_fresh_failure"] = {
                            "case": case_index,
                            "shape": [len(case["input"]), len(case["input"][0])],
                            "different_cells": int(len(difference)),
                            "first_difference": difference[0].tolist(),
                        }
                except Exception as exc:
                    row["fresh_errors"] += 1
                    if row["first_fresh_failure"] is None:
                        row["first_fresh_failure"] = {
                            "case": case_index,
                            "error": repr(exc),
                        }
    report = {
        "known_cases": len(known),
        "fresh_cases": 5000,
        "seed": 196_800_263,
        "elapsed_seconds": time.monotonic() - started,
        "rows": list(rows.values()),
    }
    (HERE / "dual_known_fresh5000.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

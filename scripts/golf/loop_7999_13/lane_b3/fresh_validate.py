#!/usr/bin/env python3
"""Validate a B3 model directly against generator-produced expected outputs."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import sanitize_model  # noqa: E402


HASHES = {
    89: "3e980e27",
    196: "810b9b61",
    208: "890034e9",
    255: "a64e4611",
    340: "d687bc17",
    365: "e50d258f",
    370: "e8dc4411",
}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def expected(grid: list[list[int]]) -> np.ndarray:
    return encode(grid).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=sorted(HASHES))
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=799913)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    module = importlib.import_module(f"task_{HASHES[args.task]}")
    model = sanitize_model(onnx.load(args.onnx))
    if model is None:
        raise SystemExit("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), options)

    start = time.monotonic()
    valid = attempts = correct = wrong = errors = oversize = 0
    first_failure = None
    while valid < args.count:
        attempts += 1
        try:
            case = module.generate()
        except Exception as exc:
            errors += 1
            if first_failure is None:
                first_failure = {"attempt": attempts, "stage": "generate", "error": repr(exc)}
            if args.stop_on_failure:
                break
            continue
        height = len(case["input"])
        width = len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        try:
            raw = session.run(["output"], {"input": encode(case["input"])})[0]
            got = raw > 0.0
            want = expected(case["output"])
            if np.array_equal(got, want):
                correct += 1
            else:
                wrong += 1
                if first_failure is None:
                    diff = np.argwhere(got != want)
                    first_failure = {
                        "valid_case": valid,
                        "attempt": attempts,
                        "shape": [height, width],
                        "different_cells": int(len(diff)),
                        "first_difference": diff[0].tolist() if len(diff) else None,
                    }
                if args.stop_on_failure:
                    break
        except Exception as exc:
            errors += 1
            if first_failure is None:
                first_failure = {"valid_case": valid, "attempt": attempts, "error": repr(exc)}
            if args.stop_on_failure:
                break

    report = {
        "task": args.task,
        "onnx": str(args.onnx.resolve().relative_to(ROOT)),
        "seed": args.seed,
        "requested_valid": args.count,
        "attempts": attempts,
        "oversize_skipped": oversize,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "accuracy": correct / valid if valid else 0.0,
        "stopped_early": valid < args.count,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - start,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if valid == args.count and wrong == 0 and errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

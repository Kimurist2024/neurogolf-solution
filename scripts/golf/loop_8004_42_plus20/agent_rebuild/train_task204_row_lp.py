#!/usr/bin/env python3
"""Compile task204's bounded horizontal-box rule into one final Conv."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
KERNEL = 19  # generator max rectangle length 10 => radius 9 sees both sides


def solve(records: dict[bytes, int], channel: int) -> tuple[np.ndarray, float]:
    features = np.asarray([np.frombuffer(key, np.int8) for key in records], np.float64)
    labels = np.asarray(list(records.values()), np.float64)
    dimension = features.shape[1]
    signed = -(labels[:, None] * np.c_[features, np.ones(len(features))])
    eye = np.eye(dimension)
    absolute = np.block([[eye, -eye], [-eye, -eye]])
    absolute = np.c_[absolute[:, :dimension], np.zeros((2 * dimension, 1)), absolute[:, dimension:]]
    result = linprog(
        np.r_[np.zeros(dimension + 1), np.ones(dimension)],
        A_ub=np.r_[np.c_[signed, np.zeros((len(features), dimension))], absolute],
        b_ub=np.r_[-np.ones(len(features)), np.zeros(2 * dimension)],
        bounds=[(None, None)] * (dimension + 1) + [(0, None)] * dimension,
        method="highs",
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"channel{channel}: {result.message}")
    weights, bias = result.x[:dimension], float(result.x[dimension])
    if np.min(labels * (features @ weights + bias)) < 0.99:
        raise AssertionError("margin")
    print(f"channel{channel} unique={len(records)} nnz={np.count_nonzero(np.abs(weights)>1e-7)} bias={bias}")
    return weights.astype(np.float32), bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-train", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20_400_501)
    args = parser.parse_args()
    known = json.loads((ROOT / "inputs/neurogolf-2026/task204.json").read_text())
    examples = known["train"] + known["test"] + known["arc-gen"]
    generator = importlib.import_module("task_868de0fa")
    raw_rule = runpy.run_path(str(ROOT / "inputs/sakana-gcg-2025/raw/task204.py"))["p"]
    random.seed(args.seed)
    examples.extend(generator.generate() for _ in range(args.fresh_train))
    records: list[dict[bytes, int]] = [dict() for _ in range(10)]
    checked = 0
    for example in examples:
        grid, expected = example["input"], example["output"]
        decoded = raw_rule([row[:] for row in grid])
        if decoded != expected:
            raise AssertionError("true rule mismatch")
        checked += 1
        height, width = len(grid), len(grid[0])
        padded = np.zeros((10, 30, 30 + KERNEL - 1), np.int8)
        for row, values in enumerate(grid):
            for col, color in enumerate(values):
                padded[color, row, col + KERNEL // 2] = 1
        for row in range(30):
            for col in range(30):
                patch = padded[:, row, col : col + KERNEL].reshape(-1).tobytes()
                output_color = expected[row][col] if row < height and col < width else -1
                for channel in range(10):
                    label = 1 if channel == output_color else -1
                    previous = records[channel].get(patch)
                    if previous is not None and previous != label:
                        raise AssertionError(f"channel{channel}: conflicting row patch")
                    records[channel][patch] = label
    weights = np.zeros((10, 10 * KERNEL), np.float32)
    biases = np.zeros(10, np.float32)
    for channel in range(10):
        weights[channel], biases[channel] = solve(records[channel], channel)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], pads=[0, 9, 0, 9])],
        "task204_bounded_row_rule",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(weights.reshape(10, 10, 1, KERNEL), "W"),
            numpy_helper.from_array(biases, "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output = HERE / "candidates/task204_bounded_row_lp.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    report = {
        "task": 204,
        "known_examples": sum(len(known[key]) for key in ("train", "test", "arc-gen")),
        "fresh_training_examples": args.fresh_train,
        "numpy_reference_matches": checked,
        "kernel": [1, KERNEL],
        "output": str(output.relative_to(ROOT)),
    }
    (HERE / "task204_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

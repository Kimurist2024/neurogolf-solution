#!/usr/bin/env python3
"""Compile task125's exact 3x3 rule into a one-Conv candidate."""

from __future__ import annotations

import argparse
import importlib
import json
import random
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


def reference(grid: list[list[int]]) -> list[list[int]]:
    # Literal expansion of raw/task125.py.  It scans bottom-right to top-left;
    # already rewritten 8-cells therefore participate in later 3x3 counts.
    output = [row[:] for row in grid]
    height, width = len(grid), len(grid[0])
    for index in range(height * width - 1, -1, -1):
        row, col = divmod(index, width)
        marked_sum = sum(
            output[row - dr][col - dc] & 4
            for dr in (1, 0, height - 1)
            for dc in (1, 0, width - 1)
        )
        if output[row][col] // 8 * marked_sum:
            output[row][col] -= (marked_sum < 16) + 4
    return output


def collect(examples: list[dict[str, object]]) -> tuple[list[dict[bytes, int]], int]:
    records: list[dict[bytes, int]] = [dict() for _ in range(10)]
    checked = 0
    for example in examples:
        grid = example["input"]
        expected = example["output"]
        assert isinstance(grid, list) and isinstance(expected, list)
        decoded = reference(grid)
        if decoded != expected:
            raise AssertionError("numpy reference mismatch")
        checked += 1
        height, width = len(grid), len(grid[0])
        padded = np.zeros((10, 32, 32), dtype=np.int8)
        for row, values in enumerate(grid):
            for col, color in enumerate(values):
                padded[color, row + 1, col + 1] = 1
        for row in range(30):
            for col in range(30):
                patch = padded[:, row : row + 3, col : col + 3].reshape(-1).tobytes()
                output_color = decoded[row][col] if row < height and col < width else -1
                for channel in range(10):
                    label = 1 if channel == output_color else -1
                    previous = records[channel].get(patch)
                    if previous is not None and previous != label:
                        raise AssertionError(f"channel {channel}: conflicting local patch")
                    records[channel][patch] = label
    return records, checked


def solve(records: dict[bytes, int], label: str) -> tuple[np.ndarray, float]:
    features = np.asarray([np.frombuffer(key, np.int8) for key in records], np.float64)
    labels = np.asarray(list(records.values()), np.float64)
    dimension = features.shape[1]
    signed_features = -(labels[:, None] * np.c_[features, np.ones(len(features))])
    identity = np.eye(dimension)
    absolute = np.block([[identity, -identity], [-identity, -identity]])
    absolute = np.c_[absolute[:, :dimension], np.zeros((2 * dimension, 1)), absolute[:, dimension:]]
    result = linprog(
        np.r_[np.zeros(dimension + 1), np.ones(dimension)],
        A_ub=np.r_[np.c_[signed_features, np.zeros((len(features), dimension))], absolute],
        b_ub=np.r_[-np.ones(len(features)), np.zeros(2 * dimension)],
        bounds=[(None, None)] * (dimension + 1) + [(0, None)] * dimension,
        method="highs",
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"{label}: {result.message}")
    weights, bias = result.x[:dimension], float(result.x[dimension])
    score = features @ weights + bias
    if np.min(labels * score) < 0.99:
        raise AssertionError("LP margin")
    print(label, len(records), np.count_nonzero(np.abs(weights) > 1e-7), bias)
    return weights.astype(np.float32), bias


def build(weights: np.ndarray, biases: np.ndarray) -> onnx.ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], pads=[1, 1, 1, 1])],
        "task125_exact_local",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(weights.reshape(10, 10, 3, 3), "W"),
            numpy_helper.from_array(biases.astype(np.float32), "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-train", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12_500_501)
    args = parser.parse_args()
    known = json.loads((ROOT / "inputs/neurogolf-2026/task125.json").read_text())
    examples = known["train"] + known["test"] + known["arc-gen"]
    generator = importlib.import_module("task_543a7ed5")
    random.seed(args.seed)
    examples.extend(generator.generate() for _ in range(args.fresh_train))
    records, checked = collect(examples)
    weights = np.zeros((10, 90), np.float32)
    biases = np.zeros(10, np.float32)
    for channel in range(10):
        weights[channel], biases[channel] = solve(records[channel], f"channel{channel}")
    output = HERE / "candidates/task125_exact_local.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build(weights, biases), output)
    report = {
        "task": 125,
        "rule_type": "A_local",
        "known_examples": sum(len(known[key]) for key in ("train", "test", "arc-gen")),
        "fresh_training_examples": args.fresh_train,
        "numpy_reference_matches": checked,
        "seed": args.seed,
        "output": str(output.relative_to(ROOT)),
    }
    (HERE / "task125_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

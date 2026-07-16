#!/usr/bin/env python3
"""Probe a bounded square local rule as a single output-free Conv."""

from __future__ import annotations

import argparse
import copy
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
HASHES = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())


def normalize(value):
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def solve(records: dict[bytes, int], channel: int, dimension: int) -> tuple[np.ndarray, float]:
    features = np.asarray([np.frombuffer(key, np.int8) for key in records], np.float64)
    labels = np.asarray(list(records.values()), np.float64)
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
    print(f"channel{channel} unique={len(records)} nnz={np.count_nonzero(abs(weights)>1e-7)} bias={bias}")
    return weights.astype(np.float32), bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--kernel", type=int, choices=(3, 5, 7), required=True)
    parser.add_argument("--fresh-train", type=int, default=500)
    parser.add_argument("--seed", type=int, default=77_000_501)
    args = parser.parse_args()
    task = args.task
    known = json.loads((ROOT / f"inputs/neurogolf-2026/task{task:03d}.json").read_text())
    examples = known["train"] + known["test"] + known["arc-gen"]
    generator = importlib.import_module(f"task_{HASHES[f'{task:03d}']}")
    raw_rule = runpy.run_path(str(ROOT / f"inputs/sakana-gcg-2025/raw/task{task:03d}.py"))["p"]
    random.seed(args.seed)
    examples.extend(generator.generate() for _ in range(args.fresh_train))
    records: list[dict[bytes, int]] = [dict() for _ in range(10)]
    radius = args.kernel // 2
    checked = 0
    for example in examples:
        grid, expected = example["input"], example["output"]
        actual = normalize(raw_rule(copy.deepcopy(grid)))
        if actual != normalize(expected):
            raise AssertionError("raw true-rule mismatch")
        checked += 1
        ih, iw = len(grid), len(grid[0])
        oh, ow = len(expected), len(expected[0])
        padded = np.zeros((10, 30 + 2 * radius, 30 + 2 * radius), np.int8)
        for row, values in enumerate(grid):
            for col, color in enumerate(values):
                padded[color, row + radius, col + radius] = 1
        for row in range(30):
            for col in range(30):
                patch = padded[:, row : row + args.kernel, col : col + args.kernel].reshape(-1).tobytes()
                output_color = expected[row][col] if row < oh and col < ow else -1
                for channel in range(10):
                    label = 1 if channel == output_color else -1
                    previous = records[channel].get(patch)
                    if previous is not None and previous != label:
                        raise AssertionError(f"channel{channel}: conflicting local patch")
                    records[channel][patch] = label
    dimension = 10 * args.kernel * args.kernel
    weights = np.zeros((10, dimension), np.float32)
    biases = np.zeros(10, np.float32)
    for channel in range(10):
        weights[channel], biases[channel] = solve(records[channel], channel, dimension)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], pads=[radius] * 4)],
        f"task{task:03d}_bounded_local",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [
            numpy_helper.from_array(weights.reshape(10, 10, args.kernel, args.kernel), "W"),
            numpy_helper.from_array(biases, "B"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output = HERE / f"candidates/task{task:03d}_local_k{args.kernel}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    report = {
        "task": task,
        "kernel": args.kernel,
        "known_examples": sum(len(known[key]) for key in ("train", "test", "arc-gen")),
        "fresh_training_examples": args.fresh_train,
        "numpy_reference_matches": checked,
        "output": str(output.relative_to(ROOT)),
    }
    (HERE / f"task{task:03d}_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

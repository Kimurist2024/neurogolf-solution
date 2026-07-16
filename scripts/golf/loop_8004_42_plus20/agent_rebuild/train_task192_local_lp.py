#!/usr/bin/env python3
"""Build a compact, generator-derived task192 3x3 linear local model.

The true rule is decoded directly from ``raw/task192.py``:

* A is the most frequent non-zero color.
* A cell becomes A iff it is non-zero and sees A in both its horizontal and
  vertical radius-one windows; every other in-grid cell is background.

The generator only produces a restricted family of local patches (disjoint
solid rectangles plus isolated noise).  We solve a sparse L1 linear separator
for those patches, rather than embedding examples or an output lookup table.
The global A selection remains the incumbent's input-derived histogram.
"""

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
    flat = sum(grid, [])
    selected = max(range(1, 10), key=flat.count)
    height, width = len(grid), len(grid[0])
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            horizontal = any(
                grid[row][other] == selected
                for other in range(max(0, col - 1), min(width, col + 2))
            )
            vertical = any(
                grid[other][col] == selected
                for other in range(max(0, row - 1), min(height, row + 2))
            )
            if grid[row][col] != 0 and horizontal and vertical:
                output[row][col] = selected
    return output


def patch_rows(example: dict[str, object], kernel_width: int) -> tuple[int, list[tuple[bytes, int, int]]]:
    grid = example["input"]
    expected = example["output"]
    assert isinstance(grid, list) and isinstance(expected, list)
    decoded = reference(grid)
    if decoded != expected:
        raise AssertionError("decoded numpy reference disagrees with generator")
    height, width = len(grid), len(grid[0])
    selected = max(range(1, 10), key=sum(grid, []).count)
    left = 1
    right = kernel_width - 1 - left
    # The submitted tensor is always 30x30.  Border-adjacent output cells just
    # outside the logical grid can still see the final logical row/column, so
    # they must be represented faithfully rather than collapsed to a zero patch.
    padded = np.zeros((10, 30 + 2, 30 + left + right), dtype=np.int8)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            padded[color, row + 1, col + left] = 1
    rows: list[tuple[bytes, int, int]] = []
    for row in range(30):
        for col in range(30):
            patch = padded[:, row : row + 3, col : col + kernel_width].reshape(-1).tobytes()
            color = decoded[row][col] if row < height and col < width else -1
            rows.append((patch, color, selected))
    return selected, rows


def solve_l1(records: dict[bytes, int], label: str) -> tuple[np.ndarray, float]:
    features = np.asarray(
        [np.frombuffer(key, dtype=np.int8) for key in records], dtype=np.float64
    )
    labels = np.asarray(list(records.values()), dtype=np.int8)
    dimension = features.shape[1]

    # Positive scores are constrained >= +1 and negatives <= -1.
    positive = labels > 0
    signed = np.where(positive, 1.0, -1.0)
    class_a = -(signed[:, None] * np.c_[features, np.ones(len(features))])
    class_b = -np.ones(len(features))

    # L1 minimization gives small, sparse and numerically stable coefficients.
    identity = np.eye(dimension)
    abs_a = np.block([[identity, -identity], [-identity, -identity]])
    abs_b = np.zeros(2 * dimension)
    objective = np.r_[np.zeros(dimension + 1), np.ones(dimension)]
    constraints = np.c_[class_a, np.zeros((len(class_a), dimension))]
    abs_a = np.c_[abs_a[:, :dimension], np.zeros((2 * dimension, 1)), abs_a[:, dimension:]]
    result = linprog(
        objective,
        A_ub=np.r_[constraints, abs_a],
        b_ub=np.r_[class_b, abs_b],
        bounds=[(None, None)] * (dimension + 1) + [(0.0, None)] * dimension,
        method="highs",
        options={"time_limit": 120},
    )
    if not result.success:
        raise RuntimeError(f"{label}: LP failed: {result.message}")
    weights = result.x[:dimension]
    bias = float(result.x[dimension])
    scores = features @ weights + bias
    if np.any(scores[positive] < 0.99) or np.any(scores[~positive] > -0.99):
        raise AssertionError(f"{label}: separator margin check failed")
    print(
        label,
        f"unique={len(records)}",
        f"positive={int(positive.sum())}",
        f"nonzero_weights={int(np.count_nonzero(np.abs(weights) > 1e-7))}",
        f"max_abs={float(np.max(np.abs(weights))):.3f}",
        f"bias={bias:.3f}",
        flush=True,
    )
    return weights.astype(np.float32), bias


def make_model(weights: np.ndarray, biases: np.ndarray, kernel_width: int) -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    initializers = [
        numpy_helper.from_array(np.asarray([0] + [1] * 9, np.float32), "sel_nz"),
        numpy_helper.from_array(np.asarray([biases[0]] + [-1_000_000.0] * 9, np.float32), "bias_base"),
        numpy_helper.from_array(biases.astype(np.float32), "bias_table"),
        numpy_helper.from_array(weights.reshape(10, 10, 3, kernel_width), "W"),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "sel_nz"], ["hist"], equation="bchw,c->c"),
        helper.make_node("ArgMax", ["hist"], ["selected"], axis=0, keepdims=1),
        helper.make_node("Gather", ["bias_table", "selected"], ["selected_bias"], axis=0),
        helper.make_node("ScatterElements", ["bias_base", "selected", "selected_bias"], ["bias"], axis=0),
        helper.make_node(
            "Conv",
            ["input", "W", "bias"],
            ["output"],
            pads=[1, 1, 1, kernel_width - 2],
        ),
    ]
    graph = helper.make_graph(nodes, "task192_true_local_lp", [input_info], [output_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-train", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=19_200_501)
    parser.add_argument("--kernel-width", type=int, choices=(3, 4, 5), default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    known = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
    examples = known["train"] + known["test"] + known["arc-gen"]
    generator = importlib.import_module("task_7e0986d6")
    random.seed(args.seed)
    examples.extend(generator.generate() for _ in range(args.fresh_train))

    records: list[dict[bytes, int]] = [dict() for _ in range(10)]
    references_ok = 0
    for example in examples:
        selected, rows = patch_rows(example, args.kernel_width)
        references_ok += 1
        for patch, output_color, _ in rows:
            labels = ((0, 1 if output_color == 0 else -1), (selected, 1 if output_color == selected else -1))
            for channel, label in labels:
                previous = records[channel].get(patch)
                if previous is not None and previous != label:
                    raise AssertionError(f"channel {channel}: conflicting identical local patch")
                records[channel][patch] = label

    weight_rows = np.zeros((10, 30 * args.kernel_width), dtype=np.float32)
    biases = np.zeros(10, dtype=np.float32)
    weight_rows[0], biases[0] = solve_l1(records[0], "channel0")
    for channel in range(1, 10):
        weight_rows[channel], biases[channel] = solve_l1(records[channel], f"channel{channel}")

    output = args.output or (HERE / f"candidates/task192_true_local_lp_k{args.kernel_width}.onnx")
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(weight_rows, biases, args.kernel_width), output)
    report = {
        "task": 192,
        "rule_type": "A_local",
        "known_examples": sum(len(known[key]) for key in ("train", "test", "arc-gen")),
        "fresh_training_examples": args.fresh_train,
        "kernel_width": args.kernel_width,
        "numpy_reference_matches": references_ok,
        "seed": args.seed,
        "output": str(output.relative_to(ROOT)),
    }
    report_path = args.report or (HERE / "task192_training.json")
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Synthesize a compact task192 3x3 local classifier from generator rules.

This is a SOUND probe, not an adoption script.  It samples only fresh outputs
from the public task generator, extracts the complete 3x3 one-hot patch at
each in-grid cell, and solves one small linear-separation problem per selected
nonzero color.  The resulting graph uses the rule's global histogram/ArgMax
and a single terminal 3x3 Conv.  No visible fixture or expected-output table is
embedded in the model.
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
TASKS = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASKS))


def selected_color(grid: np.ndarray) -> int:
    """Return the most frequent color in 1..9, with lower-color tie break."""
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts[1:]) + 1)


def patch_key(
    padded_onehot: np.ndarray, row: int, col: int, kernel_width: int
) -> bytes:
    patch = padded_onehot[row : row + 3, col : col + kernel_width]
    return patch.transpose(2, 0, 1).reshape(-1).tobytes()


def collect(
    count: int, seed: int, kernel_width: int, left_pad: int
) -> dict[int, dict[int, set[bytes]]]:
    generator = importlib.import_module("task_7e0986d6")
    random.seed(seed)
    patterns = {color: {0: set(), 1: set()} for color in range(10)}
    for _ in range(count):
        example = generator.generate()
        grid = np.asarray(example["input"], dtype=np.uint8)
        expected = np.asarray(example["output"], dtype=np.uint8)
        color = selected_color(grid)
        onehot = np.zeros((30, 30, 10), dtype=np.uint8)
        onehot[: grid.shape[0], : grid.shape[1]] = np.eye(10, dtype=np.uint8)[grid]
        right_pad = kernel_width - 1 - left_pad
        padded = np.pad(onehot, ((1, 1), (left_pad, right_pad), (0, 0)))
        for row in range(30):
            for col in range(30):
                selected_label = int(
                    row < grid.shape[0]
                    and col < grid.shape[1]
                    and expected[row, col] == color
                )
                background_label = int(
                    row < grid.shape[0]
                    and col < grid.shape[1]
                    and expected[row, col] == 0
                )
                key = patch_key(padded, row, col, kernel_width)
                patterns[color][selected_label].add(key)
                patterns[0][background_label].add(key)
    return patterns


def decode_patterns(items: set[bytes], dimension: int) -> np.ndarray:
    if not items:
        return np.empty((0, dimension), dtype=np.float64)
    return np.stack(
        [np.frombuffer(item, dtype=np.uint8).astype(np.float64) for item in items]
    )


def solve_separator(negative: set[bytes], positive: set[bytes]) -> np.ndarray:
    """Find a minimum-L1 separator with unit margin.

    The selected-channel bias is fixed at -1.  This makes the all-zero padded
    exterior strictly lose to channel zero and avoids relying on an in-grid
    constant feature.
    """
    dimension = len(next(iter(negative)))
    if len(next(iter(positive))) != dimension:
        raise RuntimeError("positive/negative feature dimensions differ")
    x0 = decode_patterns(negative, dimension)
    x1 = decode_patterns(positive, dimension)
    if not len(x0) or not len(x1):
        raise RuntimeError("both classes must be represented")
    x = np.concatenate((x1, x0), axis=0)
    labels = np.concatenate((np.ones(len(x1)), -np.ones(len(x0))))
    fixed_bias = -1.0

    # Variables are [w(dimension), abs_bound(dimension)].
    objective = np.concatenate((np.zeros(dimension), np.ones(dimension)))
    classify = np.concatenate(
        (-labels[:, None] * x, np.zeros((len(x), dimension))), axis=1
    )
    identity = np.eye(dimension)
    abs_positive = np.concatenate((identity, -identity), axis=1)
    abs_negative = np.concatenate((-identity, -identity), axis=1)
    constraints = np.concatenate((classify, abs_positive, abs_negative), axis=0)
    bounds = np.concatenate(
        (-np.ones(len(x)) + labels * fixed_bias, np.zeros(2 * dimension)), axis=0
    )
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * dimension + [(0.0, None)] * dimension,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9,
                 "primal_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"local rule is not linearly separable: {result.message}")
    weights = result.x[:dimension]
    margins = labels * (x @ weights + fixed_bias)
    if float(np.min(margins)) < 0.999999:
        raise RuntimeError(f"separator margin regressed: {float(np.min(margins))}")
    return weights


def build_model(
    separators: dict[int, np.ndarray], kernel_width: int, left_pad: int
) -> onnx.ModelProto:
    kernel = np.zeros((10, 10, 3, kernel_width), dtype=np.float32)
    for color, separator in separators.items():
        kernel[color] = separator.reshape(10, 3, kernel_width).astype(np.float32)

    sel_nz = np.ones(10, dtype=np.float32)
    sel_nz[0] = 0.0
    bias_base = np.full(10, -1_000_000.0, dtype=np.float32)
    bias_base[0] = -1.0
    bias_update = np.full(1, -1.0, dtype=np.float32)

    nodes = [
        helper.make_node(
            "Einsum", ["input", "sel_nz"], ["hist"],
            name="nonzero_histogram", equation="bchw,c->c"
        ),
        helper.make_node(
            "ArgMax", ["hist"], ["selected"],
            name="selected_nonzero_color", axis=0, keepdims=1
        ),
        helper.make_node(
            "ScatterElements", ["bias_base", "selected", "bias_update"], ["bias"],
            name="route_selected_color", axis=0
        ),
        helper.make_node(
            "Conv", ["input", "kernel", "bias"], ["output"],
            name="local_rule", pads=[1, left_pad, 1, kernel_width - 1 - left_pad]
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "task192_conv3_generator_rule_probe",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
            [
                numpy_helper.from_array(sel_nz, "sel_nz"),
                numpy_helper.from_array(bias_base, "bias_base"),
                numpy_helper.from_array(bias_update, "bias_update"),
                numpy_helper.from_array(kernel, "kernel"),
            ],
        ),
        producer_name="codex-lane-b31",
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=192_031)
    parser.add_argument("--width", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--left-pad", type=int)
    args = parser.parse_args()
    left_pad = args.left_pad if args.left_pad is not None else args.width // 2
    if not 0 <= left_pad < args.width:
        parser.error("--left-pad must be in [0,width)")

    HERE.mkdir(parents=True, exist_ok=True)
    patterns = collect(args.cases, args.seed, args.width, left_pad)
    separators = {
        color: solve_separator(classes[0], classes[1])
        for color, classes in patterns.items()
    }
    model = build_model(separators, args.width, left_pad)
    output = HERE / f"task192_conv{args.width}_l{left_pad}_probe.onnx"
    onnx.save(model, output)
    manifest = {
        "candidate": str(output.relative_to(ROOT)),
        "training_source": "fresh public generator task_7e0986d6 only",
        "seed": args.seed,
        "cases": args.cases,
        "kernel_width": args.width,
        "left_pad": left_pad,
        "unique_patterns": {
            str(color): {
                "negative": len(classes[0]), "positive": len(classes[1])
            }
            for color, classes in patterns.items()
        },
        "max_abs_coefficient": {
            str(color): float(np.max(np.abs(weights)))
            for color, weights in separators.items()
        },
        "initializer_elements": sum(
            int(np.prod(initializer.dims)) for initializer in model.graph.initializer
        ),
    }
    (HERE / f"conv{args.width}_l{left_pad}_probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

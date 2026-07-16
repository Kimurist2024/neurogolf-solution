#!/usr/bin/env python3
"""Inspect task237 baseline intermediates on generator instances."""

from __future__ import annotations

import copy
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def one_hot(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def main() -> None:
    model = onnx.load(HERE / "baseline/task237.onnx")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    names = [
        "packed_grid",
        "packed_code",
        "packed_q",
        "max_q",
        "max_col_index_base",
        "w9_flag_u8",
        "max_col_index",
        "row_color_grid_h",
        "start_code",
        "latest_color_4d",
        "start_for_ray",
        "col_index_active",
        "ray_condition",
        "default_hash",
        "combined_hash",
        "output",
    ]
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    traced.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(traced.SerializeToString(), options)
    generator = importlib.import_module("task_99fa7670")
    for width in range(3, 10):
        random.seed(10_000 + width)
        example = generator.generate(
            width=width,
            height=7,
            rows=[0, 2, 5],
            cols=[min(1, width - 2), width - 2, 0],
            colors=[2, 7, 4],
        )
        values = session.run(names, {"input": one_hot(example["input"])})
        print(f"width={width}")
        for name, value in zip(names[:-4], values[:-4]):
            print(f"  {name}: {np.asarray(value).reshape(-1).tolist()}")
        cond = values[-4].astype(np.uint8).reshape(9, 9)
        default = values[-3].reshape(-1)
        labels = values[-1].argmax(1).reshape(30, 30)
        print(f"  ray_condition rows0..6: {cond[:7].tolist()}")
        print(f"  default_hash: {default.tolist()}")
        print(f"  labels rows0..6 cols0..8: {labels[:7, :9].tolist()}")


if __name__ == "__main__":
    main()

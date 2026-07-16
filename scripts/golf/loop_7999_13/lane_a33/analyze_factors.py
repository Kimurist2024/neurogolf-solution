#!/usr/bin/env python3
"""Inspect the pinned Wave17 task132/task222 tensor factors."""

from __future__ import annotations

import collections
import io
from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"


def attr(node: onnx.NodeProto, name: str) -> onnx.AttributeProto:
    return next(item for item in node.attribute if item.name == name)


def inspect(task: int) -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
    node = model.graph.node[0]
    equation = attr(node, "equation").s.decode()
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    arrays = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    print(f"\n=== task{task:03d}: {equation} ===")
    print("letters", sorted(set(equation) - {",", "-", ">"}), "count", len(set(equation) - {",", "-", ">"}))
    for index, (name, term) in enumerate(zip(node.input, terms, strict=True)):
        shape = arrays[name].shape if name in arrays else tuple(dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim)
        print(f"{index:02d} {name:>8s} {term:>5s} {shape}")
    print("output", rhs)
    print("input counts", collections.Counter(node.input))
    for name, array in arrays.items():
        print(f"\n{name} shape={array.shape} size={array.size} rank={np.linalg.matrix_rank(array.reshape(array.shape[0], -1)) if array.ndim else 1}")
        print(array)
        for axis in range(array.ndim):
            moved = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
            rank = np.linalg.matrix_rank(moved)
            zero = [i for i, row in enumerate(moved) if np.all(row == 0)]
            duplicates = [(i, j) for i in range(len(moved)) for j in range(i) if np.array_equal(moved[i], moved[j])]
            negatives = [(i, j) for i in range(len(moved)) for j in range(i) if np.array_equal(moved[i], -moved[j])]
            print(f" axis{axis}: rank={rank} zero={zero} dup={duplicates} neg={negatives}")


if __name__ == "__main__":
    inspect(132)
    inspect(222)

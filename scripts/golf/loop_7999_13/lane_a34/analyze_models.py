#!/usr/bin/env python3
"""Dump Wave17 task099/task398 graph structure and factor identities."""

from __future__ import annotations

import collections
from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"


def attrs(node: onnx.NodeProto) -> dict[str, object]:
    result = {}
    for item in node.attribute:
        if item.name == "equation":
            result[item.name] = item.s.decode()
        elif item.ints:
            result[item.name] = list(item.ints)
        elif item.s:
            result[item.name] = item.s.decode(errors="replace")
        elif item.HasField("i"):
            result[item.name] = item.i
        elif item.HasField("f"):
            result[item.name] = item.f
    return result


def inspect(task: int) -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read(f"task{task:03d}.onnx")
    (Path(__file__).resolve().parent / f"baseline_task{task:03d}.onnx").write_bytes(payload)
    model = onnx.load_model_from_string(payload)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    print(f"\n=== task{task:03d} nodes={len(model.graph.node)} params={sum(x.size for x in arrays.values())} ===")
    print("input", [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.input])
    print("output", [(x.name, [d.dim_value for d in x.type.tensor_type.shape.dim]) for x in model.graph.output])
    for index, node in enumerate(model.graph.node):
        print(index, node.op_type, list(node.input), "->", list(node.output), attrs(node))
    for name, array in arrays.items():
        print(f"\n{name} dtype={array.dtype} shape={array.shape} size={array.size}")
        print(array)
        if array.ndim:
            for axis in range(array.ndim):
                rows = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1)
                print(
                    " axis", axis,
                    "rank", np.linalg.matrix_rank(rows),
                    "zero", [i for i, row in enumerate(rows) if np.all(row == 0)],
                    "dup", [(i, j) for i in range(len(rows)) for j in range(i) if np.array_equal(rows[i], rows[j])],
                    "neg", [(i, j) for i in range(len(rows)) for j in range(i) if np.array_equal(rows[i], -rows[j])],
                )
    shapes: dict[tuple[int, ...], list[str]] = collections.defaultdict(list)
    for name, array in arrays.items():
        shapes[array.shape].append(name)
    print("same shapes", {shape: names for shape, names in shapes.items() if len(names) > 1})


if __name__ == "__main__":
    inspect(99)
    inspect(398)

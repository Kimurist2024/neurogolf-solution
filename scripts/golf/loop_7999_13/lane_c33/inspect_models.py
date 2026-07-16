#!/usr/bin/env python3
"""Inventory C33 graph equations, shapes, reuse, values, and ranks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?" for dim in value.type.tensor_type.shape.dim]


def main() -> None:
    result = {}
    for path in sorted((HERE / "baseline").glob("*.onnx")):
        model = onnx.load(path)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        uses = Counter(name for node in model.graph.node for name in node.input)
        row = {
            "nodes": [],
            "initializers": [],
            "value_info": {value.name: shape(value) for value in list(inferred.graph.value_info) + list(inferred.graph.output)},
        }
        for node in model.graph.node:
            attrs = {}
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.STRING:
                    attrs[attr.name] = attr.s.decode("ascii", errors="replace")
                elif attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = attr.i
                else:
                    attrs[attr.name] = f"type={attr.type}"
            row["nodes"].append({"op": node.op_type, "inputs": list(node.input), "outputs": list(node.output), "attrs": attrs})
        for item in model.graph.initializer:
            array = numpy_helper.to_array(item)
            ranks = []
            for axis in range(array.ndim):
                matrix = np.moveaxis(array, axis, 0).reshape(array.shape[axis], -1).astype(np.float64)
                ranks.append(int(np.linalg.matrix_rank(matrix)))
            row["initializers"].append(
                {
                    "name": item.name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size": int(array.size),
                    "uses": uses[item.name],
                    "unfold_ranks": ranks,
                    "unique": np.unique(array).tolist(),
                    "data": array.tolist() if array.size <= 240 else None,
                }
            )
        result[path.stem] = row
    (HERE / "model_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    for task, row in result.items():
        print(task, row["value_info"])
        for index, node in enumerate(row["nodes"]):
            print(index, node)
        for item in row["initializers"]:
            print(item["name"], item["shape"], item["size"], item["uses"], item["unfold_ranks"])


if __name__ == "__main__":
    main()

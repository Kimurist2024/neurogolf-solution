#!/usr/bin/env python3
"""Inspect safe Split nodes whose unused variadic outputs need Slice replacement."""

from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"
TARGETS = {19: 11, 124: 5}


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> None:
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task, node_index in TARGETS.items():
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
            shapes = {
                value.name: dims(value)
                for value in list(inferred.graph.input)
                + list(inferred.graph.value_info)
                + list(inferred.graph.output)
            }
            arrays = {
                item.name: {
                    "dtype": str(numpy_helper.to_array(item).dtype),
                    "shape": list(numpy_helper.to_array(item).shape),
                    "values": numpy_helper.to_array(item).tolist(),
                }
                for item in model.graph.initializer
            }
            node = model.graph.node[node_index]
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            rows.append(
                {
                    "task": task,
                    "node_index": node_index,
                    "node": {
                        "op": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attributes": attributes,
                    },
                    "input_shapes": {name: shapes.get(name) for name in node.input},
                    "output_shapes": {name: shapes.get(name) for name in node.output},
                    "input_initializers": {name: arrays.get(name) for name in node.input if name in arrays},
                    "all_initializers": arrays,
                    "existing_slice_nodes": [
                        {
                            "index": index,
                            "inputs": list(item.input),
                            "outputs": list(item.output),
                            "input_initializers": {
                                name: arrays.get(name) for name in item.input if name in arrays
                            },
                        }
                        for index, item in enumerate(model.graph.node)
                        if item.op_type == "Slice"
                    ],
                }
            )
    (HERE / "split_replacement_anatomy.json").write_text(json.dumps(rows, indent=2, default=list) + "\n")
    print(json.dumps(rows, indent=2, default=list))


if __name__ == "__main__":
    main()

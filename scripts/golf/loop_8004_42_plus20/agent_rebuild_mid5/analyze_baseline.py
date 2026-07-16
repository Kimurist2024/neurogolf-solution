#!/usr/bin/env python3
"""Extract and describe this lane's four immutable baseline members."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8004.50.zip"
TASKS = (23, 187, 209, 367)


def dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(d.dim_value) if d.HasField("dim_value") else d.dim_param
        for d in value_info.type.tensor_type.shape.dim
    ]


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    report = {}
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            payload = archive.read(name)
            path = HERE / f"baseline_{name}"
            path.write_bytes(payload)
            model = onnx.load_model_from_string(payload)
            inferred = shape_inference.infer_shapes(
                model, strict_mode=True, data_prop=True
            )
            initializers = {item.name for item in inferred.graph.initializer}
            params = sum(int(np.prod(item.dims)) for item in inferred.graph.initializer)
            values = []
            memory = 0
            for value_info in inferred.graph.value_info:
                shape = dims(value_info)
                elem_type = value_info.type.tensor_type.elem_type
                itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(elem_type)).itemsize
                count = int(np.prod(shape)) if all(isinstance(d, int) for d in shape) else None
                nbytes = None if count is None else count * itemsize
                if value_info.name not in initializers and nbytes is not None:
                    memory += nbytes
                values.append(
                    {
                        "name": value_info.name,
                        "dtype": helper.tensor_dtype_to_string(elem_type),
                        "shape": shape,
                        "bytes": nbytes,
                    }
                )
            report[str(task)] = {
                "file_bytes": len(payload),
                "cost": params + memory,
                "params": params,
                "memory": memory,
                "nodes": [
                    {
                        "index": index,
                        "op": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "attrs": {
                            attr.name: helper.get_attribute_value(attr)
                            for attr in node.attribute
                        },
                    }
                    for index, node in enumerate(model.graph.node)
                ],
                "initializers": [
                    {
                        "name": item.name,
                        "dtype": helper.tensor_dtype_to_string(item.data_type),
                        "shape": list(item.dims),
                        "values": onnx.numpy_helper.to_array(item).tolist(),
                    }
                    for item in model.graph.initializer
                ],
                "values": values,
            }
    (HERE / "baseline_analysis.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    for task, item in report.items():
        print(
            f"task{int(task):03d}: cost={item['cost']} "
            f"memory={item['memory']} params={item['params']} "
            f"nodes={len(item['nodes'])}"
        )
        for node in item["nodes"]:
            print(
                f"  {node['index']:02d} {node['op']:<16} "
                f"{node['inputs']} -> {node['outputs']}"
            )


if __name__ == "__main__":
    main()

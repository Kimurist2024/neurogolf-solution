#!/usr/bin/env python3
"""Emit a compact, deterministic structural dump of the eight changed LB nets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "base_models"
TASKS = (73, 111, 122, 260, 271, 285, 289, 359)


def shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dims = value_info.type.tensor_type.shape.dim
    result: list[int | str | None] = []
    for dim in dims:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def attrs(node: onnx.NodeProto) -> dict[str, object]:
    result: dict[str, object] = {}
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            value = value.decode("utf-8", "replace")
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, onnx.TensorProto):
            arr = numpy_helper.to_array(value)
            value = {"dtype": str(arr.dtype), "shape": list(arr.shape), "values": arr.tolist()}
        result[attr.name] = value
    return result


def dump(task: int) -> dict[str, object]:
    path = BASE / f"task{task:03d}.onnx"
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    types: dict[str, object] = {}
    for vi in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tt = vi.type.tensor_type
        types[vi.name] = {"dtype": onnx.TensorProto.DataType.Name(tt.elem_type), "shape": shape(vi)}
    inits = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        flat = arr.reshape(-1)
        if flat.size <= 120:
            values: object = arr.tolist()
        else:
            values = {
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "unique": np.unique(flat).tolist()[:40],
                "unique_count": int(np.unique(flat).size),
            }
        inits.append({
            "name": init.name,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "size": int(arr.size),
            "values": values,
        })
    nodes = []
    for i, node in enumerate(model.graph.node):
        nodes.append({
            "index": i,
            "name": node.name,
            "op": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "output_types": {out: types.get(out) for out in node.output},
            "attrs": attrs(node),
        })
    return {
        "task": task,
        "path": str(path),
        "ir_version": model.ir_version,
        "opsets": {op.domain: op.version for op in model.opset_import},
        "inputs": [{"name": x.name, **types[x.name]} for x in model.graph.input],
        "outputs": [{"name": x.name, **types[x.name]} for x in model.graph.output],
        "initializers": inits,
        "nodes": nodes,
    }


def main() -> None:
    out = Path(__file__).with_name("graph_audit.json")
    out.write_text(json.dumps([dump(task) for task in TASKS], indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

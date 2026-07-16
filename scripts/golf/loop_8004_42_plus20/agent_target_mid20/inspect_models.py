#!/usr/bin/env python3
"""Emit exact graph anatomy for the four lane incumbents (read-only input)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
CURRENT = HERE / "current"


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> int:
    report: dict[str, object] = {}
    for path in sorted(CURRENT.glob("task*.onnx")):
        model = onnx.load(path)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        values = {
            value.name: {
                "dtype": value.type.tensor_type.elem_type,
                "shape": shape_of(value),
            }
            for value in [
                *inferred.graph.input,
                *inferred.graph.value_info,
                *inferred.graph.output,
            ]
        }
        inits = {}
        for init in inferred.graph.initializer:
            array = numpy_helper.to_array(init)
            flat = array.reshape(-1)
            entry: dict[str, object] = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "params": int(max(1, math.prod(array.shape))),
                "min": float(flat.min()) if flat.size else None,
                "max": float(flat.max()) if flat.size else None,
            }
            if flat.size <= 100:
                entry["values"] = array.tolist()
            inits[init.name] = entry
        nodes = []
        for index, node in enumerate(inferred.graph.node):
            attrs = {}
            for attr in node.attribute:
                try:
                    attrs[attr.name] = onnx.helper.get_attribute_value(attr)
                except Exception as exc:  # pragma: no cover - audit fallback
                    attrs[attr.name] = repr(exc)
            nodes.append(
                {
                    "index": index,
                    "name": node.name,
                    "domain": node.domain,
                    "op": node.op_type,
                    "inputs": list(node.input),
                    "outputs": [
                        {"name": name, **values.get(name, {})} for name in node.output
                    ],
                    "attrs": attrs,
                }
            )
        report[path.stem] = {
            "path": str(path),
            "ir_version": model.ir_version,
            "opsets": {item.domain: item.version for item in model.opset_import},
            "inputs": [{"name": v.name, **values[v.name]} for v in inferred.graph.input],
            "outputs": [{"name": v.name, **values[v.name]} for v in inferred.graph.output],
            "op_hist": dict(Counter(node.op_type for node in inferred.graph.node)),
            "initializers": inits,
            "nodes": nodes,
        }
    output = HERE / "current_anatomy.json"
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

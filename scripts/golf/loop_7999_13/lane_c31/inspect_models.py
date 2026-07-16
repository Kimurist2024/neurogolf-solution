#!/usr/bin/env python3
"""Dump compact structural and initializer facts for C31 incumbents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def main() -> None:
    result = {}
    for path in sorted((HERE / "baseline").glob("*.onnx")):
        model = onnx.load(path)
        row = {
            "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "nodes": [],
            "initializers": [],
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
            row["nodes"].append(
                {"op": node.op_type, "inputs": list(node.input), "outputs": list(node.output), "attrs": attrs}
            )
        for item in model.graph.initializer:
            array = numpy_helper.to_array(item)
            data = array.tolist() if array.size <= 400 else None
            row["initializers"].append(
                {
                    "name": item.name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "size": int(array.size),
                    "rank": int(np.linalg.matrix_rank(array)) if array.ndim == 2 else None,
                    "unique": np.unique(array).tolist(),
                    "data": data,
                }
            )
        result[path.stem] = row
    (HERE / "model_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

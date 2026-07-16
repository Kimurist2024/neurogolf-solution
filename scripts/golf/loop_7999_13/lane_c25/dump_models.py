#!/usr/bin/env python3
"""Dump graph structure, initializer reachability, and scorer parameter counts."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def tensor_record(tensor: onnx.TensorProto, referenced: bool) -> dict[str, object]:
    array = numpy_helper.to_array(tensor)
    flat = array.reshape(-1)
    return {
        "name": tensor.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "elements": int(math.prod(array.shape)) if array.shape else 1,
        "referenced": referenced,
        "sample": flat[:16].tolist(),
        "min": float(np.min(flat)) if flat.size else None,
        "max": float(np.max(flat)) if flat.size else None,
    }


def main() -> None:
    output: dict[str, object] = {}
    for task in (131, 251):
        path = HERE / "base" / f"task{task}.onnx"
        model = onnx.load(path)
        used_inputs = {name for node in model.graph.node for name in node.input if name}
        output[f"task{task}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_bytes": path.stat().st_size,
            "scorer_params": scoring.calculate_params(model),
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "inputs": [value.name for value in model.graph.input],
            "outputs": [value.name for value in model.graph.output],
            "initializers": [tensor_record(item, item.name in used_inputs) for item in model.graph.initializer],
            "nodes": [
                {
                    "index": index,
                    "name": node.name,
                    "op": node.op_type,
                    "domain": node.domain,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": [attr.name for attr in node.attribute],
                }
                for index, node in enumerate(model.graph.node)
            ],
        }
    (HERE / "model_dump.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for task, record in output.items():
        unused = [item["name"] for item in record["initializers"] if not item["referenced"]]
        print(task, "params=", record["scorer_params"], "nodes=", len(record["nodes"]), "unused=", unused)


if __name__ == "__main__":
    main()

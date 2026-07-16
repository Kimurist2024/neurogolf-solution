#!/usr/bin/env python3
"""Stage and inventory the integer-valued precontraction targets."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
AUTHORITY = REPO / "submission_base_8009.46.zip"
BASE = HERE / "base"
TASKS = (74, 200, 211)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    result = {}
    for item in node.attribute:
        value = onnx.helper.get_attribute_value(item)
        result[item.name] = value.decode(errors="replace") if isinstance(value, bytes) else value
    return result


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    if not value.type.HasField("tensor_type"):
        return []
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def inventory(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = {
        value.name: dims(value)
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    uses: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nodes = []
    for node_index, node in enumerate(model.graph.node):
        terms = None
        if node.op_type == "Einsum":
            equation = attrs(node)["equation"]
            terms = equation.split("->", 1)[0].split(",")
        for position, name in enumerate(node.input):
            uses[name].append(
                {
                    "node": node_index,
                    "op": node.op_type,
                    "position": position,
                    "term": terms[position] if terms else None,
                }
            )
        nodes.append(
            {
                "index": node_index,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input_shapes": [shapes.get(name) for name in node.input],
                "output_shapes": [shapes.get(name) for name in node.output],
                "attrs": attrs(node),
            }
        )
    initializers = []
    for item in model.graph.initializer:
        value = np.asarray(numpy_helper.to_array(item))
        initializers.append(
            {
                "name": item.name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "size": int(value.size),
                "values": value.tolist() if value.size <= 500 else None,
                "uses": uses[item.name],
            }
        )
    memory, params, cost = cost_of(str(path))
    return {
        "sha256": sha256(path.read_bytes()),
        "profile": {"memory": memory, "params": params, "cost": cost},
        "nodes": nodes,
        "initializers": initializers,
    }


def main() -> None:
    authority_sha = sha256(AUTHORITY.read_bytes())
    if authority_sha != "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927":
        raise RuntimeError(authority_sha)
    BASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            (BASE / f"task{task:03d}.onnx").write_bytes(archive.read(f"task{task:03d}.onnx"))
    result = {
        "authority_sha256": authority_sha,
        "tasks": {str(task): inventory(BASE / f"task{task:03d}.onnx") for task in TASKS},
    }
    (HERE / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    for task in TASKS:
        item = result["tasks"][str(task)]
        print(f"task{task:03d} sha={item['sha256']} profile={item['profile']}")
        for node in item["nodes"]:
            print(f" n{node['index']:02d} {node['op']} {node['inputs']} -> {node['outputs']} attrs={node['attrs']}")


if __name__ == "__main__":
    main()

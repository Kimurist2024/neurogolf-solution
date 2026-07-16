#!/usr/bin/env python3
"""Stage and fully inventory the four shared-Concat/Einsum tasks."""

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
STAGED_013 = REPO / "others" / "71407" / "task013.onnx"
BASE = HERE / "base"
TASKS = (13, 55, 99, 281)

sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def attributes(node: onnx.NodeProto) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in node.attribute:
        value = onnx.helper.get_attribute_value(item)
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        values[item.name] = value
    return values


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    if not value.type.HasField("tensor_type"):
        return []
    result: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def model_inventory(path: Path, source: str) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    shapes = {
        value.name: dims(value)
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    consumers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for position, name in enumerate(node.input):
            consumers[name].append({"node": index, "op": node.op_type, "position": position})
    nodes = []
    for index, node in enumerate(model.graph.node):
        nodes.append(
            {
                "index": index,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input_shapes": {name: shapes.get(name) for name in node.input},
                "output_shapes": {name: shapes.get(name) for name in node.output},
                "attributes": attributes(node),
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
                "values": value.tolist() if value.size <= 200 else None,
                "consumers": consumers[item.name],
            }
        )
    memory, params, cost = cost_of(str(path))
    return {
        "source": source,
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path.read_bytes()),
        "profile": {"memory": memory, "params": params, "cost": cost},
        "inputs": {value.name: dims(value) for value in inferred.graph.input},
        "outputs": {value.name: dims(value) for value in inferred.graph.output},
        "nodes": nodes,
        "initializers": initializers,
    }


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    authority_sha = sha256(AUTHORITY.read_bytes())
    if authority_sha != "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927":
        raise RuntimeError(authority_sha)
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            destination = BASE / f"task{task:03d}.onnx"
            data = STAGED_013.read_bytes() if task == 13 else archive.read(f"task{task:03d}.onnx")
            destination.write_bytes(data)
    if sha256((BASE / "task013.onnx").read_bytes()) != "97d6a181110e43e8a5b20031ac766bc38fa8d5787070a7bc026306a2da1c7173":
        raise RuntimeError("task013 staged SHA mismatch")
    result = {
        "authority": str(AUTHORITY.relative_to(REPO)),
        "authority_sha256": authority_sha,
        "tasks": {
            str(task): model_inventory(
                BASE / f"task{task:03d}.onnx",
                "others/71407 staged" if task == 13 else "8009.46 authority",
            )
            for task in TASKS
        },
    }
    (HERE / "inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    for task in TASKS:
        item = result["tasks"][str(task)]
        print(f"task{task:03d} sha={item['sha256']} profile={item['profile']}")
        for node in item["nodes"]:
            print(
                f"  n{node['index']:02d} {node['op']} {node['inputs']} -> {node['outputs']} "
                f"shape={node['output_shapes']} attrs={node['attributes']}"
            )


if __name__ == "__main__":
    main()

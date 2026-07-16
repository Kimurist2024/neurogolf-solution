#!/usr/bin/env python3
"""Static memory/initializer/CSE anatomy for lane 138 authority members."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = (157, 368, 370)


def shape(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def initializer_key(item: onnx.TensorProto) -> tuple[Any, ...]:
    clone = onnx.TensorProto()
    clone.CopyFrom(item)
    clone.name = ""
    return int(item.data_type), tuple(item.dims), clone.SerializeToString()


def node_key(node: onnx.NodeProto) -> tuple[Any, ...]:
    return (
        node.domain,
        node.op_type,
        tuple(node.input),
        tuple(sorted((attr.name, attr.SerializeToString()) for attr in node.attribute)),
        len(node.output),
    )


def inspect(task: int) -> dict[str, Any]:
    path = HERE / f"baseline/task{task:03d}.onnx"
    data = path.read_bytes()
    model = onnx.load_model_from_string(data)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    values = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    output_rows = []
    by_op: Counter[str] = Counter()
    for node in inferred.graph.node:
        for output in node.output:
            if not output or output == "output":
                continue
            value = values[output]
            dims = shape(value)
            dtype = int(value.type.tensor_type.elem_type)
            memory = math.prod(dims) * helper.tensor_dtype_to_np_dtype(dtype).itemsize
            by_op[node.op_type] += memory
            output_rows.append(
                {
                    "name": output,
                    "op": node.op_type,
                    "dtype": onnx.TensorProto.DataType.Name(dtype),
                    "shape": dims,
                    "bytes": memory,
                }
            )
    consumers: defaultdict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node.output[0] if node.output else node.name)
    initializer_rows = []
    duplicate_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        duplicate_groups[initializer_key(item)].append(item.name)
        initializer_rows.append(
            {
                "name": item.name,
                "dtype": onnx.TensorProto.DataType.Name(item.data_type),
                "shape": list(item.dims),
                "elements": int(math.prod(item.dims)),
                "uses": len(consumers[item.name]),
                "values": array.reshape(-1).tolist() if array.size <= 30 else None,
            }
        )
    node_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for node in model.graph.node:
        node_groups[node_key(node)].append(node.output[0] if node.output else node.name)
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "nodes": len(model.graph.node),
        "value_info": len(model.graph.value_info),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
        "declared_static_memory": sum(row["bytes"] for row in output_rows),
        "memory_by_op": dict(by_op.most_common()),
        "largest_intermediates": sorted(
            output_rows, key=lambda row: (-row["bytes"], row["name"])
        )[:30],
        "parameter_elements": sum(row["elements"] for row in initializer_rows),
        "initializers": sorted(
            initializer_rows, key=lambda row: (-row["elements"], row["name"])
        ),
        "unused_initializers": sorted(
            row["name"] for row in initializer_rows if row["uses"] == 0
        ),
        "duplicate_initializer_groups": sorted(
            sorted(names) for names in duplicate_groups.values() if len(names) > 1
        ),
        "duplicate_node_groups": sorted(
            sorted(names) for names in node_groups.values() if len(names) > 1
        ),
    }


def main() -> None:
    result = {str(task): inspect(task) for task in TASKS}
    (HERE / "audit/model_anatomy.json").write_text(json.dumps(result, indent=2) + "\n")
    for task in TASKS:
        row = result[str(task)]
        print(
            f"task{task:03d} memory={row['declared_static_memory']} "
            f"params={row['parameter_elements']} unused={len(row['unused_initializers'])} "
            f"dup_init={len(row['duplicate_initializer_groups'])} "
            f"dup_node={len(row['duplicate_node_groups'])}"
        )


if __name__ == "__main__":
    main()

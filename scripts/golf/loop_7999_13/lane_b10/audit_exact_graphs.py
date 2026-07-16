#!/usr/bin/env python3
"""Emit a compact, deterministic graph audit for the exact B10 baselines."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
LANE = ROOT / "scripts/golf/loop_7999_13/lane_b10"
TASKS = (123, 134, 143, 162, 169, 184, 206)


def value_shape(value_info: onnx.ValueInfoProto) -> list[int | str | None] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape


def main() -> None:
    result: dict[str, object] = {}
    for task in TASKS:
        path = LANE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        graph = model.graph
        consumers: dict[str, list[int]] = defaultdict(list)
        for index, node in enumerate(graph.node):
            for name in node.input:
                if name:
                    consumers[name].append(index)

        initializers = []
        for tensor in graph.initializer:
            array = numpy_helper.to_array(tensor)
            flat = array.reshape(-1)
            preview = flat[:32].tolist()
            initializers.append(
                {
                    "name": tensor.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "elements": int(array.size),
                    "min": float(np.min(array)) if array.size else None,
                    "max": float(np.max(array)) if array.size else None,
                    "unique": int(np.unique(array).size) if array.size else 0,
                    "preview": preview,
                    "consumers": consumers.get(tensor.name, []),
                }
            )

        nodes = []
        for index, node in enumerate(graph.node):
            attrs = {}
            for attr in node.attribute:
                value = onnx.helper.get_attribute_value(attr)
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                elif not isinstance(value, (str, int, float, list, tuple)):
                    value = str(value)
                attrs[attr.name] = value
            nodes.append(
                {
                    "index": index,
                    "name": node.name,
                    "op": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attrs": attrs,
                    "output_consumers": {
                        output: consumers.get(output, []) for output in node.output
                    },
                }
            )

        declared = {}
        for value in [*graph.input, *graph.output, *graph.value_info]:
            declared[value.name] = {
                "elem_type": value.type.tensor_type.elem_type,
                "shape": value_shape(value),
            }

        output_names = {value.name for value in graph.output}
        dead_nodes = [
            node["index"]
            for node in nodes
            if all(
                not consumers.get(output) and output not in output_names
                for output in node["outputs"]
            )
        ]
        result[str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "op_counts": dict(Counter(node.op_type for node in graph.node)),
            "declared": declared,
            "initializers": initializers,
            "nodes": nodes,
            "dead_terminal_nodes": dead_nodes,
        }

    (LANE / "exact_graph_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

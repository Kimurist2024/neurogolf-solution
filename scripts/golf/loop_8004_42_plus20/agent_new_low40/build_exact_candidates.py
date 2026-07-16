#!/usr/bin/env python3
"""Find mathematically exact local rewrites for the low40 lane.

The scanner deliberately emits no model unless an equivalence proof exists.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
TARGETS = (22, 181, 104, 294, 128, 152, 203, 236)


def replace_uses(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def remove_value_info(model: onnx.ModelProto, names: set[str]) -> None:
    kept = [value for value in model.graph.value_info if value.name not in names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)


def initializer_key(item: onnx.TensorProto) -> bytes:
    clone = copy.deepcopy(item)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def scan_task(task: int) -> dict[str, object]:
    model = onnx.load(HERE / f"base/task{task:03d}.onnx")
    used = {name for node in model.graph.node for name in node.input}
    used.update(output.name for output in model.graph.output)
    unused = [item.name for item in model.graph.initializer if item.name not in used]
    duplicate_initializers = []
    for left in range(len(model.graph.initializer)):
        for right in range(left + 1, len(model.graph.initializer)):
            if initializer_key(model.graph.initializer[left]) == initializer_key(model.graph.initializer[right]):
                duplicate_initializers.append(
                    [model.graph.initializer[left].name, model.graph.initializer[right].name]
                )
    duplicate_nodes = []
    for left in range(len(model.graph.node)):
        for right in range(left + 1, len(model.graph.node)):
            if node_key(model.graph.node[left]) == node_key(model.graph.node[right]):
                duplicate_nodes.append([left, right])
    return {
        "task": task,
        "identity_nodes": [index for index, node in enumerate(model.graph.node) if node.op_type == "Identity"],
        "unused_initializers": unused,
        "duplicate_initializers": duplicate_initializers,
        "duplicate_deterministic_nodes": duplicate_nodes,
        "candidate_count": len(unused) + len(duplicate_initializers) + len(duplicate_nodes)
        + sum(node.op_type == "Identity" for node in model.graph.node),
    }


if __name__ == "__main__":
    rows = [scan_task(task) for task in TARGETS]
    result = {
        "targets": list(TARGETS),
        "rows": rows,
        "accepted_exact_candidates": [],
        "note": (
            "task294 has three ConstantOfShape(l) nodes, but their tensor attributes are "
            "different constants 31, 29, and 30, so they are not aliasable. The all-400 "
            "exact Einsum scan independently found no target hit."
        ),
    }
    destination = HERE / "exact_candidate_scan.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)

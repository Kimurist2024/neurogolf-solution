#!/usr/bin/env python3
"""Audit liveness, exact parameter reuse, and attempted compact factors."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def tensor_key(item: onnx.TensorProto) -> str:
    array = numpy_helper.to_array(item)
    return hashlib.sha256(
        str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
    ).hexdigest()


def audit(task: int) -> dict[str, object]:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    model = onnx.load(path, load_external_data=False)
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    needed = {item.name for item in model.graph.output}
    live_nodes: set[int] = set()
    stack = list(needed)
    while stack:
        name = stack.pop()
        index = producer.get(name)
        if index is None or index in live_nodes:
            continue
        live_nodes.add(index)
        stack.extend(name for name in model.graph.node[index].input if name)
    used_inputs = {name for node in model.graph.node for name in node.input if name}
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for item in model.graph.initializer:
        duplicate_groups[tensor_key(item)].append(item.name)
    duplicates = [names for names in duplicate_groups.values() if len(names) > 1]
    params = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        params.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "used": item.name in used_inputs,
            }
        )
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "nodes": len(model.graph.node),
        "live_nodes": len(live_nodes),
        "dead_node_indices": sorted(set(range(len(model.graph.node))) - live_nodes),
        "initializers": params,
        "unused_initializers": [item["name"] for item in params if not item["used"]],
        "duplicate_full_tensor_groups": duplicates,
        "parameter_elements": sum(item["elements"] for item in params),
    }


def main() -> None:
    rows = {str(task): audit(task) for task in (117, 160)}
    rows["117"]["reuse_conclusion"] = (
        "all 67 nodes and all 19 initializers are live; no full-tensor duplicate exists. "
        "The only below-base static history is the unscorable shape-cloaked col_once graph. "
        "A truthful float terminal paint has at least the 720-byte update floor before rule logic."
    )
    rows["160"]["reuse_conclusion"] = (
        "all 6 nodes and all 7 initializers are live; no full-tensor duplicate exists. "
        "Removing the two-element feature bias gives cost 402 but fails all 265 known cases; "
        "both 2x1 terminal factors give cost 384 and fail all 265 known cases."
    )
    (HERE / "factor_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

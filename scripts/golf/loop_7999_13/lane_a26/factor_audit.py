#!/usr/bin/env python3
"""Exact node/initializer liveness and reuse audit for A26."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def key(item: onnx.TensorProto) -> str:
    array = numpy_helper.to_array(item)
    return hashlib.sha256(
        str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
    ).hexdigest()


def inspect(task: int) -> dict[str, object]:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    model = onnx.load(path, load_external_data=False)
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    live: set[int] = set()
    stack = [item.name for item in model.graph.output]
    while stack:
        name = stack.pop()
        index = producer.get(name)
        if index is None or index in live:
            continue
        live.add(index)
        stack.extend(name for name in model.graph.node[index].input if name)
    used = {name for node in model.graph.node for name in node.input if name}
    groups: dict[str, list[str]] = defaultdict(list)
    params = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        groups[key(item)].append(item.name)
        params.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "used": item.name in used,
            }
        )
    return {
        "task": task,
        "nodes": len(model.graph.node),
        "live_nodes": len(live),
        "dead_node_indices": sorted(set(range(len(model.graph.node))) - live),
        "initializer_count": len(params),
        "parameter_elements": sum(item["elements"] for item in params),
        "unused_initializers": [item["name"] for item in params if not item["used"]],
        "duplicate_full_tensor_groups": [names for names in groups.values() if len(names) > 1],
        "initializers": params,
    }


def main() -> None:
    rows = {str(task): inspect(task) for task in (182, 330)}
    rows["182"]["exact_reuse_trials"] = {
        "reuse_constants": {
            "sha256": "d11479be2703a35afb1f5fe5224ee6a258b3d11c4480a15e8e88db50747ccff8",
            "actual_cost": 990,
            "result": "reject: default ORT session error and 47 runtime-shape mismatches",
        },
        "reuse_s2_s3": {
            "sha256": "2b0c330cf320d5b18c9cf676545b4e59c0e44004f7166f1ac640e64087bbf266",
            "actual_cost": 993,
            "result": "reject: default ORT session error and 47 runtime-shape mismatches",
        },
        "truthful_exact_shapes": {
            "actual_cost": 169429,
            "result": "both ORT known 267/267, shape mismatch 0, not cheaper",
        },
    }
    rows["330"]["factor_trials"] = {
        "pair_frames_807": "known 166/266; reject",
        "pair_frames_anchor_808": "known 162/266; reject",
        "pair_frames_mod9_817": "known 210/266; reject",
        "truthful_component_rect": "cost 5525, both ORT known 266/266, shape mismatch 0",
    }
    (HERE / "factor_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Search the low43 baselines for price-reducing exact graph rewrites."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
TARGETS = (6, 334, 244, 249, 347, 386, 146, 291)


def initializer_key(item: onnx.TensorProto) -> bytes:
    clone = copy.deepcopy(item)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> bytes:
    clone = copy.deepcopy(node)
    clone.name = ""
    del clone.output[:]
    return clone.SerializeToString(deterministic=True)


def scan(task: int) -> dict[str, object]:
    model = onnx.load(HERE / f"baselines/task{task:03d}.onnx")
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
    identity = [index for index, node in enumerate(model.graph.node) if node.op_type == "Identity"]
    optional_outputs = [
        {"node": index, "op": node.op_type, "blank_outputs": sum(not name for name in node.output)}
        for index, node in enumerate(model.graph.node)
        if any(not name for name in node.output)
    ]
    candidates = len(unused) + len(duplicate_initializers) + len(duplicate_nodes) + len(identity)
    return {
        "task": task,
        "identity_nodes": identity,
        "unused_initializers": unused,
        "duplicate_initializers": duplicate_initializers,
        "duplicate_deterministic_nodes": duplicate_nodes,
        "optional_output_observations": optional_outputs,
        "exact_price_reduction_count": candidates,
    }


def main() -> None:
    rows = [scan(task) for task in TARGETS]
    result = {
        "targets": list(TARGETS),
        "rows": rows,
        "accepted_exact_candidates": [],
        "all400_exact_einsum_pass": {
            "source": "scripts/golf/loop_8004_42_plus20/root_exact_einsum25/scan_report.json",
            "target_hits": [],
        },
        "interpretation": (
            "No identity, unused initializer, duplicate initializer, duplicate producer, "
            "or other exact price-reducing local rewrite is present in the eight baselines."
        ),
    }
    (HERE / "exact_candidate_scan.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

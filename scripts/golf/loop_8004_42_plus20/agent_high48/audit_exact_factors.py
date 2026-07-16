#!/usr/bin/env python3
"""Find safe exact dead/duplicate opportunities in the high48 incumbents."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
TARGETS = (8, 275, 134, 112, 168, 109, 160, 170)


def init_key(item: onnx.TensorProto) -> tuple[int, tuple[int, ...], bytes]:
    array = numpy_helper.to_array(item)
    return int(item.data_type), tuple(int(value) for value in item.dims), array.tobytes()


def node_key(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.CopyFrom(node)
    del clone.output[:]
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def main() -> None:
    rows = []
    for task in TARGETS:
        path = HERE / "baselines" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        uses = collections.Counter(name for node in model.graph.node for name in node.input if name)
        outputs = {item.name for item in model.graph.output}
        dead_initializers = [item.name for item in model.graph.initializer if uses[item.name] == 0]
        groups: dict[tuple[int, tuple[int, ...], bytes], list[str]] = collections.defaultdict(list)
        for item in model.graph.initializer:
            groups[init_key(item)].append(item.name)
        duplicate_initializers = [names for names in groups.values() if len(names) > 1]
        nodes: dict[bytes, list[dict[str, object]]] = collections.defaultdict(list)
        for index, node in enumerate(model.graph.node):
            nodes[node_key(node)].append(
                {"index": index, "op": node.op_type, "outputs": list(node.output)}
            )
        duplicate_nodes = [group for group in nodes.values() if len(group) > 1]
        dead_node_outputs = [
            name
            for node in model.graph.node
            for name in node.output
            if name and uses[name] == 0 and name not in outputs
        ]
        removable_dead_nodes = [
            {"index": index, "op": node.op_type, "outputs": list(node.output)}
            for index, node in enumerate(model.graph.node)
            if node.output
            and all(name and uses[name] == 0 and name not in outputs for name in node.output)
        ]
        rows.append(
            {
                "task": task,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "dead_initializers": dead_initializers,
                "duplicate_initializer_groups": duplicate_initializers,
                "duplicate_node_groups": duplicate_nodes,
                "dead_node_outputs": dead_node_outputs,
                "removable_dead_nodes": removable_dead_nodes,
                "required_multi_output_dead_outputs": [
                    name
                    for node in model.graph.node
                    if any(name in dead_node_outputs for name in node.output)
                    and not all(name in dead_node_outputs for name in node.output)
                    for name in node.output
                    if name in dead_node_outputs
                ],
                "exact_safe_opportunity": bool(
                    dead_initializers
                    or duplicate_initializers
                    or duplicate_nodes
                    or removable_dead_nodes
                ),
            }
        )
    (HERE / "exact_factor_audit.json").write_text(
        json.dumps(
            {
                "targets_completed": len(rows),
                "opportunities": [row for row in rows if row["exact_safe_opportunity"]],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            f"task{row['task']:03d} opportunity={row['exact_safe_opportunity']} "
            f"dead_init={len(row['dead_initializers'])} dup_init={len(row['duplicate_initializer_groups'])} "
            f"dup_node={len(row['duplicate_node_groups'])} removable_dead={len(row['removable_dead_nodes'])}"
        )


if __name__ == "__main__":
    main()

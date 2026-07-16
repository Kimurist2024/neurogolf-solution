#!/usr/bin/env python3
"""Read-only exact-CSE opportunity inventory across all 400 authority models."""

from __future__ import annotations

import copy
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx

import scan_build_exact_cse as shared


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"


def resolve(name: str, aliases: dict[str, str]) -> str:
    while name in aliases:
        name = aliases[name]
    return name


def audit(model: onnx.ModelProto) -> dict[str, object]:
    reached = shared.reachable_nodes(model)
    graph_outputs = {value.name for value in model.graph.output}
    reachable_inputs = {
        name
        for index in reached
        for name in model.graph.node[index].input
        if name
    }
    groups: dict[tuple[int, tuple[int, ...], bytes], list[onnx.TensorProto]] = defaultdict(list)
    for initializer in model.graph.initializer:
        if initializer.name in reachable_inputs or initializer.name in graph_outputs:
            groups[shared.tensor_key(initializer)].append(initializer)
    initializer_aliases: dict[str, str] = {}
    duplicate_initializer_groups: list[dict[str, object]] = []
    for tensors in groups.values():
        if len(tensors) < 2:
            continue
        canonical = next((item for item in tensors if item.name in graph_outputs), tensors[0])
        removed = [item for item in tensors if item.name != canonical.name and item.name not in graph_outputs]
        if not removed:
            continue
        for item in removed:
            initializer_aliases[item.name] = canonical.name
        duplicate_initializer_groups.append(
            {
                "canonical": canonical.name,
                "duplicates": [item.name for item in removed],
                "elements": sum(
                    int(np.prod(item.dims, dtype=np.int64)) if item.dims else 1
                    for item in removed
                ),
            }
        )

    nodes = [copy.deepcopy(node) for node in model.graph.node]
    for node in nodes:
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name, initializer_aliases)
    canonical_nodes: dict[tuple[str, str, tuple[str, ...], tuple[bytes, ...]], str] = {}
    node_aliases: dict[str, str] = {}
    duplicate_nodes: list[dict[str, object]] = []
    for index, node in enumerate(nodes):
        for input_index, name in enumerate(node.input):
            node.input[input_index] = resolve(name, node_aliases)
        eligible = (
            index in reached
            and len(node.output) == 1
            and node.output[0] not in graph_outputs
            and node.op_type not in shared.NONDETERMINISTIC
            and all(
                attribute.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attribute in node.attribute
            )
        )
        if not eligible:
            continue
        signature = shared.node_key(node)
        if signature in canonical_nodes:
            replacement = canonical_nodes[signature]
            node_aliases[node.output[0]] = replacement
            duplicate_nodes.append(
                {
                    "removed": node.output[0],
                    "replacement": replacement,
                    "op": node.op_type,
                    "constant_payload_cse": node.op_type == "Constant",
                }
            )
        else:
            canonical_nodes[signature] = node.output[0]
    return {
        "initializer_groups": duplicate_initializer_groups,
        "initializer_aliases": len(initializer_aliases),
        "initializer_elements": sum(int(group["elements"]) for group in duplicate_initializer_groups),
        "constant_payload_cse": sum(bool(row["constant_payload_cse"]) for row in duplicate_nodes),
        "deterministic_node_cse": sum(not bool(row["constant_payload_cse"]) for row in duplicate_nodes),
        "node_changes": duplicate_nodes,
    }


def main() -> None:
    manifest = json.loads((HERE / "scan_build_manifest.json").read_text())
    excluded = {int(row["task"]): row["reasons"] for row in manifest["excluded"]}
    rows: list[dict[str, object]] = []
    safe_opportunity_tasks: list[int] = []
    excluded_opportunity_tasks: list[int] = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            opportunity = audit(model)
            count = (
                int(opportunity["initializer_aliases"])
                + int(opportunity["constant_payload_cse"])
                + int(opportunity["deterministic_node_cse"])
            )
            if not count:
                continue
            reasons = excluded.get(task, [])
            row = {
                "task": task,
                "priority_150_plus": task >= 150,
                "excluded": bool(reasons),
                "exclusion_reasons": reasons,
                **opportunity,
            }
            rows.append(row)
            if reasons:
                excluded_opportunity_tasks.append(task)
            else:
                safe_opportunity_tasks.append(task)
    rows.sort(key=lambda row: (not bool(row["priority_150_plus"]), int(row["task"])))
    result = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "task_count": 400,
        "opportunity_task_count": len(rows),
        "safe_opportunity_tasks": safe_opportunity_tasks,
        "excluded_opportunity_tasks": excluded_opportunity_tasks,
        "rows": rows,
    }
    (HERE / "raw_opportunity_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "opportunity_task_count": len(rows),
        "safe_opportunity_tasks": safe_opportunity_tasks,
        "excluded_opportunity_tasks": excluded_opportunity_tasks,
        "initializer_aliases": sum(int(row["initializer_aliases"]) for row in rows),
        "constant_payload_cse": sum(int(row["constant_payload_cse"]) for row in rows),
        "deterministic_node_cse": sum(int(row["deterministic_node_cse"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()

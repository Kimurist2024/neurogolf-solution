#!/usr/bin/env python3
"""Inventory initializers, uses, aliases, defaults, attributes, and CastLike refs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
TASKS = (2, 12, 107)


def main() -> int:
    report: dict[str, object] = {
        "cost_rule": "initializer parameter cost is element count; all rewrites still require actual runtime profiling",
        "tasks": {},
    }
    for task in TASKS:
        model = onnx.load(HERE / f"current/task{task:03d}.onnx")
        consumers: dict[str, list[dict[str, object]]] = defaultdict(list)
        for index, node in enumerate(model.graph.node):
            for input_index, name in enumerate(node.input):
                if name:
                    consumers[name].append(
                        {"node_index": index, "op": node.op_type, "input_index": input_index}
                    )
        rows = []
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            uses = consumers.get(init.name, [])
            castlike_type_uses = [
                use for use in uses if use["op"] == "CastLike" and use["input_index"] == 1
            ]
            rows.append({
                "name": init.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "uses": uses,
                "castlike_type_reference_only": bool(castlike_type_uses and len(castlike_type_uses) == len(uses)),
            })
        aliases = []
        for index, left in enumerate(model.graph.initializer):
            left_array = numpy_helper.to_array(left)
            for right in model.graph.initializer[index + 1:]:
                right_array = numpy_helper.to_array(right)
                if (left_array.dtype == right_array.dtype and left_array.shape == right_array.shape
                        and left_array.tobytes() == right_array.tobytes()):
                    aliases.append([left.name, right.name])
        attrs = []
        default_input_slots = []
        for node_index, node in enumerate(model.graph.node):
            attrs.append({
                "node_index": node_index,
                "name": node.name,
                "op": node.op_type,
                "attributes": [attribute.name for attribute in node.attribute],
            })
            for input_index, name in enumerate(node.input):
                if name == "":
                    default_input_slots.append({
                        "node_index": node_index,
                        "name": node.name,
                        "op": node.op_type,
                        "input_index": input_index,
                    })
        report["tasks"][str(task)] = {
            "initializer_count": len(rows),
            "parameter_elements": sum(row["elements"] for row in rows),
            "aliases": aliases,
            "castlike_type_reference_only": [
                row["name"] for row in rows if row["castlike_type_reference_only"]
            ],
            "unused": [row["name"] for row in rows if not row["uses"]],
            "default_input_slots": default_input_slots,
            "nodes_with_attributes": attrs,
            "initializers": rows,
        }
    (HERE / "initializer_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze initializer uses and exact aliases for high154."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
TASKS = (367, 382, 270)


def main() -> int:
    report: dict[str, object] = {
        "cost_rule": "initializer parameter cost is element-count, independent of dtype byte width",
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
            type_uses = [
                use for use in uses if use["op"] == "CastLike" and use["input_index"] == 1
            ]
            other = [use for use in uses if use not in type_uses]
            rows.append({
                "name": init.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "uses": uses,
                "removable_by_castlike_to_cast": bool(type_uses and not other),
            })
        aliases = []
        for index, left in enumerate(model.graph.initializer):
            left_array = numpy_helper.to_array(left)
            for right in model.graph.initializer[index + 1:]:
                right_array = numpy_helper.to_array(right)
                if (left_array.dtype == right_array.dtype and left_array.shape == right_array.shape
                        and left_array.tobytes() == right_array.tobytes()):
                    aliases.append([left.name, right.name])
        report["tasks"][str(task)] = {
            "initializer_count": len(rows),
            "parameter_elements": sum(row["elements"] for row in rows),
            "aliases": aliases,
            "castlike_only": [row["name"] for row in rows if row["removable_by_castlike_to_cast"]],
            "unused": [row["name"] for row in rows if not row["uses"]],
            "initializers": rows,
        }
    (HERE / "initializer_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

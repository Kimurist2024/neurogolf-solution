#!/usr/bin/env python3
"""Audit initializer aliases, type-only CastLike uses, and dead values."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent


def main() -> int:
    report: dict[str, object] = {
        "cost_rule": "initializer parameter cost is element-count, independent of dtype byte width",
        "tasks": {},
    }
    for task in (216, 285, 226):
        model = onnx.load(HERE / f"current/task{task:03d}.onnx")
        consumers: dict[str, list[dict[str, object]]] = defaultdict(list)
        for index, node in enumerate(model.graph.node):
            for input_index, name in enumerate(node.input):
                if name:
                    consumers[name].append({"node_index": index, "op": node.op_type, "input_index": input_index})
        rows = []
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            uses = consumers.get(init.name, [])
            type_uses = [u for u in uses if u["op"] == "CastLike" and u["input_index"] == 1]
            other = [u for u in uses if u not in type_uses]
            rows.append({
                "name": init.name, "dtype": str(array.dtype), "shape": list(array.shape),
                "elements": int(array.size), "uses": uses,
                "removable_by_castlike_to_cast": bool(type_uses and not other),
            })
        aliases = []
        for i, left in enumerate(model.graph.initializer):
            la = numpy_helper.to_array(left)
            for right in model.graph.initializer[i + 1:]:
                ra = numpy_helper.to_array(right)
                if la.dtype == ra.dtype and la.shape == ra.shape and la.tobytes() == ra.tobytes():
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

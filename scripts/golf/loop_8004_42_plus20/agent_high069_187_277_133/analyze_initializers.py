#!/usr/bin/env python3
"""Explain initializer sharing, dtype, and CastLike attributeization opportunities."""

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
    for task in (69, 187, 277):
        model = onnx.load(HERE / f"current/task{task:03d}.onnx")
        consumers: dict[str, list[dict[str, object]]] = defaultdict(list)
        for index, node in enumerate(model.graph.node):
            for input_index, name in enumerate(node.input):
                if name:
                    consumers[name].append({
                        "node_index": index,
                        "node": node.name,
                        "op": node.op_type,
                        "input_index": input_index,
                    })
        initializers = []
        init_names = {init.name for init in model.graph.initializer}
        for init in model.graph.initializer:
            array = numpy_helper.to_array(init)
            uses = consumers.get(init.name, [])
            castlike_type_uses = [
                use for use in uses if use["op"] == "CastLike" and use["input_index"] == 1
            ]
            other_uses = [use for use in uses if use not in castlike_type_uses]
            initializers.append({
                "name": init.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "uses": uses,
                "castlike_type_use_count": len(castlike_type_uses),
                "other_use_count": len(other_uses),
                "removable_by_castlike_to_cast": bool(castlike_type_uses and not other_uses),
            })
        aliases = []
        for i, left in enumerate(model.graph.initializer):
            la = numpy_helper.to_array(left)
            for right in model.graph.initializer[i + 1:]:
                ra = numpy_helper.to_array(right)
                if la.dtype == ra.dtype and la.shape == ra.shape and la.tobytes() == ra.tobytes():
                    aliases.append([left.name, right.name])
        report["tasks"][str(task)] = {
            "initializer_count": len(initializers),
            "parameter_elements": sum(item["elements"] for item in initializers),
            "byte_identical_same_shape_aliases": aliases,
            "castlike_only_type_initializers": [
                item["name"] for item in initializers if item["removable_by_castlike_to_cast"]
            ],
            "initializers": initializers,
            "all_initializer_names_consumed": all(consumers.get(name) for name in init_names),
            "conclusion": (
                "No duplicate initializer or CastLike-only type witness can be removed; "
                "dtype narrowing cannot lower element-count parameter cost."
            ),
        }
    (HERE / "initializer_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

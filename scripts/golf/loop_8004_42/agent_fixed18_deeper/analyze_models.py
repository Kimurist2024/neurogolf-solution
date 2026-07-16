#!/usr/bin/env python3
"""Static anatomy report for the 18 LB-fixed models.

The report is intentionally read-only.  It finds only graph-local, semantics-
preserving shaving opportunities: unreachable nodes, unused initializers, and
byte-identical initializers that can potentially be shared.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
CURRENT = Path(__file__).resolve().parent / "current"


def reachable_node_indices(model: onnx.ModelProto) -> set[int]:
    producers = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    needed = [output.name for output in model.graph.output]
    reachable: set[int] = set()
    while needed:
        name = needed.pop()
        index = producers.get(name)
        if index is None or index in reachable:
            continue
        reachable.add(index)
        needed.extend(value for value in model.graph.node[index].input if value)
    return reachable


def analyze(path: Path) -> dict[str, object]:
    model = onnx.load(str(path))
    reachable = reachable_node_indices(model)
    used_names = {
        value
        for index in reachable
        for value in model.graph.node[index].input
        if value
    }
    unused_initializers = [
        init.name for init in model.graph.initializer if init.name not in used_names
    ]
    consumers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            consumers[name].append((node.op_type, input_index))
    castlike_only_initializers = [
        init.name
        for init in model.graph.initializer
        if consumers[init.name]
        and all(op_type == "CastLike" and input_index == 1 for op_type, input_index in consumers[init.name])
    ]

    by_payload: dict[tuple[int, tuple[int, ...], bytes], list[str]] = defaultdict(list)
    for init in model.graph.initializer:
        key = (init.data_type, tuple(init.dims), init.raw_data or init.SerializeToString())
        by_payload[key].append(init.name)
    duplicate_groups = [names for names in by_payload.values() if len(names) > 1]

    return {
        "path": str(path.relative_to(ROOT)),
        "nodes": len(model.graph.node),
        "op_hist": dict(Counter(node.op_type for node in model.graph.node)),
        "initializers": len(model.graph.initializer),
        "initializer_params": sum(
            max(1, __import__("math").prod(init.dims)) for init in model.graph.initializer
        ),
        "unreachable_node_indices": sorted(set(range(len(model.graph.node))) - reachable),
        "unused_initializers": unused_initializers,
        "castlike_only_initializers": castlike_only_initializers,
        "duplicate_initializer_groups": duplicate_groups,
        "output_types": [output.type.tensor_type.elem_type for output in model.graph.output],
        "opsets": {item.domain: item.version for item in model.opset_import},
    }


def main() -> int:
    rows = [analyze(path) for path in sorted(CURRENT.glob("task*.onnx"))]
    out = Path(__file__).resolve().parent / "static_anatomy.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{Path(str(row['path'])).stem}: nodes={row['nodes']} "
            f"params={row['initializer_params']} dead={row['unreachable_node_indices']} "
            f"unused={row['unused_initializers']} carriers={row['castlike_only_initializers']} "
            f"dup={row['duplicate_initializer_groups']}"
        )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

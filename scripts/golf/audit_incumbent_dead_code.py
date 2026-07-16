#!/usr/bin/env python3
"""Find output-unreachable nodes and initializers in relaxed95 incumbents."""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "artifacts/relaxed95_loop/incumbents.json"


def walk_records(value: Any):
    if isinstance(value, dict):
        if "task" in value and "path" in value:
            yield value
        for child in value.values():
            yield from walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_records(child)


def incumbent_paths(costs: dict[str, int]) -> dict[int, Path]:
    choices: dict[int, list[Path]] = {}
    for filename in glob.glob(str(ROOT / "artifacts/relaxed95_loop/round*.json")):
        data = json.loads(Path(filename).read_text())
        for record in walk_records(data):
            try:
                task = int(record["task"])
                cost = int(
                    record.get(
                        "candidate_cost", record.get("cost", record.get("new_cost"))
                    )
                )
            except (TypeError, ValueError):
                continue
            path = ROOT / record["path"]
            if str(task) in costs and cost == int(costs[str(task)]) and path.is_file():
                choices.setdefault(task, []).append(path)

    for path in (ROOT / "artifacts/relaxed95_loop/zip_candidates").glob(
        "task*_cost*.onnx"
    ):
        task = int(path.name[4:7])
        if str(task) in costs:
            choices.setdefault(task, []).insert(0, path)

    for task in map(int, costs):
        handcrafted = ROOT / f"artifacts/handcrafted/task{task:03d}.onnx"
        if handcrafted.is_file():
            choices.setdefault(task, []).append(handcrafted)

    return {task: paths[0] for task, paths in choices.items() if paths}


def dead_code(model: onnx.ModelProto):
    needed = {output.name for output in model.graph.output}
    live_nodes: set[int] = set()
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(output and output in needed for output in node.output):
            live_nodes.add(index)
            needed.update(value for value in node.input if value)
    dead_nodes = [
        (index, node.op_type, list(node.output))
        for index, node in enumerate(model.graph.node)
        if index not in live_nodes
    ]
    unused_initializers = [
        (initializer.name, math.prod(initializer.dims))
        for initializer in model.graph.initializer
        if initializer.name not in needed
    ]
    return dead_nodes, unused_initializers


def main() -> None:
    costs = json.loads(LEDGER.read_text())["costs"]
    paths = incumbent_paths(costs)
    print(f"resolved={len(paths)}/{len(costs)}")
    for task in sorted(paths):
        model = onnx.load(paths[task])
        dead_nodes, unused_initializers = dead_code(model)
        if dead_nodes or unused_initializers:
            savings = sum(size for _, size in unused_initializers)
            print(
                f"task{task:03d} cost={costs[str(task)]} path={paths[task]} "
                f"dead_nodes={dead_nodes} unused_initializers={unused_initializers} "
                f"param_floor_gain={savings}"
            )


if __name__ == "__main__":
    main()

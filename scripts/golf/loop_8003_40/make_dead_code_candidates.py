#!/usr/bin/env python3
"""Remove output-unreachable ONNX nodes and initializers from selected models.

The transform is deliberately narrow: it only retains the backwards slice from
the declared graph outputs.  ONNX graph nodes are side-effect free, so this is
an exact rewrite for every input on which the source model produces an output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backwards_slice(model: onnx.ModelProto) -> tuple[list[int], set[str]]:
    needed = {output.name for output in model.graph.output}
    live: list[int] = []
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(output and output in needed for output in node.output):
            live.append(index)
            needed.update(value for value in node.input if value)
    live.reverse()
    return live, needed


def rewrite(source: Path, destination: Path) -> dict[str, object]:
    model = onnx.load(source, load_external_data=False)
    live, needed = backwards_slice(model)
    live_set = set(live)

    dead_nodes = [
        {
            "index": index,
            "op_type": node.op_type,
            "outputs": list(node.output),
        }
        for index, node in enumerate(model.graph.node)
        if index not in live_set
    ]
    unused_initializers = [
        {
            "name": initializer.name,
            "elements": math.prod(initializer.dims),
        }
        for initializer in model.graph.initializer
        if initializer.name not in needed
    ]

    kept_nodes = [model.graph.node[index] for index in live]
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in needed
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)

    return {
        "source": str(source),
        "destination": str(destination),
        "source_sha256": sha256(source),
        "candidate_sha256": sha256(destination),
        "dead_nodes": dead_nodes,
        "unused_initializers": unused_initializers,
        "strict_checker": "PASS",
    }


def parse_tasks(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=parse_tasks, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for task in args.tasks:
        source = args.base_dir / f"task{task:03d}.onnx"
        destination = args.out_dir / source.name
        record = rewrite(source, destination)
        record["task"] = task
        records.append(record)
        print(
            f"task{task:03d}: removed {len(record['dead_nodes'])} nodes, "
            f"{len(record['unused_initializers'])} initializers"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"candidates": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()

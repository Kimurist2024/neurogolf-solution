#!/usr/bin/env python3
"""Audit duplicate Constant tensors and exact duplicate producer nodes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import onnx


def constant_payload(node: onnx.NodeProto) -> bytes | None:
    if node.op_type != "Constant" or node.domain:
        return None
    # Attribute order is irrelevant.  Deterministically serialize a copy.
    clone = onnx.NodeProto()
    clone.op_type = node.op_type
    clone.domain = node.domain
    clone.attribute.extend(sorted(node.attribute, key=lambda attr: attr.name))
    return clone.SerializeToString(deterministic=True)


def node_signature(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.op_type = node.op_type
    clone.domain = node.domain
    clone.input.extend(node.input)
    clone.attribute.extend(sorted(node.attribute, key=lambda attr: attr.name))
    return clone.SerializeToString(deterministic=True)


def audit(task: int, model: onnx.ModelProto) -> dict[str, object] | None:
    constants: dict[bytes, list[str]] = defaultdict(list)
    producers: dict[bytes, list[str]] = defaultdict(list)
    graph_outputs = {item.name for item in model.graph.output}
    for node in model.graph.node:
        payload = constant_payload(node)
        if payload is not None and len(node.output) == 1:
            constants[payload].append(node.output[0])
        if len(node.output) == 1 and node.output[0] not in graph_outputs:
            producers[node_signature(node)].append(node.output[0])
    constant_groups = [names for names in constants.values() if len(names) > 1]
    producer_groups = [names for names in producers.values() if len(names) > 1]
    if not constant_groups and not producer_groups:
        return None
    return {
        "task": task,
        "duplicate_constant_groups": constant_groups,
        "duplicate_producer_groups": producer_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/golf/loop_7999_13/duplicate_constants_audit.json"),
    )
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            payload = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_model(io.BytesIO(payload))
            row = audit(task, model)
            if row:
                row["sha256"] = hashlib.sha256(payload).hexdigest()
                rows.append(row)
    result = {"source_zip": str(args.zip), "candidate_task_count": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

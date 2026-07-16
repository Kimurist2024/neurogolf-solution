#!/usr/bin/env python3
"""Build semantics-preserving common-subexpression candidates for B20.

This is deliberately conservative: only single-output nodes with identical
operator/domain/inputs/attributes are merged.  Initializers and graph I/O are
left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx


def attr_key(attr: onnx.AttributeProto) -> bytes:
    copy = onnx.AttributeProto()
    copy.CopyFrom(attr)
    return copy.SerializeToString(deterministic=True)


def node_key(node: onnx.NodeProto) -> tuple[object, ...]:
    return (
        node.domain,
        node.op_type,
        tuple(node.input),
        tuple(sorted((attr.name, attr_key(attr)) for attr in node.attribute)),
    )


def rewrite_inputs(node: onnx.NodeProto, aliases: dict[str, str]) -> None:
    for index, name in enumerate(node.input):
        while name in aliases:
            name = aliases[name]
        node.input[index] = name


def build(source: Path, output: Path) -> dict[str, object]:
    model = onnx.load(source)
    graph_outputs = {value.name for value in model.graph.output}
    aliases: dict[str, str] = {}
    seen: dict[tuple[object, ...], str] = {}
    kept: list[onnx.NodeProto] = []
    removed: list[dict[str, object]] = []

    for node in model.graph.node:
        rewrite_inputs(node, aliases)
        if len(node.output) != 1 or node.output[0] in graph_outputs:
            kept.append(node)
            continue
        key = node_key(node)
        previous = seen.get(key)
        if previous is None:
            seen[key] = node.output[0]
            kept.append(node)
            continue
        aliases[node.output[0]] = previous
        removed.append(
            {
                "op_type": node.op_type,
                "output": node.output[0],
                "reused": previous,
            }
        )

    for node in kept:
        rewrite_inputs(node, aliases)
    model.graph.ClearField("node")
    model.graph.node.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return {
        "source": str(source),
        "output": str(output),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "nodes_before": len(kept) + len(removed),
        "nodes_after": len(kept),
        "removed_count": len(removed),
        "removed": removed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.output)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

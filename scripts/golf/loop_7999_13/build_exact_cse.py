#!/usr/bin/env python3
"""Build exact common-subexpression candidates with stale alias metadata removed."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import onnx


def signature(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.op_type = node.op_type
    clone.domain = node.domain
    clone.input.extend(node.input)
    clone.attribute.extend(sorted(node.attribute, key=lambda attr: attr.name))
    return clone.SerializeToString(deterministic=True)


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, str]]]:
    graph_outputs = {item.name for item in model.graph.output}
    canonical: dict[bytes, str] = {}
    replacements: dict[str, str] = {}
    kept: list[onnx.NodeProto] = []
    changes: list[dict[str, str]] = []
    for source in model.graph.node:
        node = onnx.NodeProto()
        node.CopyFrom(source)
        for index, name in enumerate(node.input):
            while name in replacements:
                name = replacements[name]
            node.input[index] = name
        if len(node.output) == 1 and node.output[0] not in graph_outputs:
            node_key = signature(node)
            if node_key in canonical:
                replacements[node.output[0]] = canonical[node_key]
                changes.append({"removed": node.output[0], "replacement": canonical[node_key]})
                continue
            canonical[node_key] = node.output[0]
        kept.append(node)

    # Resolve aliases in all surviving consumers (including aliases discovered
    # after an earlier consumer copy) and discard stale value_info for aliases.
    for node in kept:
        for index, name in enumerate(node.input):
            while name in replacements:
                name = replacements[name]
            node.input[index] = name
    del model.graph.node[:]
    model.graph.node.extend(kept)
    retained_vi = [value for value in model.graph.value_info if value.name not in replacements]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_vi)
    onnx.checker.check_model(model)
    return model, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("submission_base_7999.13.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="162,165,169")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in (int(item) for item in args.tasks.split(",") if item):
            model = onnx.load_model(io.BytesIO(archive.read(f"task{task:03d}.onnx")))
            candidate, changes = build(model)
            output = args.output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, output)
            rows.append({"task": task, "path": str(output), "changes": changes})
    (args.output_dir / "build_manifest.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

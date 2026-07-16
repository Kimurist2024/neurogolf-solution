#!/usr/bin/env python3
"""Build behavior-preserving graph simplification probes for task366."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline_task366.onnx"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attr_key(node: onnx.NodeProto) -> bytes:
    clone = onnx.NodeProto()
    clone.CopyFrom(node)
    del clone.input[:]
    del clone.output[:]
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


source = onnx.load(SOURCE)
inferred = onnx.shape_inference.infer_shapes(source, strict_mode=True, data_prop=True)
types = {}
for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
    if value.type.HasField("tensor_type"):
        types[value.name] = value.type.tensor_type.elem_type

identity_outputs = {
    node.output[0]: node.input[0]
    for node in source.graph.node
    if node.op_type == "Identity" and len(node.input) == len(node.output) == 1
}

same_type_cast_outputs = {}
for node in source.graph.node:
    if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
        continue
    to = next((attr.i for attr in node.attribute if attr.name == "to"), None)
    if to is not None and types.get(node.input[0]) == to:
        same_type_cast_outputs[node.output[0]] = node.input[0]

same_type_castlike_outputs = {}
for node in source.graph.node:
    if node.op_type != "CastLike" or len(node.input) != 2 or len(node.output) != 1:
        continue
    if types.get(node.input[0]) == types.get(node.input[1]):
        same_type_castlike_outputs[node.output[0]] = node.input[0]

# Structural CSE: exact same operator, ordered inputs, domain, and attributes.
seen = {}
cse_outputs = {}
for node in source.graph.node:
    if len(node.output) != 1 or not node.output[0]:
        continue
    key = (node.domain, node.op_type, tuple(node.input), attr_key(node))
    if key in seen:
        cse_outputs[node.output[0]] = seen[key]
    else:
        seen[key] = node.output[0]


def resolve(name: str, replacements: dict[str, str]) -> str:
    seen_names = set()
    while name in replacements and name not in seen_names:
        seen_names.add(name)
        name = replacements[name]
    return name


def build(label: str, replacements: dict[str, str]) -> dict[str, object]:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    removed = set(replacements)
    kept = []
    removed_nodes = []
    for index, node in enumerate(model.graph.node):
        if any(output in removed for output in node.output):
            removed_nodes.append({"index": index, "op": node.op_type, "outputs": list(node.output)})
            continue
        for i, name in enumerate(node.input):
            node.input[i] = resolve(name, replacements)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    for output in model.graph.output:
        output.name = resolve(output.name, replacements)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path = HERE / f"probe_{label}.onnx"
    onnx.save(model, path)
    return {
        "label": label,
        "path": path.name,
        "sha256": digest(path),
        "replacements": replacements,
        "removed_nodes": removed_nodes,
        "node_count": len(model.graph.node),
    }


probes = []
for label, replacements in (
    ("identity", identity_outputs),
    ("same_type_cast", same_type_cast_outputs),
    ("same_type_castlike", same_type_castlike_outputs),
    ("cse", cse_outputs),
):
    if replacements:
        probes.append(build(label, replacements))

combined = {}
for mapping in (identity_outputs, same_type_cast_outputs, same_type_castlike_outputs, cse_outputs):
    combined.update(mapping)
if combined:
    probes.append(build("combined_exact", combined))

manifest = {
    "source": SOURCE.name,
    "source_sha256": digest(SOURCE),
    "counts": {
        "identity": len(identity_outputs),
        "same_type_cast": len(same_type_cast_outputs),
        "same_type_castlike": len(same_type_castlike_outputs),
        "cse": len(cse_outputs),
    },
    "probes": probes,
}
(HERE / "exact_probe_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["counts"]))

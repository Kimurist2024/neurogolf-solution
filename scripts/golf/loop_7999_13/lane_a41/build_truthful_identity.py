#!/usr/bin/env python3
"""Remove the sole Identity while retaining repaired, explicit tensor shapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
source = onnx.load(HERE / "truthful_annotation_control.onnx")
identities = [node for node in source.graph.node if node.op_type == "Identity"]
if len(identities) != 1:
    raise RuntimeError(f"expected one Identity, found {len(identities)}")
identity = identities[0]
old, replacement = identity.output[0], identity.input[0]

kept = []
for node in source.graph.node:
    if node is identity:
        continue
    for index, name in enumerate(node.input):
        if name == old:
            node.input[index] = replacement
    kept.append(node)
del source.graph.node[:]
source.graph.node.extend(kept)
for output in source.graph.output:
    if output.name == old:
        output.name = replacement
kept_vi = [value for value in source.graph.value_info if value.name != old]
del source.graph.value_info[:]
source.graph.value_info.extend(kept_vi)

onnx.checker.check_model(source, full_check=True)
onnx.shape_inference.infer_shapes(source, strict_mode=True, data_prop=True)
output = HERE / "truthful_identity_bypass.onnx"
onnx.save(source, output)
(HERE / "truthful_identity_manifest.json").write_text(
    json.dumps(
        {
            "output": output.name,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "removed": {"op": "Identity", "output": old, "input": replacement},
            "node_count": len(source.graph.node),
            "value_info_count": len(source.graph.value_info),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

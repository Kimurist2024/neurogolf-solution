#!/usr/bin/env python3
"""Test standard Attention causal masking in place of task025 mask_line."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task025.onnx"
OUTPUT = HERE / "task025_causal_mask.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph
    changed = []
    for node in graph.node:
        if node.op_type != "Attention" or len(node.input) < 4 or node.input[3] != "mask_line":
            continue
        node.input[3] = ""
        node.attribute.append(helper.make_attribute("is_causal", 1))
        changed.append(node.output[0])
    assert changed == ["vcolg", "vleftq_58", "hcolg", "hleftq_98"]
    kept = [item for item in graph.initializer if item.name != "mask_line"]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUTPUT)
    payload = {
        "candidate": str(OUTPUT.relative_to(HERE.parents[3])),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "changed_attention_outputs": changed,
        "removed_initializer": "mask_line:float64[1,1,1,2]",
        "parameter_reduction": 2,
    }
    (HERE / "task025_causal_build.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

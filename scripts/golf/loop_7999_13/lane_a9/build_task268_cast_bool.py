#!/usr/bin/env python3
"""Replace task268's dtype-only bool CastLike anchor with standard Cast."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task268.onnx"
OUTPUT = HERE / "task268_cast_bool.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph
    node = next(node for node in graph.node if node.output == ["_cond30_bool"])
    assert node.op_type == "CastLike" and list(node.input) == ["_csp_9_30", "_bool_like"]
    node.op_type = "Cast"
    del node.input[:]
    node.input.append("_csp_9_30")
    del node.attribute[:]
    node.attribute.append(helper.make_attribute("to", TensorProto.BOOL))
    kept = [item for item in graph.initializer if item.name != "_bool_like"]
    assert len(kept) + 1 == len(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUTPUT)
    payload = {
        "candidate": str(OUTPUT.relative_to(HERE.parents[3])),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "removed_initializer": "_bool_like:bool[]",
        "parameter_reduction": 1,
        "nodes_changed": ["_cond30_bool: CastLike -> Cast(to=BOOL)"],
    }
    (HERE / "task268_cast_bool_build.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exact task145 initializer alias: derive vertical slopes by transposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "task145.onnx"
OUT = HERE / "task145_slope_transpose_alias.onnx"
EXPECTED = "35cf952052882ff0198d01b64b75e7d36b2ba054b758089c6b54310559544d19"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    source = SOURCE.read_bytes()
    actual = sha(source)
    if actual != EXPECTED:
        raise RuntimeError(f"authority task145 changed: {actual}")
    model = onnx.load_from_string(source)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if not np.array_equal(
        arrays["slope_up"], np.transpose(arrays["slope_left"], (0, 1, 3, 2))
    ):
        raise RuntimeError("up slope is not the transpose of left slope")
    if not np.array_equal(
        arrays["slope_down"], np.transpose(arrays["slope_right"], (0, 1, 3, 2))
    ):
        raise RuntimeError("down slope is not the transpose of right slope")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "slope_up":
                node.input[index] = "slope_up_alias"
            elif name == "slope_down":
                node.input[index] = "slope_down_alias"
    alias_nodes = [
        helper.make_node(
            "Transpose", ["slope_left"], ["slope_up_alias"],
            perm=[0, 1, 3, 2], name="derive_slope_up",
        ),
        helper.make_node(
            "Transpose", ["slope_right"], ["slope_down_alias"],
            perm=[0, 1, 3, 2], name="derive_slope_down",
        ),
    ]
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(alias_nodes + original_nodes)
    keep = [
        item for item in model.graph.initializer
        if item.name not in {"slope_up", "slope_down"}
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUT)
    result = {
        "authority_sha256": actual,
        "candidate": str(OUT),
        "candidate_sha256": sha(OUT.read_bytes()),
        "removed_initializers": ["slope_up", "slope_down"],
        "removed_parameter_elements": 40,
        "added_nodes": ["derive_slope_up", "derive_slope_down"],
        "all_input_equivalence": "typed initializer transpose identity",
    }
    (HERE / "build.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Trim task297's all-zero second Conv kernel column and preserve width by crop pads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "baseline" / "task297.onnx"
OUTPUT = HERE / "task297_trim_zero_kernel.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    init = next(item for item in model.graph.initializer if item.name == "conv_w")
    weights = numpy_helper.to_array(init)
    if weights.shape != (1, 10, 1, 2) or np.count_nonzero(weights[..., 1]) != 0:
        raise RuntimeError("unexpected task297 baseline kernel")
    init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(weights[..., :1]), "conv_w"))

    node = next(item for item in model.graph.node if item.output[0] == "color_f")
    del node.attribute[:]
    node.attribute.extend(
        [
            helper.make_attribute("dilations", [1, 1]),
            helper.make_attribute("strides", [30, 1]),
            helper.make_attribute("pads", [0, 0, 0, -24]),
        ]
    )

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUTPUT)
    payload = {
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "edit": "conv_w (1,10,1,2) -> (1,10,1,1); pads [0,0,0,-24]",
        "parameter_reduction": 10,
    }
    (HERE / "task297_trim_zero_kernel_build.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

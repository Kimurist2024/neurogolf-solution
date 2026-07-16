#!/usr/bin/env python3
"""Build exact/local candidates for A28 without touching shared artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


ROOT = Path(__file__).resolve().parent


def task250_conv_integer() -> None:
    source = ROOT / "baseline" / "task250.onnx"
    target = ROOT / "candidates" / "task250_conv_integer.onnx"
    model = onnx.load(source)

    initializers = {item.name: item for item in model.graph.initializer}
    weights = numpy_helper.to_array(initializers["W"]).astype(np.int16)
    effective = (weights - 128).astype(np.int8)
    replacement = numpy_helper.from_array(effective, name="W_i8")

    kept = [item for item in model.graph.initializer if item.name not in {"W", "yscale"}]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.append(replacement)

    node = model.graph.node[-1]
    assert node.op_type == "QLinearConv" and node.output == ["output"]
    node.op_type = "ConvInteger"
    del node.input[:]
    node.input.extend(["feature10", "W_i8", "xzp"])
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT32

    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(model, target)


if __name__ == "__main__":
    task250_conv_integer()

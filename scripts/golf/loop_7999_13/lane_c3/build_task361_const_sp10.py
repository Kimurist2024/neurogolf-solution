#!/usr/bin/env python3
"""Fold task361's input-independent [10, 10] Concat into an initializer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
source = HERE / "base" / "task361.onnx"
target = HERE / "candidates" / "task361_const_sp10.onnx"

model = onnx.load(source)
matches = [node for node in model.graph.node if "sp10_rt" in node.output]
if len(matches) != 1 or matches[0].op_type != "Concat":
    raise RuntimeError(f"unexpected sp10 producer: {matches}")
producer = matches[0]
if list(producer.input) != ["ten_i64", "ten_i64"]:
    raise RuntimeError(f"unexpected sp10 inputs: {list(producer.input)}")

model.graph.node.remove(producer)
model.graph.initializer.append(
    numpy_helper.from_array(np.asarray([10, 10], dtype=np.int64), name="sp10_rt")
)
onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
target.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, target)
print(target)

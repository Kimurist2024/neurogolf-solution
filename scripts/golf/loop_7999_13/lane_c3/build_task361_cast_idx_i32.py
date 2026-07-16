#!/usr/bin/env python3
"""Replace task361's type-only CastLike reference with a typed Cast."""

from __future__ import annotations

from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
source = HERE / "base" / "task361.onnx"
target = HERE / "candidates" / "task361_cast_idx_i32.onnx"

model = onnx.load(source)
matches = [node for node in model.graph.node if "idx_i32" in node.output]
if len(matches) != 1 or matches[0].op_type != "CastLike":
    raise RuntimeError(f"unexpected idx_i32 producer: {matches}")
node = matches[0]
if list(node.input) != ["idx_src", "idx_ref"]:
    raise RuntimeError(f"unexpected CastLike inputs: {list(node.input)}")

node.op_type = "Cast"
del node.input[:]
node.input.append("idx_src")
node.attribute.append(onnx.helper.make_attribute("to", onnx.TensorProto.INT32))

kept = [initializer for initializer in model.graph.initializer if initializer.name != "idx_ref"]
if len(kept) + 1 != len(model.graph.initializer):
    raise RuntimeError("idx_ref initializer missing or duplicated")
del model.graph.initializer[:]
model.graph.initializer.extend(kept)

onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
target.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, target)
print(target)

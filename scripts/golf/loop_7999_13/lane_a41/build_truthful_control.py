#!/usr/bin/env python3
"""Replace task366's false intermediate declarations with observed real shapes."""

from __future__ import annotations

import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
model = onnx.load(HERE / "baseline_task366.onnx")
trace = json.loads((HERE / "runtime_shape_trace.json").read_text())
actual = {item["tensor"]: item["actual"] for item in trace["mismatches"]}

values = {item.name: item for item in list(model.graph.value_info) + list(model.graph.output)}
for name, shape in actual.items():
    value = values[name]
    del value.type.tensor_type.shape.dim[:]
    for size in shape:
        value.type.tensor_type.shape.dim.add().dim_value = int(size)

onnx.checker.check_model(model, full_check=True)
inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)

# Strict inference must not reintroduce any of the false declarations.
inferred_values = {
    item.name: [dim.dim_value for dim in item.type.tensor_type.shape.dim]
    for item in list(inferred.graph.value_info) + list(inferred.graph.output)
}
wrong = {name: {"wanted": shape, "inferred": inferred_values.get(name)} for name, shape in actual.items() if inferred_values.get(name) != shape}
if wrong:
    raise RuntimeError(f"strict inference did not preserve truthful declarations: {wrong}")

out = HERE / "truthful_annotation_control.onnx"
onnx.save(inferred, out)
(HERE / "truthful_control_manifest.json").write_text(
    json.dumps({"path": out.name, "repaired_shapes": len(actual), "strict_inference_wrong": wrong}, indent=2) + "\n"
)
print(json.dumps({"path": out.name, "repaired_shapes": len(actual)}))

#!/usr/bin/env python3
"""Experimental exact-domain task364: use 1-byte float8 Boolean encodings."""

from pathlib import Path
import hashlib
import json

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
source = (HERE / "task364.onnx").read_bytes()
model = onnx.load_from_string(source)
float8 = TensorProto.FLOAT8E4M3FN
for item in model.graph.initializer:
    if item.name == "zero_f16":
        item.data_type = float8
        item.raw_data = bytes([0])
for value in model.graph.value_info:
    if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
        value.type.tensor_type.elem_type = float8
out = HERE / "task364_float8.onnx"
result = {
    "authority_sha256": hashlib.sha256(source).hexdigest(),
    "float8_type": helper.tensor_dtype_to_string(float8),
}
try:
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result["checker"] = "pass"
    onnx.save(model, out)
    result["candidate_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
except Exception as exc:
    result["checker"] = f"{type(exc).__name__}: {exc}"
(HERE / "float8_build.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

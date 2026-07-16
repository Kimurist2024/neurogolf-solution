#!/usr/bin/env python3
"""Remove task324's redundant [0,2] vector by exact in-Einsum reuse.

The final Einsum contains ``onehot_values_outscale=[0,2]`` twice.  Replacing
both by the already-live ``onehot_values=[0,1]`` divides the contraction by
four.  An additional fully-contracted occurrence of the already-live
``signpow=[[1,1,1],[1,-1,1]]`` contributes its exact element sum, four, and
therefore restores the same mathematical contraction without a new tensor or
node.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8023.08.zip"
AUTHORITY_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
SOURCE_SHA256 = "01d83616d8c4ca19ac99b4fb4efc130d1be25023f386a7bf332cea773b7ace84"
OUTPUT = HERE / "candidates" / "task324_outscale_reuse.onnx"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def params(model: onnx.ModelProto) -> int:
    return int(sum(math.prod(item.dims) for item in model.graph.initializer))


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA drift")
    with zipfile.ZipFile(AUTHORITY) as archive:
        source = archive.read("task324.onnx")
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError("task324 member drift")
    model = onnx.load_model_from_string(source)
    arrays = {
        item.name: numpy_helper.to_array(item)
        for item in model.graph.initializer
    }
    if arrays["onehot_values"].tolist() != [0.0, 1.0]:
        raise RuntimeError("onehot_values drift")
    if arrays["onehot_values_outscale"].tolist() != [0.0, 2.0]:
        raise RuntimeError("outscale drift")
    if float(arrays["signpow"].sum()) != 4.0:
        raise RuntimeError("signpow scalar proof drift")

    node = model.graph.node[-1]
    if node.op_type != "Einsum":
        raise RuntimeError("final node is not Einsum")
    attr = next(item for item in node.attribute if item.name == "equation")
    equation = attr.s.decode("ascii")
    lhs, rhs = equation.split("->")
    terms = lhs.split(",")
    if len(terms) != len(node.input):
        raise RuntimeError("equation/input arity mismatch")
    replacements = 0
    inputs = list(node.input)
    for index, name in enumerate(inputs):
        if name == "onehot_values_outscale":
            inputs[index] = "onehot_values"
            replacements += 1
    if replacements != 2:
        raise RuntimeError(f"expected two outscale uses, saw {replacements}")
    # d and i are unused in the authority equation.  They disappear by
    # reduction, multiplying the contraction by sum(signpow) == 4.
    inputs.append("signpow")
    terms.append("di")
    del node.input[:]
    node.input.extend(inputs)
    attr.s = (",".join(terms) + "->" + rhs).encode("ascii")
    node.doc_string = "exact outscale reuse; extra signpow contraction sums to four"

    kept = [
        item for item in model.graph.initializer
        if item.name != "onehot_values_outscale"
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.producer_name = "codex-task324-exact-outscale-reuse"
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blob = model.SerializeToString()
    OUTPUT.write_bytes(blob)
    result = {
        "task": 324,
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "source_sha256": SOURCE_SHA256,
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "candidate_sha256": sha256(blob),
        "source_params": 198,
        "candidate_params": params(model),
        "param_saving": 198 - params(model),
        "proof": {
            "outscale_relation": "[0,2] = 2 * [0,1]",
            "outscale_occurrences": replacements,
            "signpow_sum": float(arrays["signpow"].sum()),
            "net_factor": "(1/2)^2 * 4 = 1",
        },
        "full_check": True,
        "strict_data_prop": True,
    }
    (HERE / "task324_outscale_reuse_build.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

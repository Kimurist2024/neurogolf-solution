#!/usr/bin/env python3
"""Static and one-input runtime-shape audit of the sole >0.05 lead."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "scripts/golf/loop_7999_13/lane_a23/candidates/task205_r02.onnx"
SHARED = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402


def load_shared():
    spec = importlib.util.spec_from_file_location("high_gain_shape_audit", SHARED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    model = onnx.load(MODEL)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    non_positive: list[dict[str, object]] = []
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                non_positive.append({"tensor": value.name, "dimension": str(dim)})
    shared = load_shared()
    shape_trace = shared.runtime_shape_trace(205, model)
    payload = {
        "model": str(MODEL.relative_to(ROOT)),
        "full_check": True,
        "strict_shape_data_prop": True,
        "non_positive_or_symbolic_dimensions": non_positive,
        "runtime_shape_trace": shape_trace,
        "conv_bias_findings": check_conv_bias(model),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
    }
    out = HERE / "evidence/task205_static_shape.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

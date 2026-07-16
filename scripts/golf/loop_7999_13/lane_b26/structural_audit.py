#!/usr/bin/env python3
"""Strict structural, runtime-shape, cost, and Conv-UB audit for B26."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ITEMS = [
    (328, "baseline", HERE / "baseline_task328.onnx", 58),
    (328, "candidate", HERE / "task328_reuse_j_diagonal.onnx", 58),
    (358, "baseline", HERE / "baseline_task358.onnx", 44),
    (358, "candidate", HERE / "task358_combine_r2_r3.onnx", 42),
]


def load_trace_helper():
    path = ROOT / "scripts/golf/loop_7999_13/lane_b15/audit_candidates.py"
    spec = importlib.util.spec_from_file_location("b26_trace_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?" for dim in value.type.tensor_type.shape.dim]


def audit(task: int, role: str, path: Path, expected_max_einsum: int, tracer) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    static_positive = all(
        value.type.HasField("tensor_type")
        and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)
        for value in values
    )
    runtime = tracer.trace_runtime_shapes(copy.deepcopy(model), task)
    memory, params, cost = (int(value) for value in cost_of(str(path)))
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    if max_einsum != expected_max_einsum:
        raise RuntimeError((task, role, max_einsum, expected_max_einsum))
    row = {
        "task": task,
        "role": role,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "serialized_bytes": path.stat().st_size,
        "checker_full": True,
        "strict_shape_inference": True,
        "static_positive_shapes": static_positive,
        "input_shapes": [shape(value) for value in model.graph.input],
        "output_shapes": [shape(value) for value in model.graph.output],
        "runtime_shape": runtime,
        "shape_truthful": runtime.get("shape_cloak") is False,
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_functions_sparse_nested": not model.functions
        and not model.graph.sparse_initializer
        and all(attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for node in model.graph.node for attr in node.attribute),
        "no_banned_ops": all(node.op_type.upper() not in BANNED and "SEQUENCE" not in node.op_type.upper() for node in model.graph.node),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
        ),
        "conv_bias_findings": check_conv_bias(model),
        "max_einsum_inputs": max_einsum,
        "memory": memory,
        "params": params,
        "cost": cost,
    }
    row["pass"] = bool(
        row["static_positive_shapes"]
        and row["shape_truthful"]
        and row["standard_domains"]
        and row["no_functions_sparse_nested"]
        and row["no_banned_ops"]
        and row["finite_initializers"]
        and not row["conv_bias_findings"]
    )
    return row


def main() -> int:
    tracer = load_trace_helper()
    rows = [audit(*item, tracer) for item in ITEMS]
    if not all(row["pass"] for row in rows):
        raise RuntimeError("structural gate failed")
    payload = {"rows": rows}
    (HERE / "structural_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in ("task", "role", "sha256", "shape_truthful", "conv_bias_findings", "max_einsum_inputs", "cost", "pass")} for row in rows], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

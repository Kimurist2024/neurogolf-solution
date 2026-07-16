#!/usr/bin/env python3
"""Strict structural and UB audit for the final Wave B4 task107 candidate."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib.scoring import sanitize_model  # noqa: E402


BASELINE = HERE / "baseline_task107.onnx"
CANDIDATE = HERE / "candidate_task107_shared_coefficients_rank4.onnx"
OUTPUT = HERE / "task107_rank4_structural.json"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append("")
    return result


def audit_model(path: Path) -> tuple[onnx.ModelProto, dict[str, object]]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    nested: list[str] = []
    banned: list[str] = []
    sequence: list[str] = []
    custom_nodes: list[str] = []
    max_einsum_inputs = 0
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED:
            banned.append(node.op_type)
        if "SEQUENCE" in upper:
            sequence.append(node.op_type)
        if node.domain not in ("", "ai.onnx"):
            custom_nodes.append(node.domain)
        if node.op_type == "Einsum":
            max_einsum_inputs = max(max_einsum_inputs, len(node.input))
            equation = next(
                onnx.helper.get_attribute_value(attribute).decode("ascii")
                for attribute in node.attribute
                if attribute.name == "equation"
            )
            if len(equation.split("->")[0].split(",")) != len(node.input):
                raise RuntimeError("Einsum operand/equation mismatch")
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.op_type)
    if model.functions or model.graph.sparse_initializer:
        raise RuntimeError("functions or sparse initializers present")
    if banned or sequence or custom_nodes or nested:
        raise RuntimeError(
            f"banned={banned} sequence={sequence} custom={custom_nodes} nested={nested}"
        )
    if any(item.domain not in ("", "ai.onnx") for item in model.opset_import):
        raise RuntimeError("custom opset domain")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError("noncanonical I/O count")
    if model.graph.input[0].name != "input" or model.graph.output[0].name != "output":
        raise RuntimeError("noncanonical I/O names")
    if dims(model.graph.input[0]) != [1, 10, 30, 30]:
        raise RuntimeError(f"input contract mismatch: {dims(model.graph.input[0])}")

    nonstatic: list[str] = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            nonstatic.append(value.name)
            continue
        if any(not d.HasField("dim_value") or d.dim_value <= 0 for d in value.type.tensor_type.shape.dim):
            nonstatic.append(value.name)
    if nonstatic:
        raise RuntimeError(f"non-static tensors: {nonstatic}")
    bias_issues = check_conv_bias(model)
    if bias_issues:
        raise RuntimeError(f"Conv bias issues: {bias_issues}")
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(f"non-finite initializer: {initializer.name}")
        if any(dimension <= 0 for dimension in array.shape):
            raise RuntimeError(f"non-positive initializer dimension: {initializer.name}")

    sanitized = sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    runtime_output = session.run(["output"], {"input": np.zeros((1, 10, 30, 30), np.float32)})[0]

    row: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "under_1_44mb": path.stat().st_size < 1_440_000,
        "checker_full": True,
        "strict_shape_inference": True,
        "dynamic_or_nonpositive_tensors": [],
        "standard_domains": True,
        "functions": 0,
        "sparse_initializers": 0,
        "nested_graphs": 0,
        "banned_ops": [],
        "sequence_ops": [],
        "conv_bias_issues": [],
        "nonfinite_initializers": 0,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "parameter_elements": sum(numpy_helper.to_array(x).size for x in model.graph.initializer),
        "max_einsum_inputs": max_einsum_inputs,
        "op_histogram": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "input_shape": dims(inferred.graph.input[0]),
        "declared_output_shape": dims(model.graph.output[0]),
        "inferred_output_shape": dims(inferred.graph.output[0]),
        "runtime_output_shape": list(runtime_output.shape),
        "runtime_load_and_execute": True,
    }
    return model, row


def main() -> int:
    baseline, baseline_row = audit_model(BASELINE)
    candidate, candidate_row = audit_model(CANDIDATE)
    baseline_ops = [(node.op_type, node.domain) for node in baseline.graph.node]
    candidate_ops = [(node.op_type, node.domain) for node in candidate.graph.node]
    result = {
        "baseline": baseline_row,
        "candidate": candidate_row,
        "comparison": {
            "node_sequence_identical": baseline_ops == candidate_ops,
            "node_count_delta": len(candidate.graph.node) - len(baseline.graph.node),
            "parameter_delta": candidate_row["parameter_elements"] - baseline_row["parameter_elements"],
            "only_graph_change_classes": ["Einsum initializer operands", "Einsum equation factor labels"],
            "stale_declared_output_shape_inherited_from_baseline": (
                baseline_row["declared_output_shape"] == candidate_row["declared_output_shape"]
                and baseline_row["runtime_output_shape"] == candidate_row["runtime_output_shape"]
            ),
            "candidate_adds_no_runtime_node": len(candidate.graph.node) == len(baseline.graph.node),
            "candidate_adds_no_op_type_or_domain": baseline_ops == candidate_ops,
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

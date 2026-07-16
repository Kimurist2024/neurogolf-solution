#!/usr/bin/env python3
"""Strict structural gate for retained sound controls in the A2 lane."""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402

BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
MODELS = {
    9: HERE / "sound_controls" / "task009.onnx",
    77: HERE / "sound_controls" / "task077.onnx",
    173: HERE / "sound_controls" / "task173.onnx",
}


def audit(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
    }
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    if model.functions:
        raise RuntimeError(f"{path}: local functions")
    if model.graph.sparse_initializer:
        raise RuntimeError(f"{path}: sparse initializers")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RuntimeError(f"{path}: noncanonical I/O count")
    if any(item.domain not in ("", "ai.onnx") for item in model.opset_import):
        raise RuntimeError(f"{path}: custom opset domain")
    nested: list[str] = []
    banned: list[str] = []
    custom_nodes: list[str] = []
    sequence_nodes: list[str] = []
    max_einsum_inputs = 0
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            custom_nodes.append(node.domain)
        upper = node.op_type.upper()
        if upper in BANNED:
            banned.append(node.op_type)
        if "SEQUENCE" in upper:
            sequence_nodes.append(node.op_type)
        if node.op_type == "Einsum":
            max_einsum_inputs = max(max_einsum_inputs, len(node.input))
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.op_type)
    if banned or sequence_nodes or nested or custom_nodes:
        raise RuntimeError(
            f"{path}: banned={banned}, sequence={sequence_nodes}, "
            f"nested={nested}, custom={custom_nodes}"
        )

    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    non_static: list[str] = []
    for value in values:
        if not value.type.HasField("tensor_type"):
            non_static.append(value.name)
            continue
        for dimension in value.type.tensor_type.shape.dim:
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                non_static.append(value.name)
                break
    if non_static:
        raise RuntimeError(f"{path}: non-static tensors {non_static}")
    bias_issues = check_conv_bias(model)
    if bias_issues:
        raise RuntimeError(f"{path}: Conv bias issues {bias_issues}")

    row.update(
        valid=True,
        checker_full=True,
        strict_static_positive=True,
        standard_domains=True,
        functions=0,
        sparse_initializers=0,
        nested_graphs=0,
        banned_ops=[],
        sequence_ops=[],
        conv_bias_issues=[],
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        max_einsum_inputs=max_einsum_inputs,
        op_histogram=dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        input_shape=[dimension.dim_value for dimension in inferred.graph.input[0].type.tensor_type.shape.dim],
        output_shape=[dimension.dim_value for dimension in inferred.graph.output[0].type.tensor_type.shape.dim],
    )
    return row


def main() -> None:
    result = {str(task): audit(path) for task, path in MODELS.items()}
    (HERE / "structural_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

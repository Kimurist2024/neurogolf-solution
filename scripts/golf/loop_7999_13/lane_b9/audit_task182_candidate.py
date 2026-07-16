#!/usr/bin/env python3
"""Emit the structural and official-cost audit for the B9 task182 lead."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def dimensions(value: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def main() -> None:
    path = HERE / "task182_reuse_constants.onnx"
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    with tempfile.TemporaryDirectory() as directory:
        scored = scoring.score_and_verify(
            model, 182, directory, label="b9", require_correct=True
        )
    if scored is None:
        raise RuntimeError("official score_and_verify returned None")

    producers = {
        output: node
        for node in inferred.graph.node
        for output in node.output
        if output
    }
    value_shapes = {
        value.name: dimensions(value)
        for value in list(inferred.graph.value_info)
        + list(inferred.graph.input)
        + list(inferred.graph.output)
    }
    conv_biases: list[dict[str, object]] = []
    for node in inferred.graph.node:
        bias_index = 2 if node.op_type == "Conv" else 8 if node.op_type == "QLinearConv" else -1
        if bias_index < 0 or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        weight_index = 1 if node.op_type == "Conv" else 3
        weight_name = node.input[weight_index]
        weight_initializer = next(
            (item for item in inferred.graph.initializer if item.name == weight_name),
            None,
        )
        weight_shape = (
            list(weight_initializer.dims)
            if weight_initializer is not None
            else value_shapes.get(weight_name)
        )
        if not weight_shape or not isinstance(weight_shape[0], int):
            raise RuntimeError(f"unknown output-channel dimension for {weight_name}")
        output_channels = int(weight_shape[0])
        bias_name = node.input[bias_index]
        conv_biases.append(
            {
                "op_type": node.op_type,
                "output": list(node.output),
                "weight_output_channels": output_channels,
                "bias": bias_name,
                "bias_shape": value_shapes.get(bias_name),
                "bias_is_initializer": bias_name not in producers,
                "safe": value_shapes.get(bias_name) in ([output_channels], []),
            }
        )

    report = {
        "task": 182,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checker_full": True,
        "strict_shape_inference": True,
        "domains": sorted({item.domain for item in model.opset_import}),
        "opsets": {item.domain: int(item.version) for item in model.opset_import},
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graph_attributes": sum(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node
            for attribute in node.attribute
        ),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "initializer_elements": sum(
            __import__("math").prod(item.dims) for item in model.graph.initializer
        ),
        "official_score": {
            "memory": int(scored["memory"]),
            "params": int(scored["params"]),
            "cost": int(scored["cost"]),
            "correct": bool(scored["correct"]),
        },
        "conv_biases": conv_biases,
        "standard_domain_only": all(item.domain in ("", "ai.onnx") for item in model.opset_import),
    }
    (HERE / "task182_candidate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

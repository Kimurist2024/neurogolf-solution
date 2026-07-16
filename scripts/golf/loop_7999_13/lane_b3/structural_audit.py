#!/usr/bin/env python3
"""Audit checker/static/banned/nested/function/sequence/sparse/bias gates."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import onnx


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return []
    result = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            result.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def audit(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    record: dict[str, object] = {"path": str(path), "checker": False, "strict_shape_data_prop": False}
    try:
        onnx.checker.check_model(model, full_check=True)
        record["checker"] = True
    except Exception as exc:
        record["checker_error"] = f"{type(exc).__name__}: {exc}"
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
    except Exception as exc:
        inferred = model
        record["strict_shape_error"] = f"{type(exc).__name__}: {exc}"
    all_values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    bad_shapes = {value.name: dims(value) for value in all_values if any(not isinstance(d, int) or d <= 0 for d in dims(value))}
    ops = [node.op_type for node in model.graph.node]
    nested = []
    conv_bias = []
    initializers = {init.name: init for init in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append({"node": index, "op": node.op_type, "attribute": attr.name})
        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 3 and node.input[2]:
            weight = initializers.get(node.input[1])
            bias = initializers.get(node.input[2])
            if weight is not None and bias is not None:
                expected = weight.dims[0] if node.op_type == "Conv" else weight.dims[1]
                if len(bias.dims) != 1 or bias.dims[0] != expected:
                    conv_bias.append({"node": index, "expected": expected, "bias_dims": list(bias.dims)})
    record.update(
        {
            "ops": dict(Counter(ops)),
            "bad_static_shapes": bad_shapes,
            "banned_ops": sorted({op for op in ops if op.upper() in BANNED}),
            "sequence_ops": sorted({op for op in ops if "Sequence" in op}),
            "nested_graph_attributes": nested,
            "function_count": len(model.functions),
            "sparse_initializer_count": len(model.graph.sparse_initializer),
            "conv_bias_findings": conv_bias,
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--identity-pair", type=Path, nargs=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"models": [audit(path) for path in args.model]}
    if args.identity_pair:
        left = onnx.load(args.identity_pair[0])
        right = onnx.load(args.identity_pair[1])
        left.graph.ClearField("value_info")
        right.graph.ClearField("value_info")
        report["executable_graph_identical_after_clearing_value_info"] = (
            left.SerializeToString(deterministic=True) == right.SerializeToString(deterministic=True)
        )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

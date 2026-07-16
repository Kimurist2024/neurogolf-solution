#!/usr/bin/env python3
"""Structural, domain, banned-op, bias, and explicit shape-cloak audit."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
WITNESSES = {
    330: [("g", [1, 1, 1, 1], [1, 10, 30, 30], "GroupNormalization preserves X shape")],
    354: [
        ("gn_f", [1, 1, 1, 1], [1, 10, 30, 30], "GroupNormalization preserves X shape"),
        ("data_clear_f", [1, 1, 1, 1], [1, 10, 30, 30], "GroupNormalization preserves X shape"),
    ],
    361: [("gn", [1, 1, 1, 1], [1, 10, 30, 30], "GroupNormalization preserves X shape")],
    364: [
        (
            "input_fake",
            [1, 1, 1, 1],
            [1, 10, 30, 30],
            "CenterCropPad target is Shape(input), so output preserves input shape",
        )
    ],
    368: [("gn", [1, 1, 1, 1], [1, 10, 30, 30], "GroupNormalization preserves X shape")],
}


def conv_bias_findings(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    findings: list[dict[str, object]] = []
    for node in model.graph.node:
        bias_index = None
        weight_index = None
        if node.op_type == "Conv":
            weight_index, bias_index = 1, 2
        elif node.op_type == "ConvTranspose":
            weight_index, bias_index = 1, 2
        elif node.op_type == "QLinearConv":
            weight_index, bias_index = 3, 8
        if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        weight = initializers.get(node.input[weight_index])
        bias = initializers.get(node.input[bias_index])
        expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        count = int(math.prod(bias.dims)) if bias is not None and bias.dims else None
        findings.append(
            {
                "node": node.name or node.output[0],
                "op": node.op_type,
                "bias": node.input[bias_index],
                "bias_shape": list(bias.dims) if bias is not None else None,
                "expected_channels": expected,
                "safe": bias is not None and len(bias.dims) == 1 and count == expected,
            }
        )
    return findings


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def inspect(path: Path, task: int) -> dict[str, object]:
    model = onnx.load(path)
    record: dict[str, object] = {
        "path": str(path),
        "task": task,
        "nodes": len(model.graph.node),
        "params": sum(math.prod(item.dims) if item.dims else 1 for item in model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "nonstandard_domains": [item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}],
        "banned_ops": [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED or "Sequence" in node.op_type],
        "nested_graph_attributes": sum(
            1
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "conv_bias_findings": conv_bias_findings(model),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        record["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(full_check=False, full_check_error=repr(exc))
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(strict_shape_data_prop=False, strict_shape_error=repr(exc))
    declared = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    record["shape_cloak_witnesses"] = [
        {
            "tensor": name,
            "declared": declared.get(name, expected_declared),
            "guaranteed_runtime": runtime,
            "reason": reason,
        }
        for name, expected_declared, runtime, reason in WITNESSES.get(task, [])
    ]
    record["shape_cloak_free"] = not record["shape_cloak_witnesses"]
    record["conv_bias_safe"] = all(item["safe"] for item in record["conv_bias_findings"])
    return record


def main() -> None:
    records = {
        f"base_{task}": inspect(HERE / "base" / f"task{task:03d}.onnx", task)
        for task in (310, 330, 340, 354, 361, 364, 368)
    }
    records["task310_safe"] = inspect(HERE / "task310_safe_linear_selector.onnx", 310)
    (HERE / "structural_audit.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    for key, record in records.items():
        print(
            key,
            "check",
            record["full_check"],
            "strict",
            record["strict_shape_data_prop"],
            "cloak_free",
            record["shape_cloak_free"],
            "bias_safe",
            record["conv_bias_safe"],
        )


if __name__ == "__main__":
    main()

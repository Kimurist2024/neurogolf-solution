#!/usr/bin/env python3
"""Prove dynamic Conv bias length for every C35 task192 model."""

from __future__ import annotations

import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def audit(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    values = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    initializers = {item.name: item for item in model.graph.initializer}
    rows = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 3 or not node.input[2]:
            continue
        weight = initializers[node.input[1]]
        bias_value = values[node.input[2]]
        bias_shape = tensor_shape(bias_value)
        expected = int(weight.dims[0])
        rows.append(
            {
                "node": node.output[0],
                "weight_shape": list(weight.dims),
                "expected_output_channels": expected,
                "bias": node.input[2],
                "bias_is_initializer": node.input[2] in initializers,
                "strict_inferred_bias_shape": bias_shape,
                "safe_length": bias_shape == [expected],
            }
        )
    return {"path": str(path), "conv_biases": rows, "all_safe": all(r["safe_length"] for r in rows)}


def main() -> None:
    paths = [HERE / "baseline" / "task192.onnx"] + sorted((HERE / "candidates").glob("*.onnx"))
    report = [audit(path) for path in paths]
    (HERE / "conv_bias_contract.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

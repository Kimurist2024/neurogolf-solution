#!/usr/bin/env python3
"""Strict structural audit for C26 baselines and the only cheaper candidate."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def inspect(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    row: dict[str, object] = {
        "path": str(path),
        "nodes": len(model.graph.node),
        "params": sum(math.prod(item.dims) if item.dims else 1 for item in model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "nonstandard_domains": [x.domain for x in model.opset_import if x.domain not in {"", "ai.onnx"}],
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "Sequence" in node.op_type
        ],
        "nested_graph_attributes": sum(
            1
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "conv_family_nodes": [
            node.op_type for node in model.graph.node if node.op_type in {"Conv", "ConvTranspose", "QLinearConv"}
        ],
        "giant_einsum_input_counts": [
            len(node.input) for node in model.graph.node if node.op_type == "Einsum" and len(node.input) > 16
        ],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=repr(exc))
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row["strict_shape_data_prop"] = True
        row["inferred_tensors_static_positive"] = all(
            dim.HasField("dim_value") and dim.dim_value > 0
            for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
            for dim in value.type.tensor_type.shape.dim
        )
    except Exception as exc:  # noqa: BLE001
        row.update(strict_shape_data_prop=False, strict_shape_error=repr(exc))
    row["shape_cloak_free"] = len(model.graph.value_info) == 0
    return row


def main() -> None:
    records = {
        "base310": inspect(HERE / "base" / "task310.onnx"),
        "base328": inspect(HERE / "base" / "task328.onnx"),
        "candidate328": inspect(HERE / "task328_r01_reuse_ninv.onnx"),
    }
    (HERE / "structural_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

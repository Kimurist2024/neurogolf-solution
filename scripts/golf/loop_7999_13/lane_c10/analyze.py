#!/usr/bin/env python3
"""Exact-cost and structural inventory for the C10 baseline models."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (14, 36, 75, 159, 218, 225, 245)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def params(model: onnx.ModelProto) -> int:
    return sum(math.prod(item.dims) if item.dims else 1 for item in model.graph.initializer)


def conv_bias_findings(model: onnx.ModelProto) -> list[dict[str, object]]:
    inits = {item.name: item for item in model.graph.initializer}
    rows: list[dict[str, object]] = []
    for node in model.graph.node:
        indexes = {"Conv": (1, 2), "ConvTranspose": (1, 2), "QLinearConv": (3, 8)}
        if node.op_type not in indexes:
            continue
        weight_index, bias_index = indexes[node.op_type]
        if len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        weight = inits.get(node.input[weight_index])
        bias = inits.get(node.input[bias_index])
        expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        count = int(math.prod(bias.dims)) if bias is not None and bias.dims else None
        rows.append(
            {
                "node": node.name or node.output[0],
                "op": node.op_type,
                "weight_shape": list(weight.dims) if weight is not None else None,
                "bias_shape": list(bias.dims) if bias is not None else None,
                "expected_channels": expected,
                "safe": bias is not None and len(bias.dims) == 1 and count == expected,
            }
        )
    return rows


def initializer_row(item: onnx.TensorProto) -> dict[str, object]:
    array = numpy_helper.to_array(item)
    return {
        "name": item.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "count": int(array.size),
        "nbytes": int(array.nbytes),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
        "zero_count": int(np.count_nonzero(array == 0)),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def inspect(task: int) -> dict[str, object]:
    path = HERE / "base" / f"task{task:03d}.onnx"
    model = onnx.load(path)
    producers = {
        output: {"op": node.op_type, "inputs": list(node.input), "node": node.name}
        for node in model.graph.node
        for output in node.output
        if output
    }
    declared = {
        value.name: {"shape": shape(value), "dtype": value.type.tensor_type.elem_type}
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    record: dict[str, object] = {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes": len(model.graph.node),
        "params": params(model),
        "initializers": [initializer_row(item) for item in model.graph.initializer],
        "value_info": declared,
        "inputs": {value.name: shape(value) for value in model.graph.input},
        "outputs": {value.name: shape(value) for value in model.graph.output},
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "nodes_detail": [
            {"name": node.name, "op": node.op_type, "inputs": list(node.input), "outputs": list(node.output)}
            for node in model.graph.node
        ],
        "producers": producers,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
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
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
        record["inferred_value_info"] = {
            value.name: {"shape": shape(value), "dtype": value.type.tensor_type.elem_type}
            for value in list(inferred.graph.value_info) + list(inferred.graph.output)
        }
    except Exception as exc:  # noqa: BLE001
        record.update(strict_shape_data_prop=False, strict_shape_error=repr(exc))
    with tempfile.TemporaryDirectory(prefix=f"c10_{task}_", dir="/tmp") as workdir:
        record["official_like_score"] = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, f"c10_{task}", require_correct=False
        )
    return record


def main() -> None:
    records = {str(task): inspect(task) for task in TASKS}
    (HERE / "baseline_anatomy.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    for task, row in records.items():
        score = row["official_like_score"]
        print(
            task,
            "cost", None if score is None else score["cost"],
            "correct", None if score is None else score["correct"],
            "nodes", row["nodes"],
            "params", row["params"],
            "check", row["full_check"],
            "strict", row["strict_shape_data_prop"],
            "bias_safe", all(item["safe"] for item in row["conv_bias_findings"]),
        )


if __name__ == "__main__":
    main()

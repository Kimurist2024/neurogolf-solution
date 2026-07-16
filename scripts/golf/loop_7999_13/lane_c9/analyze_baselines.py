#!/usr/bin/env python3
"""Score and structurally inspect the exact C9 ZIP members."""

from __future__ import annotations

import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (310, 330, 340, 354, 361, 364, 368)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def declared_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        else:
            result.append(dim.dim_param or "?")
    return result


def static_floor(model: onnx.ModelProto) -> tuple[int | None, list[dict[str, object]]]:
    try:
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        assert sanitized is not None
        inferred = onnx.shape_inference.infer_shapes(
            sanitized, strict_mode=True, data_prop=True
        )
    except Exception:
        return None, []
    rows: list[dict[str, object]] = []
    total = 0
    for value in inferred.graph.value_info:
        shape = declared_shape(value)
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            return None, rows
        dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        nbytes = int(math.prod(shape) * np.dtype(dtype).itemsize)
        total += nbytes
        rows.append({"name": value.name, "shape": shape, "bytes": nbytes})
    return total, rows


def analyze(task: int) -> dict[str, object]:
    path = HERE / "base" / f"task{task:03d}.onnx"
    model = onnx.load(path)
    row: dict[str, object] = {
        "task": task,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info_count": len(model.graph.value_info),
        "input_shapes": [declared_shape(value) for value in model.graph.input],
        "output_shapes": [declared_shape(value) for value in model.graph.output],
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "params": scoring.calculate_params(model),
        "banned_ops": [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED or "Sequence" in node.op_type],
        "nested_graph_attributes": sum(
            1
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "nonstandard_domains": [item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["full_check"] = False
        row["full_check_error"] = repr(exc)
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["strict_shape_data_prop"] = False
        row["strict_shape_error"] = repr(exc)
    floor, values = static_floor(model)
    row["declared_static_memory_floor"] = floor
    row["declared_intermediates"] = values
    score = scoring.score_and_verify(
        model, task, str(HERE / "score_work"), f"base{task}", require_correct=False
    )
    row["official_like_score"] = score
    return row


def main() -> None:
    result = {str(task): analyze(task) for task in TASKS}
    (HERE / "baseline_anatomy.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for task, row in result.items():
        print(task, row["official_like_score"], row["full_check"], row["strict_shape_data_prop"])


if __name__ == "__main__":
    main()

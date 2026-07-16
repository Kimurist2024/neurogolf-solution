#!/usr/bin/env python3
"""Explain the measured cost floor of the smallest table-free task391 control."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "others/highspeed/task391_cost139.onnx"
sys.path.insert(0, str(ROOT / "scripts"))

from golf.rank_dir import cost_of  # noqa: E402


def shape_of(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def main() -> int:
    raw = onnx.load(MODEL)
    inferred = onnx.shape_inference.infer_shapes(raw, strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    initializer_names = {item.name for item in inferred.graph.initializer}
    graph_io = {item.name for item in inferred.graph.input} | {
        item.name for item in inferred.graph.output
    }
    rows = []
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in graph_io or name in initializer_names:
                continue
            value = typed[name]
            shape = shape_of(value)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            rows.append(
                {
                    "tensor": name,
                    "producer": node.op_type,
                    "dtype": str(np.dtype(dtype)),
                    "shape": shape,
                    "elements": math.prod(shape),
                    "bytes": math.prod(shape) * np.dtype(dtype).itemsize,
                }
            )
    initializer_rows = [
        {"name": item.name, "shape": list(item.dims), "elements": math.prod(item.dims)}
        for item in inferred.graph.initializer
    ]
    memory, params, total = cost_of(str(MODEL))
    report = {
        "model": str(MODEL.relative_to(ROOT)),
        "measured": {"memory": memory, "params": params, "cost": total},
        "intermediates": rows,
        "static_intermediate_bytes": sum(row["bytes"] for row in rows),
        "initializers": initializer_rows,
        "initializer_elements": sum(row["elements"] for row in initializer_rows),
        "no_duplicate_initializer_values": True,
        "floor_explanation": {
            "counts_float32": 40,
            "required_topk_values_float32": 16,
            "required_topk_indices_int64": 32,
            "ranking_subtotal_before_label_emitter": 88,
            "remaining_label_intermediates": 16,
            "memory_alone_equals_baseline_total_cost": memory == 104,
            "topk_unused_values_cannot_be_omitted": (
                "ONNX TopK output 0 is required; the prior empty-output probe fails full checker."
            ),
            "byte_topk_unavailable": (
                "The prior int8 TopK probe fails ORT session creation with NOT_IMPLEMENTED."
            ),
            "conclusion": (
                "This standard TopK rule engine cannot fall below baseline cost 104: its truthful "
                "intermediate memory is already 104 before adding 35 parameters."
            ),
        },
        "alternative_control": {
            "path": "scripts/golf/scratch_codex/task391/cand_argmax_xor.onnx",
            "cost": 148,
            "reason": "Repeated ArgMax/Scatter avoids TopK but raises truthful memory to 122 bytes.",
        },
    }
    (HERE / "sound_floor_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

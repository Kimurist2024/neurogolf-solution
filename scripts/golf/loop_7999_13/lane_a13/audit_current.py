#!/usr/bin/env python3
"""Structural and initializer audit of exact A13 baseline members."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402


TASKS = (20, 30, 31, 42, 55, 59, 64)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def main() -> None:
    rows = []
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        used = {name for node in model.graph.node for name in node.input if name}
        initializers = list(model.graph.initializer)
        duplicates = []
        for index, left in enumerate(initializers):
            left_array = numpy_helper.to_array(left)
            for right in initializers[index + 1:]:
                right_array = numpy_helper.to_array(right)
                if (
                    left_array.dtype == right_array.dtype
                    and left_array.shape == right_array.shape
                    and np.array_equal(left_array, right_array)
                ):
                    duplicates.append([left.name, right.name, int(left_array.size)])
        foreign_domains = sorted(
            {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
            | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
        )
        nonstatic = []
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            if not value.type.HasField("tensor_type") or any(
                not dim.HasField("dim_value") or dim.dim_value <= 0
                for dim in value.type.tensor_type.shape.dim
            ):
                nonstatic.append(value.name)
        rows.append(
            {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checker_full": "pass",
                "strict_shape_inference": "pass",
                "standard_domains": not foreign_domains,
                "foreign_domains": foreign_domains,
                "conv_bias_issues": check_conv_bias(model),
                "functions": len(model.functions),
                "sparse_initializers": len(model.graph.sparse_initializer),
                "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type.upper() in BANNED}),
                "nonstatic_tensors": nonstatic,
                "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
                "parameter_elements": sum(int(numpy_helper.to_array(item).size) for item in initializers),
                "unused_initializers": [item.name for item in initializers if item.name not in used],
                "identical_same_shape_initializers": duplicates,
                "node_count": len(model.graph.node),
                "op_histogram": {
                    op: sum(node.op_type == op for node in model.graph.node)
                    for op in sorted({node.op_type for node in model.graph.node})
                },
            }
        )
    (HERE / "current_model_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

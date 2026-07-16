#!/usr/bin/env python3
"""Structural and initializer audit of exact A16 members."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402


TASKS = (86, 114, 115, 193, 247, 259, 263)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def main() -> None:
    rows = []
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        onnx.checker.check_model(model, full_check=True)
        strict_status = "pass"
        strict_error = None
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        except Exception as exc:  # noqa: BLE001
            inferred = None
            strict_status = "fail"
            strict_error = f"{type(exc).__name__}: {exc}"
        used = {name for node in model.graph.node for name in node.input if name}
        initializers = list(model.graph.initializer)
        duplicates = []
        for index, left in enumerate(initializers):
            a = numpy_helper.to_array(left)
            for right in initializers[index + 1:]:
                b = numpy_helper.to_array(right)
                if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b):
                    duplicates.append([left.name, right.name, int(a.size)])
        foreign = sorted(
            {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
            | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
        )
        nonstatic = []
        if inferred is not None:
            for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
                if not value.type.HasField("tensor_type") or any(
                    not dim.HasField("dim_value") or dim.dim_value <= 0
                    for dim in value.type.tensor_type.shape.dim
                ):
                    nonstatic.append(value.name)
        rows.append(
            {
                "task": task, "path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "checker_full": "pass", "strict_shape_inference": strict_status,
                "strict_shape_inference_error": strict_error,
                "standard_domains": not foreign, "foreign_domains": foreign,
                "conv_bias_issues": check_conv_bias(model),
                "functions": len(model.functions), "sparse_initializers": len(model.graph.sparse_initializer),
                "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type.upper() in BANNED}),
                "nonstatic_tensors": nonstatic,
                "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
                "parameter_elements": sum(int(numpy_helper.to_array(item).size) for item in initializers),
                "unused_initializers": [item.name for item in initializers if item.name not in used],
                "identical_same_shape_initializers": duplicates,
                "node_count": len(model.graph.node),
            }
        )
    (HERE / "current_model_audit.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

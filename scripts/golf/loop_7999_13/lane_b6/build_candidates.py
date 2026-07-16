#!/usr/bin/env python3
"""Build exact initializer-sharing candidates from immutable B6 baselines."""

from __future__ import annotations

import json
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent


def build_task388_shared_axes() -> Path:
    model = onnx.load(HERE / "baseline_task388.onnx")
    slice_node = next(node for node in model.graph.node if node.op_type == "Slice")
    assert list(slice_node.input) == ["qb", "starts_bg", "ends_bg", "axes_slice"]
    slice_node.input[3] = "axes_all"

    kept = [init for init in model.graph.initializer if init.name != "axes_slice"]
    assert len(kept) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.name = "task388_shared_int64_axes"
    model.producer_name = "b6_exact_initializer_share"

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output = HERE / "candidate_task388_shared_axes.onnx"
    onnx.save(model, output)
    return output


if __name__ == "__main__":
    try:
        print(build_task388_shared_axes())
    except Exception as exc:
        rejection = {
            "task": 388,
            "candidate": "shared_axes_all_for_slice",
            "baseline_cost": 311,
            "nominal_candidate_cost": 308,
            "status": "rejected_before_save",
            "gate": "onnx_checker_full_check",
            "exception_type": type(exc).__name__,
            "reason": str(exc),
        }
        output = HERE / "build_rejections.json"
        output.write_text(json.dumps([rejection], indent=2, sort_keys=True) + "\n")
        print(output)

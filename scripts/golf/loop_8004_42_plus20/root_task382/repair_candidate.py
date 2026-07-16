#!/usr/bin/env python3
"""Remove stale shape annotations from the task382 rule-fix model."""

from __future__ import annotations

from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_headroom/candidates/task382.onnx"
OUTPUT = Path(__file__).resolve().parent / "task382_truthful_shapes.onnx"


def main() -> int:
    model = onnx.load(SOURCE)
    del model.graph.value_info[:]
    output_shape = model.graph.output[0].type.tensor_type.shape
    del output_shape.dim[:]
    for size in (1, 10, 30, 30):
        output_shape.dim.add().dim_value = size
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

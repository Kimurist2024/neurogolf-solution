#!/usr/bin/env python3
"""Make the spec-derived task280 Pad carrier declarations runtime-truthful."""

from __future__ import annotations

import copy
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/scratch_codex/task280/cand_pad20.onnx"
OUT = HERE / "candidate_task280_truthful.onnx"


def set_shape(value: onnx.ValueInfoProto, dimensions: list[int]) -> None:
    del value.type.tensor_type.shape.dim[:]
    for dimension in dimensions:
        value.type.tensor_type.shape.dim.add().dim_value = dimension


def main() -> int:
    model = onnx.load(SOURCE)
    required = {"Upad", "Vpad", "Rf", "Cf"}
    found = set()
    for value in model.graph.value_info:
        if value.name in required:
            set_shape(value, [1, 4, 30])
            found.add(value.name)
    if found != required:
        raise RuntimeError(f"missing value_info: {sorted(required - found)}")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    onnx.save(model, OUT)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

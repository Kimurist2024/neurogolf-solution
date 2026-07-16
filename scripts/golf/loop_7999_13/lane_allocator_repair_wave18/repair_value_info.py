#!/usr/bin/env python3
"""Repair selected stale value_info dimensions after exact dead-anchor removal."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--shape", required=True, help="Comma-separated positive dimensions")
    args = parser.parse_args()

    shape = [int(part) for part in args.shape.split(",")]
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"invalid shape: {shape}")

    model = onnx.load(args.input)
    matches = [value for value in model.graph.value_info if value.name == args.tensor]
    if len(matches) != 1:
        raise ValueError(f"expected one value_info for {args.tensor!r}, found {len(matches)}")
    tensor_type = matches[0].type.tensor_type
    del tensor_type.shape.dim[:]
    for dimension in shape:
        tensor_type.shape.dim.add().dim_value = dimension

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Test whether task200 ScatterElements accepts an int32 color index."""

from pathlib import Path

import onnx
from onnx import TensorProto


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "focus/candidates/task200_POLICY90_cost342_c659ae401e4c.onnx"
OUTPUT = HERE / "task200_int32_color_index.onnx"


def main() -> int:
    model = onnx.load(SOURCE)
    cast = next(node for node in model.graph.node if node.output == ["color_idx"])
    attribute = next(item for item in cast.attribute if item.name == "to")
    attribute.i = TensorProto.INT32
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

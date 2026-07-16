#!/usr/bin/env python3
"""Build exact task302 scalar-initializer variants from the LB-fixed model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "current" / "task302.onnx"
OUT_DIR = HERE / "candidates"


def main() -> int:
    model = onnx.load(str(SOURCE))
    expected = {"n29": 29, "n30": 30, "n31": 31}

    kept = []
    removed = set()
    for node in model.graph.node:
        if node.op_type == "ConstantOfShape" and len(node.output) == 1 and node.output[0] in expected:
            removed.add(node.output[0])
            continue
        kept.append(node)
    if removed != set(expected):
        raise RuntimeError(f"unexpected ConstantOfShape set: {removed}")
    del model.graph.node[:]
    model.graph.node.extend(kept)

    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray([value], dtype=np.int64), name=name)
            for name, value in expected.items()
        ]
    )

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "task302_scalar_inits.onnx"
    onnx.save(model, str(output))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Strengthen task243's blue-seed dominance without changing scored cost."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE / "candidates/task243_truthful_constant.onnx"
OUTPUT = HERE / "seed_weight_build.json"
MAGNITUDES = (192, 256, 384, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(magnitude: int) -> Path:
    model = onnx.load(BASE)
    weights = next(init for init in model.graph.initializer if init.name == "w_dyn")
    array = numpy_helper.to_array(weights).copy()
    if array.shape != (10,) or not np.array_equal(
        array, np.asarray([1, -128, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
    ):
        raise RuntimeError(f"unexpected base weights: {array}")
    array[1] = -float(magnitude)
    replacement = numpy_helper.from_array(array, "w_dyn")
    index = next(i for i, init in enumerate(model.graph.initializer) if init.name == "w_dyn")
    model.graph.initializer[index].CopyFrom(replacement)
    model.producer_name = "codex-task243-truthful-seed-weight"
    model.graph.node[0].name = f"truthful_terminal_einsum_blue{magnitude}"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    shape = [
        int(dim.dim_value)
        for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ]
    if shape != [1, 10, 30, 30]:
        raise RuntimeError(f"blue{magnitude}: wrong output shape {shape}")
    path = HERE / f"candidates/task243_truthful_blue{magnitude}.onnx"
    onnx.save(model, path)
    return path


def main() -> None:
    rows = []
    for magnitude in MAGNITUDES:
        path = build(magnitude)
        rows.append(
            {
                "blue_weight": -magnitude,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "params": 1010,
                "truthful_shapes": True,
            }
        )
        print(path)
    OUTPUT.write_text(
        json.dumps({"base": str(BASE.relative_to(ROOT)), "variants": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

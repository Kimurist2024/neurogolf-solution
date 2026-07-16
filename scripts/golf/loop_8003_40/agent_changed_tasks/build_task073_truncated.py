#!/usr/bin/env python3
"""Build task073 probes that remove the NaN-only sixth FIR tap."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "base_models" / "task073.onnx"
OUT = HERE / "task073_truncated"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(length: int) -> dict[str, object]:
    model = onnx.load(SOURCE)
    x = next(init for init in model.graph.initializer if init.name == "x")
    values = numpy_helper.to_array(x)[:, :, :length, :].copy()
    replacement = numpy_helper.from_array(values, name="x")
    for index, init in enumerate(model.graph.initializer):
        if init.name == "x":
            model.graph.initializer[index].CopyFrom(replacement)
            break
    conv = next(node for node in model.graph.node if node.op_type == "ConvTranspose")
    for attr in conv.attribute:
        if attr.name == "pads":
            attr.ints[:] = [0, 0, length - 1, 0]
            break

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"task073_len{length}.onnx"
    onnx.save(model, destination)
    return {
        "length": length,
        "source": str(SOURCE),
        "candidate": str(destination),
        "source_sha256": sha(SOURCE),
        "candidate_sha256": sha(destination),
        "x": values.reshape(-1).tolist(),
        "pads": [0, 0, length - 1, 0],
        "checker": "PASS",
        "strict_shape_inference": "PASS",
    }


def main() -> None:
    rows = [build(length) for length in (5, 4, 3, 2, 1)]
    path = HERE / "task073_truncated_build.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

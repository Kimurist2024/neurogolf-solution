#!/usr/bin/env python3
"""Build a cost-neutral task161 margin repair by scaling its output polynomial."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / (
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task161_r01_static186.onnx"
)
OUTPUT = HERE / "candidates/task161_cost186_margin8.onnx"
SOURCE_SHA256 = "6752eeea166c8111cda053c3cc36f54b1409d81c7553d672201792f646b31e3a"
SCALE = np.float32(8.0)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if digest(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("source SHA changed")
    model = onnx.load(SOURCE)
    matches = [index for index, item in enumerate(model.graph.initializer) if item.name == "poly"]
    if matches != [2]:
        raise RuntimeError(f"unexpected poly initializer indices: {matches}")
    index = matches[0]
    original = numpy_helper.to_array(model.graph.initializer[index]).astype(np.float32, copy=True)
    replacement = numpy_helper.from_array(original * SCALE, name="poly")
    model.graph.initializer[index].CopyFrom(replacement)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": SOURCE_SHA256,
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": digest(OUTPUT),
        "scale": float(SCALE),
        "poly_before": original.tolist(),
        "poly_after": (original * SCALE).tolist(),
    })


if __name__ == "__main__":
    main()

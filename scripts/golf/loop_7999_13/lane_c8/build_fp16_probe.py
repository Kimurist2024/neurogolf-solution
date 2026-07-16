#!/usr/bin/env python3
"""Build the only C8 incumbent for which the general FP16 pass does work."""

from __future__ import annotations

import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import dtype_passes  # noqa: E402


def main() -> None:
    source = onnx.load(HERE / "base/task209.onnx")
    candidate, stats = dtype_passes.g1_fp16_convert(source)
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    output = HERE / "task209_fp16_incumbent_probe.onnx"
    onnx.save(candidate, output)
    print(output)
    print(stats)


if __name__ == "__main__":
    main()

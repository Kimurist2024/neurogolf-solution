#!/usr/bin/env python3
"""Build exact power-of-two output scales of the LB-white task328 cost352 net."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task328_r01_static352.onnx"
OUT = HERE / "task328_scales"


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    rows = []
    source_blob = SOURCE.read_bytes()
    for exponent in (32, 36, 40, 44, 48):
        model = onnx.load_model_from_string(source_blob)
        uses = {item.name: 0 for item in model.graph.initializer}
        for node in model.graph.node:
            for name in node.input:
                if name in uses:
                    uses[name] += 1
        if uses.get("Rflip") != 1:
            raise RuntimeError("Rflip is no longer a one-use output factor")
        scale = np.float32(2.0 ** exponent)
        for index, item in enumerate(model.graph.initializer):
            if item.name != "Rflip":
                continue
            array = np.asarray(numpy_helper.to_array(item), dtype=np.float32) * scale
            if not np.isfinite(array).all():
                raise RuntimeError(f"nonfinite scale at 2^{exponent}")
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(array, item.name)
            )
            break
        model.producer_name = f"codex-task328-r01-scale2p{exponent}"
        onnx.checker.check_model(model, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(
            model, strict_mode=True, data_prop=True
        )
        shape = [int(dim.dim_value) for dim in
                 inferred.graph.output[0].type.tensor_type.shape.dim]
        if shape != [1, 10, 30, 30]:
            raise RuntimeError(f"bad inferred shape: {shape}")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"task328_r01_scale2p{exponent}.onnx"
        blob = model.SerializeToString()
        path.write_bytes(blob)
        rows.append({
            "exponent": exponent,
            "scale": float(scale),
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(blob),
            "shape": shape,
        })
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(source_blob),
        "scaled_initializer": "Rflip",
        "one_use": True,
        "candidates": rows,
    }
    (HERE / "task328_r01_scales_build.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

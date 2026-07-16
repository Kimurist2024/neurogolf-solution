#!/usr/bin/env python3
"""Apply a shared state gauge so task398 Q4 can replace D exactly."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"
OUTPUT = HERE / "task398_q4_d_reuse_347.onnx"


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    item = next(value for value in model.graph.initializer if value.name == name)
    item.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name))


def build() -> onnx.ModelProto:
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read("task398.onnx")
    (HERE / "baseline_task398.onnx").write_bytes(payload)
    model = onnx.load_model_from_string(payload)
    arrays = {item.name: numpy_helper.to_array(item).astype(np.float64) for item in model.graph.initializer}
    q4 = float(arrays["Q4"][0])
    gauge = np.array([1.0 / q4, -1.0, -1.0], dtype=np.float64)

    # Every Q and K occurrence shares the same state index.  Q'=GQ and
    # K'=G^-1 K leave all contractions unchanged.  Q4' becomes D exactly.
    for name in ("Q0", "Q1", "Q2", "Q3", "Q4"):
        transformed = arrays[name] * gauge
        if name == "Q4":
            transformed = arrays["D"].copy()
        replace(model, name, transformed)
    replace(model, "K", arrays["K"] / gauge[:, None, None, None])

    node = model.graph.node[-1]
    replaced = 0
    for index, name in enumerate(node.input):
        if name == "D":
            node.input[index] = "Q4"
            replaced += 1
    assert replaced == 5
    kept = [item for item in model.graph.initializer if item.name != "D"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)

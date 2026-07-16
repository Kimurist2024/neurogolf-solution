#!/usr/bin/env python3
"""Gauge task099's shared Tri state so DBc can reuse ST_v."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/submission_7999.13_wave17_candidate_meta.zip"
OUTPUT = HERE / "task099_dbc_stv_reuse_392.onnx"


def replace(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    item = next(value for value in model.graph.initializer if value.name == name)
    item.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(array, dtype=np.float32), name))


def build() -> onnx.ModelProto:
    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = archive.read("task099.onnx")
    (HERE / "baseline_task099.onnx").write_bytes(payload)
    model = onnx.load_model_from_string(payload)
    arrays = {item.name: numpy_helper.to_array(item).astype(np.float64) for item in model.graph.initializer}

    # Extend each full-row-rank 2x3 matrix by the same coordinate row.  The
    # resulting right gauge maps DBc exactly onto the existing ST_v matrix.
    tail = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    db_basis = np.concatenate([arrays["DBc"], tail], axis=0)
    st_basis = np.concatenate([arrays["ST_v"], tail], axis=0)
    right = np.linalg.solve(db_basis, st_basis)
    inverse = np.linalg.inv(right)
    np.testing.assert_allclose(arrays["DBc"] @ right, arrays["ST_v"], rtol=1e-12, atol=1e-12)

    # Every use of Tri's first state is paired with exactly one coefficient
    # bank (or LtC/LbC).  B'=BR and Tri'=R^-1 Tri preserve all contractions.
    replace(model, "Tri", np.einsum("xy,yuv->xuv", inverse, arrays["Tri"]))
    for name in ("RTc", "DTc", "RBc", "FTc", "FBc"):
        replace(model, name, arrays[name] @ right)
    for name in ("LtC", "LbC"):
        replace(model, name, arrays[name] @ right)

    final = model.graph.node[-1]
    replaced = 0
    for index, name in enumerate(final.input):
        if name == "DBc":
            final.input[index] = "ST_v"
            replaced += 1
    assert replaced == 1
    kept = [item for item in model.graph.initializer if item.name != "DBc"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(OUTPUT)

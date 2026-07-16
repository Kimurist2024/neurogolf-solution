#!/usr/bin/env python3
"""Materialize task347 B28 baseline/history and a truthful-shape control."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE_ZIP = ROOT / "submission_base_8000.46.zip"
BASELINE = HERE / "baseline_task347.onnx"
TRUTHFUL = HERE / "baseline_task347_truthful_shapes.onnx"
HISTORY_CLOAK = ROOT / "others/2/1203/71202/7974-2/task347.onnx"
HISTORY_HONEST = ROOT / "others/2/1300/submission7300+/task347.onnx"
HISTORY_CLOAK_COPY = HERE / "history_cost51_shape_cloak.onnx"
HISTORY_HONEST_COPY = HERE / "history_cost143_shape_honest.onnx"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_shape(value: onnx.ValueInfoProto, shape: list[int]) -> None:
    tensor_type = value.type.tensor_type
    del tensor_type.shape.dim[:]
    for size in shape:
        tensor_type.shape.dim.add().dim_value = size


def main() -> int:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        baseline_data = archive.read("task347.onnx")
    BASELINE.write_bytes(baseline_data)
    HISTORY_CLOAK_COPY.write_bytes(HISTORY_CLOAK.read_bytes())
    HISTORY_HONEST_COPY.write_bytes(HISTORY_HONEST.read_bytes())

    model = onnx.load(BASELINE)
    truthful = copy.deepcopy(model)
    values = {value.name: value for value in truthful.graph.value_info}
    set_shape(values["g"], [1, 10, 30, 30])
    set_shape(values["x"], [1, 10, 30, 30])
    onnx.checker.check_model(truthful, full_check=True)
    onnx.shape_inference.infer_shapes(
        copy.deepcopy(truthful), strict_mode=True, data_prop=True
    )
    onnx.save(truthful, TRUTHFUL)

    payload = {
        "baseline_zip": str(BASELINE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASELINE_ZIP),
        "artifacts": {
            path.name: {"sha256": digest(path), "bytes": path.stat().st_size}
            for path in (BASELINE, TRUTHFUL, HISTORY_CLOAK_COPY, HISTORY_HONEST_COPY)
        },
    }
    (HERE / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

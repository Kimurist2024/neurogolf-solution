#!/usr/bin/env python3
"""Compare every declared intermediate shape with its ORT runtime shape."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
FILES = [
    HERE / "task162_base.onnx",
    HERE / "task162_reuse_bool.onnx",
    HERE / "task268_base.onnx",
]


def declared_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def one(path: Path, disabled: bool) -> dict[str, object]:
    model = onnx.load(path)
    requested: list[tuple[str, list[int | str]]] = []
    graph_outputs = {value.name for value in model.graph.output}
    for value in model.graph.value_info:
        if value.name in graph_outputs:
            continue
        model.graph.output.append(copy.deepcopy(value))
        requested.append((value.name, declared_shape(value)))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    mode = "ORT_DISABLE_ALL" if disabled else "ORT_DEFAULT"
    try:
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        names = [name for name, _ in requested]
        arrays = session.run(names, {"input": np.zeros((1, 10, 30, 30), dtype=np.float32)})
    except Exception as exc:
        return {
            "mode": mode,
            "requested": len(requested),
            "session_or_run_ok": False,
            "error": repr(exc),
            "mismatch_count": None,
            "mismatches": [],
        }
    mismatches = []
    for (name, declared), array in zip(requested, arrays, strict=True):
        runtime = list(array.shape)
        if declared != runtime:
            mismatches.append({"name": name, "declared": declared, "runtime": runtime})
    return {
        "mode": mode,
        "requested": len(requested),
        "session_or_run_ok": True,
        "error": "",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    models = []
    for path in FILES:
        models.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runs": [one(path, True), one(path, False)],
            }
        )
    payload = {"models": models}
    output = HERE / "runtime_shape_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

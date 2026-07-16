#!/usr/bin/env python3
"""Fail-closed audit of task121's exact one-parameter shave probe."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = HERE / "baseline" / "task121.onnx"
CANDIDATE = HERE / "task121_clamped_start_probe.onnx"
OUT = HERE / "task121_probe_audit.json"


def shape(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(d.dim_value) if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]


def first_input() -> dict[str, object]:
    data = json.loads((ROOT / "inputs" / "neurogolf-2026" / "task121.json").read_text())
    grid = data["train"][0]["input"]
    import numpy as np

    array = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            array[0, color, row, col] = 1.0
    return {"input": array}


def runtime(path: Path) -> list[dict[str, object]]:
    rows = []
    for label, level in (
        ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        try:
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            output = session.run(["output"], first_input())[0]
            rows.append({"mode": label, "session": "ok", "run": "ok", "output_shape": list(output.shape)})
        except Exception as exc:
            rows.append({"mode": label, "session_or_run": f"{type(exc).__name__}: {exc}"})
    return rows


def main() -> None:
    base = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    onnx.checker.check_model(base, full_check=True)
    onnx.checker.check_model(candidate, full_check=True)
    base_vi = {value.name: shape(value) for value in base.graph.value_info}
    cand_vi = {value.name: shape(value) for value in candidate.graph.value_info}
    payload = {
        "task": 121,
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "baseline_cost": {"memory": 82, "params": 43, "cost": 125},
        "candidate_cost": {"memory": 82, "params": 42, "cost": 124},
        "rewrite": "Slice axis-1 start -12 and 0 both clamp to 0 for extent 10",
        "checker_full": True,
        "shape_cloak": {
            "present": True,
            "reason": (
                "GroupNormalization preserves the input [1,10,30,30] shape, "
                "but gnq is declared [1,1,1,1]; both CastLike outputs preserve "
                "that runtime shape and are declared [1,1,1,1] as well"
            ),
            "baseline_declared": {name: base_vi[name] for name in ("gnq", "qin", "qini")},
            "candidate_declared": {name: cand_vi[name] for name in ("gnq", "qin", "qini")},
        },
        "baseline_representative_runtime": runtime(BASELINE),
        "candidate_representative_runtime": runtime(CANDIDATE),
        "decision": "reject_shape_cloak_and_runtime_errors",
        "fresh5000_started": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

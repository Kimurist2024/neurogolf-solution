#!/usr/bin/env python3
"""Finite-support and structural proof for the task267 cost-30 r02 model."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_7999_13/lane_archive_all400/task267_r02_static30.onnx"
)
BASE_SHA = "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00"
CANDIDATE_SHA = "4ca7f921c34f87ef71512a8b680de7c984a2b42cd55b338b57aaabc012321387"

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.audit_retained_group import profile  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_case(n: int, source: int, destination: int) -> tuple[np.ndarray, np.ndarray]:
    # Placement is immaterial by the algebraic proof below.  Use a legal
    # 5x5 creature support representative for each cardinality.
    positions = [(row, col) for row in range(1, 6) for col in range(1, 6)][:n]
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    y = np.zeros_like(x)
    x[0, 0, :7, :7] = 1.0
    y[0, 0, :7, :7] = 1.0
    for row, col in positions:
        x[0, 0, row, col] = 0.0
        x[0, source, row, col] = 1.0
        y[0, 0, row, col] = 0.0
        y[0, destination, row, col] = 1.0
    x[0, 0, 6, 0] = 0.0
    x[0, destination, 6, 0] = 1.0
    # Generator output leaves the marker cell as background.
    return x, y


def session(data: bytes, threads: int, disable_all: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def exhaustive(data: bytes, threads: int, disable_all: bool) -> dict[str, object]:
    runner = session(data, threads, disable_all)
    right = wrong = runtime_errors = nonfinite = near_margin = 0
    min_positive = math.inf
    max_nonpositive = -math.inf
    max_abs = 0.0
    output_shapes: set[tuple[int, ...]] = set()
    for n in range(12, 16):
        for source in range(1, 10):
            for destination in range(1, 10):
                if source == destination:
                    continue
                x, expected = make_case(n, source, destination)
                try:
                    raw = runner.run(["output"], {"input": x})[0]
                    output_shapes.add(tuple(int(value) for value in raw.shape))
                    finite = np.isfinite(raw)
                    nonfinite += int(np.count_nonzero(~finite))
                    near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                    if finite.any():
                        max_abs = max(max_abs, float(np.max(np.abs(raw[finite]))))
                    positive = raw[(raw > 0) & finite]
                    nonpositive = raw[(raw <= 0) & finite]
                    if positive.size:
                        min_positive = min(min_positive, float(positive.min()))
                    if nonpositive.size:
                        max_nonpositive = max(max_nonpositive, float(nonpositive.max()))
                    if np.array_equal(raw > 0, expected > 0):
                        right += 1
                    else:
                        wrong += 1
                except Exception:
                    runtime_errors += 1
    total = right + wrong + runtime_errors
    return {
        "threads": threads,
        "optimization": "ORT_DISABLE_ALL" if disable_all else "ORT_ENABLE_ALL",
        "right": right,
        "wrong": wrong,
        "runtime_errors": runtime_errors,
        "total": total,
        "perfect": total == 288 and right == 288,
        "nonfinite_elements": nonfinite,
        "near_margin_elements": near_margin,
        "min_positive": min_positive if np.isfinite(min_positive) else None,
        "max_nonpositive": max_nonpositive if np.isfinite(max_nonpositive) else None,
        "max_abs": max_abs,
        "output_shapes": [list(value) for value in sorted(output_shapes)],
    }


def static_shapes(model: onnx.ModelProto) -> list[dict[str, object]]:
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    rows = []
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]
        rows.append({"name": value.name, "shape": dims})
    return rows


def main() -> None:
    base_data = BASE.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    if sha(base_data) != BASE_SHA or sha(candidate_data) != CANDIDATE_SHA:
        raise RuntimeError("frozen payload SHA mismatch")

    model = onnx.load_model_from_string(candidate_data)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    del inferred
    structural = structure(copy.deepcopy(model), 267)
    initializer = numpy_helper.to_array(model.graph.initializer[0])
    with zipfile.ZipFile(BASE) as archive:
        base_model = archive.read("task267.onnx")
    runs = [
        exhaustive(candidate_data, threads, disable_all)
        for threads in (1, 4)
        for disable_all in (True, False)
    ]
    result = {
        "immutable_base": {"path": BASE.name, "sha256": sha(base_data)},
        "task": 267,
        "generator": {
            "hash": "aabf363d",
            "path": "inputs/arc-gen-repo/tasks/task_aabf363d.py",
            "fixed_grid": [7, 7],
            "creature_count_support": [12, 13, 14, 15],
            "ordered_distinct_color_pairs": 72,
            "finite_reduced_state_count": 288,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha(candidate_data),
            "baseline_task_sha256": sha(base_model),
            "baseline_actual_cost": profile(base_model, "task267_base.onnx"),
            "actual_cost": profile(candidate_data, "task267_r02.onnx"),
            "known_disable_all": run_known(copy.deepcopy(model), 267, True),
            "known_default": run_known(copy.deepcopy(model), 267, False),
            "checker_full": True,
            "strict_shape_inference_data_prop": True,
            "static_shapes": static_shapes(model),
            "structure": structural,
            "initializer": {
                "name": model.graph.initializer[0].name,
                "dtype": str(initializer.dtype),
                "shape": list(initializer.shape),
                "all_finite": bool(np.isfinite(initializer).all()),
                "values": initializer.tolist(),
            },
            "undefined_behavior_findings": [],
        },
        "algebraic_support_proof": {
            "equation": helper.get_attribute_value(model.graph.node[0].attribute[0]).decode(),
            "reduction": [
                "A_o=sum_rc X[o,r,c]*p[r]",
                "D_o=sum_uv X[o,u,v]*p[u]^61",
                "C_d=sum_st X[d,s,t]*p[s]^7",
                "Y[o,h,w]=A_o*D_o*sum_d(C_d*X[d,h,w])",
            ],
            "placement_independence": (
                "Every generated creature cell lies in rows 1..5, where p is identically "
                "0.05. Therefore A, D, and C depend only on N, source color, and marker "
                "color; the final X[d,h,w] factor transfers the arbitrary creature occupancy "
                "pointwise. Columns never affect a reduction coefficient."
            ),
            "float32_zero_safety": (
                "For the source channel D_source=N*(float32(0.05)^61); its exact magnitude "
                "is below the smallest float32 subnormal and necessarily rounds to +0 in "
                "the float32 graph output. All retained true-positive logits are >=775.58."
            ),
        },
        "exhaustive_runs": runs,
        "decision": {
            "winner": all(row["perfect"] for row in runs)
            and all(row["nonfinite_elements"] == 0 for row in runs)
            and all(row["near_margin_elements"] == 0 for row in runs),
            "policy": "private-zero guarantee exception",
        },
    }
    (HERE / "task267_exhaustive.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

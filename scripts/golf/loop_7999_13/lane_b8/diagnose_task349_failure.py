#!/usr/bin/env python3
"""Reproduce and characterize the task349 fresh failure at seed 349101."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT))

import task_db93a21d as generator  # noqa: E402
from scripts.lib.scoring import sanitize_model  # noqa: E402


MODELS = {
    "exact_baseline_3964": HERE / "baseline_task349.onnx",
    "table_crop_3956": HERE / "candidates/task349_radius_tables_len9.onnx",
    "relation_3954": HERE / "candidates/task349_radius_tables_len9_top_relation.onnx",
    "prior_nominal_sound_4861": ROOT / "scripts/golf/scratch_codex/task349/k6_iter_runs_i8_sparse_v13_scalar_zero.onnx",
}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def session(model: onnx.ModelProto) -> ort.InferenceSession:
    sanitized = sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def main() -> int:
    random.seed(349101)
    np.random.seed(349101)
    case = None
    for _ in range(421):
        case = generator.generate()
    assert case is not None
    x = encode(case["input"])
    want = encode(case["output"]).astype(bool)

    results: dict[str, object] = {}
    for label, path in MODELS.items():
        got = session(onnx.load(path)).run(["output"], {"input": x})[0] > 0.0
        diff = np.argwhere(got != want)
        results[label] = {
            "path": str(path.relative_to(ROOT)),
            "different_entries": int(len(diff)),
            "differences": diff.tolist(),
        }

    baseline = onnx.load(MODELS["exact_baseline_3964"])
    inferred = onnx.shape_inference.infer_shapes(baseline, strict_mode=True)
    value_map = {value.name: value for value in inferred.graph.value_info}
    internal_names = [
        "R",
        "bottom_true",
        "patch_sumR",
        "h_patch_mask",
        "h_patch_updates",
        "halo_diff",
        "H",
    ]
    for name in internal_names:
        baseline.graph.output.append(value_map[name])
    internal_values = session(baseline).run(None, {"input": x})[1:]
    internals = {
        name: np.asarray(value).reshape(-1).tolist()
        for name, value in zip(internal_names, internal_values, strict=True)
    }

    report = {
        "task": 349,
        "seed": 349101,
        "valid_case": 421,
        "shape": [len(case["input"]), len(case["input"][0])],
        "input": case["input"],
        "expected_output": case["output"],
        "models": results,
        "baseline_internals": internals,
        "diagnosis": {
            "patch_signature": int(internals["patch_sumR"][0]),
            "matched_hardcoded_signature": 495564,
            "hardcoded_patch_indices": [9, 12],
            "hardcoded_patch_updates": [-24576, 24576],
            "effect": "false-positive patch deletes green halo at rows 9..11, cols 13..14",
            "conclusion": "the exact baseline and all derived candidates are not generator-complete",
        },
    }
    out = HERE / "task349_failure_case.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["diagnosis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

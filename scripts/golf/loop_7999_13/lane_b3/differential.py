#!/usr/bin/env python3
"""Independent generator-distribution differential with explicit error accounting."""

from __future__ import annotations

import argparse
import importlib
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

from scripts.lib.scoring import sanitize_model  # noqa: E402


HASHES = {89: "3e980e27", 208: "890034e9"}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def session(path: Path) -> ort.InferenceSession:
    model = sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=sorted(HASHES))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    module = importlib.import_module(f"task_{HASHES[args.task]}")
    baseline = session(args.baseline)
    candidate = session(args.candidate)
    report = {
        "task": args.task,
        "seed": args.seed,
        "requested": args.count,
        "attempts": 0,
        "executable": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "mismatches": 0,
        "both_failed": 0,
        "one_sided_failed": 0,
        "baseline_failed_only": 0,
        "candidate_failed_only": 0,
        "max_abs_difference": 0.0,
        "first_failure": None,
    }
    valid = 0
    while valid < args.count:
        report["attempts"] += 1
        case = module.generate()
        if max(len(case["input"]), len(case["input"][0])) > 30:
            continue
        valid += 1
        data = {"input": encode(case["input"])}
        baseline_raw = candidate_raw = None
        baseline_error = candidate_error = None
        try:
            baseline_raw = baseline.run(["output"], data)[0]
        except Exception as exc:
            baseline_error = repr(exc)
        try:
            candidate_raw = candidate.run(["output"], data)[0]
        except Exception as exc:
            candidate_error = repr(exc)
        if baseline_error or candidate_error:
            if baseline_error and candidate_error:
                report["both_failed"] += 1
            else:
                report["one_sided_failed"] += 1
                if baseline_error:
                    report["baseline_failed_only"] += 1
                else:
                    report["candidate_failed_only"] += 1
            if report["first_failure"] is None:
                report["first_failure"] = {
                    "case": valid,
                    "baseline_error": baseline_error,
                    "candidate_error": candidate_error,
                }
            continue
        report["executable"] += 1
        if np.array_equal(baseline_raw, candidate_raw):
            report["raw_equal"] += 1
        if np.array_equal(baseline_raw > 0, candidate_raw > 0):
            report["threshold_equal"] += 1
        else:
            report["mismatches"] += 1
            if report["first_failure"] is None:
                report["first_failure"] = {"case": valid, "threshold_mismatch": True}
        report["max_abs_difference"] = max(
            report["max_abs_difference"],
            float(np.max(np.abs(baseline_raw.astype(np.float64) - candidate_raw.astype(np.float64)))),
        )

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["mismatches"] == 0 and report["one_sided_failed"] == 0 and report["both_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

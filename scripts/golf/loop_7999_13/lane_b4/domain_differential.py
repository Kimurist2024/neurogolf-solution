#!/usr/bin/env python3
"""Independent generator differential for the final task107 B4 candidate."""

from __future__ import annotations

import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import sanitize_model  # noqa: E402


TASK = 107
TASK_HASH = "469497ad"
COUNT = 3000
SEED = 799_913_107
BASELINE = HERE / "baseline_task107.onnx"
CANDIDATE = HERE / "candidate_task107_shared_coefficients_rank4.onnx"
OUTPUT = HERE / "task107_rank4_domain_differential.json"


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
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options)


def main() -> int:
    generator = importlib.import_module(f"task_{TASK_HASH}")
    baseline_session = session(BASELINE)
    candidate_session = session(CANDIDATE)
    random.seed(SEED)

    start = time.monotonic()
    valid = attempts = oversize = generation_errors = 0
    baseline_errors = candidate_errors = candidate_only_errors = 0
    decoded_equal = raw_equal = baseline_gold = candidate_gold = 0
    max_abs_raw_difference = 0.0
    first_failure: dict[str, object] | None = None
    while valid < COUNT:
        attempts += 1
        try:
            case = generator.generate()
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if first_failure is None:
                first_failure = {"stage": "generate", "attempt": attempts, "error": repr(exc)}
            continue
        height = len(case["input"])
        width = len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        encoded = encode(case["input"])
        expected = encode(case["output"]).astype(bool)
        baseline_raw = candidate_raw = None
        try:
            baseline_raw = baseline_session.run(["output"], {"input": encoded})[0]
        except Exception as exc:  # noqa: BLE001
            baseline_errors += 1
            if first_failure is None:
                first_failure = {"stage": "baseline", "valid_case": valid, "error": repr(exc)}
        try:
            candidate_raw = candidate_session.run(["output"], {"input": encoded})[0]
        except Exception as exc:  # noqa: BLE001
            candidate_errors += 1
            if baseline_raw is not None:
                candidate_only_errors += 1
            if first_failure is None:
                first_failure = {"stage": "candidate", "valid_case": valid, "error": repr(exc)}
        if baseline_raw is None or candidate_raw is None:
            continue
        baseline_decoded = baseline_raw > 0.0
        candidate_decoded = candidate_raw > 0.0
        if np.array_equal(baseline_decoded, expected):
            baseline_gold += 1
        if np.array_equal(candidate_decoded, expected):
            candidate_gold += 1
        if np.array_equal(baseline_decoded, candidate_decoded):
            decoded_equal += 1
        elif first_failure is None:
            difference = np.argwhere(baseline_decoded != candidate_decoded)
            first_failure = {
                "stage": "decoded_differential",
                "valid_case": valid,
                "shape": [height, width],
                "different_elements": int(len(difference)),
                "first_difference": difference[0].tolist() if len(difference) else None,
            }
        if np.array_equal(baseline_raw, candidate_raw):
            raw_equal += 1
        else:
            finite_difference = np.abs(
                np.nan_to_num(baseline_raw, nan=0.0, posinf=0.0, neginf=0.0)
                - np.nan_to_num(candidate_raw, nan=0.0, posinf=0.0, neginf=0.0)
            )
            max_abs_raw_difference = max(max_abs_raw_difference, float(finite_difference.max()))

    report = {
        "task": TASK,
        "baseline": str(BASELINE.relative_to(ROOT)),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "seed": SEED,
        "requested_valid": COUNT,
        "attempts": attempts,
        "valid": valid,
        "oversize_skipped": oversize,
        "generation_errors": generation_errors,
        "baseline_runtime_errors": baseline_errors,
        "candidate_runtime_errors": candidate_errors,
        "candidate_only_runtime_errors": candidate_only_errors,
        "baseline_gold_correct": baseline_gold,
        "candidate_gold_correct": candidate_gold,
        "decoded_equal": decoded_equal,
        "raw_bitwise_equal": raw_equal,
        "max_abs_raw_difference": max_abs_raw_difference,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - start,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    passed = (
        valid == COUNT
        and generation_errors == 0
        and baseline_errors == 0
        and candidate_errors == 0
        and candidate_only_errors == 0
        and baseline_gold == COUNT
        and candidate_gold == COUNT
        and decoded_equal == COUNT
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

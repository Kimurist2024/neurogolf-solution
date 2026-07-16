#!/usr/bin/env python3
"""Compare the task349 incumbent and a semantics-preserving shave on fresh data.

This audit is deliberately stricter than a normal accuracy check: it records
raw output equality between the two ONNX graphs, gold accuracy, and runtime
errors under both disabled and default ORT graph optimization.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[3]
TASKS = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASKS))
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import sanitize_model  # noqa: E402


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def make_session(path: Path, mode: str) -> ort.InferenceSession:
    model = sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options)


def audit_mode(
    incumbent_path: Path,
    candidate_path: Path,
    mode: str,
    count: int,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module("task_db93a21d")
    incumbent = make_session(incumbent_path, mode)
    candidate = make_session(candidate_path, mode)

    result: dict[str, object] = {
        "mode": mode,
        "seed": seed,
        "requested": count,
        "attempts": 0,
        "valid": 0,
        "oversize": 0,
        "raw_equal": 0,
        "raw_different": 0,
        "incumbent_gold_right": 0,
        "candidate_gold_right": 0,
        "incumbent_errors": 0,
        "candidate_errors": 0,
        "first_raw_difference": None,
        "first_incumbent_gold_failure": None,
        "first_candidate_gold_failure": None,
        "first_error": None,
    }

    while result["valid"] < count:
        result["attempts"] += 1
        case = generator.generate()
        height = len(case["input"])
        width = len(case["input"][0])
        if max(height, width) > 30:
            result["oversize"] += 1
            continue
        result["valid"] += 1
        case_index = int(result["valid"])
        input_array = encode(case["input"])
        gold = encode(case["output"]) > 0.0

        incumbent_output = candidate_output = None
        try:
            incumbent_output = incumbent.run(["output"], {"input": input_array})[0]
        except Exception as exc:  # pragma: no cover - evidence path
            result["incumbent_errors"] += 1
            if result["first_error"] is None:
                result["first_error"] = {
                    "case": case_index,
                    "model": "incumbent",
                    "error": repr(exc),
                }
        try:
            candidate_output = candidate.run(["output"], {"input": input_array})[0]
        except Exception as exc:  # pragma: no cover - evidence path
            result["candidate_errors"] += 1
            if result["first_error"] is None:
                result["first_error"] = {
                    "case": case_index,
                    "model": "candidate",
                    "error": repr(exc),
                }

        if incumbent_output is None or candidate_output is None:
            continue

        if np.array_equal(incumbent_output, candidate_output):
            result["raw_equal"] += 1
        else:
            result["raw_different"] += 1
            if result["first_raw_difference"] is None:
                differing = np.argwhere(incumbent_output != candidate_output)
                result["first_raw_difference"] = {
                    "case": case_index,
                    "shape": [height, width],
                    "different_values": int(len(differing)),
                    "first_index": differing[0].tolist() if len(differing) else None,
                }

        incumbent_right = np.array_equal(incumbent_output > 0.0, gold)
        candidate_right = np.array_equal(candidate_output > 0.0, gold)
        if incumbent_right:
            result["incumbent_gold_right"] += 1
        elif result["first_incumbent_gold_failure"] is None:
            result["first_incumbent_gold_failure"] = {
                "case": case_index,
                "shape": [height, width],
            }
        if candidate_right:
            result["candidate_gold_right"] += 1
        elif result["first_candidate_gold_failure"] is None:
            result["first_candidate_gold_failure"] = {
                "case": case_index,
                "shape": [height, width],
            }

    result["raw_equivalent"] = (
        result["raw_equal"] == count
        and result["raw_different"] == 0
        and result["incumbent_errors"] == 0
        and result["candidate_errors"] == 0
    )
    result["candidate_not_worse"] = (
        result["candidate_gold_right"] >= result["incumbent_gold_right"]
        and result["candidate_errors"] <= result["incumbent_errors"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=8002349)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    modes = [
        audit_mode(args.incumbent, args.candidate, "disabled", args.count, args.seed),
        audit_mode(args.incumbent, args.candidate, "default", args.count, args.seed),
    ]
    payload = {
        "task": 349,
        "incumbent": str(args.incumbent),
        "candidate": str(args.candidate),
        "count_per_mode": args.count,
        "modes": modes,
        "accept_equivalence_gate": all(
            bool(row["raw_equivalent"]) and bool(row["candidate_not_worse"])
            for row in modes
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["accept_equivalence_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fresh generator audit in either ORT mode; count all requested cases."""

from __future__ import annotations

import argparse
import copy
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
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT))

from scripts.lib.scoring import sanitize_model  # noqa: E402


HASHES = {138: "5daaa586", 187: "7b6016b9"}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def make_session(path: Path, mode: str) -> ort.InferenceSession:
    model = sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=["disabled", "default"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    module = importlib.import_module(f"task_{HASHES[args.task]}")
    start = time.monotonic()
    try:
        session = make_session(args.onnx, args.mode)
    except Exception as exc:  # noqa: BLE001
        report = {
            "task": args.task,
            "onnx": str(args.onnx.resolve().relative_to(ROOT)),
            "seed": args.seed,
            "mode": args.mode,
            "requested_valid": args.count,
            "valid": 0,
            "correct": 0,
            "wrong": 0,
            "errors": 1,
            "accuracy": 0.0,
            "session_error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - start,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 2

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    valid = attempts = correct = wrong = errors = oversize = 0
    first_failure: dict[str, object] | None = None
    while valid < args.count:
        attempts += 1
        try:
            case = module.generate()
        except Exception as exc:  # noqa: BLE001
            errors += 1
            first_failure = first_failure or {
                "stage": "generate", "attempt": attempts, "error": repr(exc)
            }
            continue
        height, width = len(case["input"]), len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        try:
            raw = session.run([output_name], {input_name: encode(case["input"])})[0]
            got = raw > 0.0
            want = encode(case["output"]).astype(bool)
            if np.array_equal(got, want):
                correct += 1
            else:
                wrong += 1
                if first_failure is None:
                    diff = np.argwhere(got != want)
                    first_failure = {
                        "valid_case": valid,
                        "attempt": attempts,
                        "shape": [height, width],
                        "different_entries": int(len(diff)),
                        "first_difference": diff[0].tolist() if len(diff) else None,
                    }
        except Exception as exc:  # noqa: BLE001
            errors += 1
            first_failure = first_failure or {
                "stage": "inference",
                "valid_case": valid,
                "attempt": attempts,
                "error": repr(exc),
            }

    report = {
        "task": args.task,
        "onnx": str(args.onnx.resolve().relative_to(ROOT)),
        "seed": args.seed,
        "mode": args.mode,
        "requested_valid": args.count,
        "valid": valid,
        "attempts": attempts,
        "oversize_skipped": oversize,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "accuracy": correct / valid if valid else 0.0,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - start,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if correct == args.count and wrong == 0 and errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

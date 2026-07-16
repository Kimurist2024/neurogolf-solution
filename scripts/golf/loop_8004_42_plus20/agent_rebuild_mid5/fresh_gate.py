#!/usr/bin/env python3
"""Known/fresh dual-ORT exactness gate for this lane."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[4]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
HASHES = {23: "150deff5", 187: "7b6016b9", 209: "8a004b2b", 367: "e73095fd"}


def onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    output = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(values.shape)
    output[0, values, rows, cols] = 1.0
    return output


def session(path: Path, mode: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=920_000_000)
    parser.add_argument("--mode", choices=("disabled", "default"), required=True)
    parser.add_argument("--known", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(TASK_DIR))
    common = importlib.import_module("common")
    generator = importlib.import_module(f"task_{HASHES[args.task]}")
    runtime = session(args.onnx, args.mode)

    if args.known:
        data = json.loads(
            (ROOT / f"inputs/neurogolf-2026/task{args.task:03d}.json").read_text()
        )
        cases = [
            item
            for split in ("train", "test", "arc-gen")
            for item in data.get(split, [])
        ]
    else:
        cases = None

    failures = runtime_errors = unstable = 0
    total = len(cases) if cases is not None else args.count
    first_failure = None
    for index in range(total):
        seed = args.seed + index
        if cases is None:
            random.seed(seed)
            common.random.seed(seed)
            example = generator.generate()
        else:
            example = cases[index]
        try:
            raw = runtime.run(None, {"input": onehot(example["input"])})[0]
        except Exception as exc:  # noqa: BLE001
            runtime_errors += 1
            if first_failure is None:
                first_failure = {"index": index, "seed": seed, "error": repr(exc)}
            continue
        actual = np.asarray(raw) > 0
        expected = onehot(example["output"]).astype(bool)
        if np.any((raw > 0) & (raw < 0.25)):
            unstable += 1
        if not np.array_equal(actual, expected):
            failures += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "seed": seed,
                    "diff": np.argwhere(actual != expected)[:20].tolist(),
                    "input": example["input"],
                    "expected": example["output"],
                }
    result = {
        "task": args.task,
        "path": str(args.onnx),
        "stream": "known" if args.known else "fresh",
        "mode": args.mode,
        "seed": args.seed,
        "total": total,
        "failures": failures,
        "accuracy": (total - failures - runtime_errors) / total,
        "runtime_errors": runtime_errors,
        "unstable": unstable,
        "first_failure": first_failure,
    }
    print(json.dumps(result))
    return 0 if failures == runtime_errors == unstable == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent two-seed fresh and raw-equivalence validation for task101."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
BASE = HERE / "baseline/task101.onnx"
CANDIDATE = HERE / "candidates/task101_exact_broadcast_expand_a57a944d958b.onnx"
# Disjoint seed ranges: each stream consumes exactly 5,000 consecutive seeds.
SEEDS = (1_011_181, 2_011_181)
COUNT = 5_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(arr.shape)
    out[0, arr, rows, cols] = 1.0
    return out


def session(path: Path, disabled: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path.read_bytes(), options, providers=["CPUExecutionProvider"])


def main() -> None:
    sys.path.insert(0, str(TASKS))
    common = importlib.import_module("common")
    task = importlib.import_module("task_447fd412")
    models = {
        "base_disabled": session(BASE, True),
        "candidate_disabled": session(CANDIDATE, True),
        "base_default": session(BASE, False),
        "candidate_default": session(CANDIDATE, False),
    }
    result: dict[str, object] = {
        "task": 101,
        "generator": "inputs/arc-gen-repo/tasks/task_447fd412.py",
        "generator_sha256": hashlib.sha256((TASKS / "task_447fd412.py").read_bytes()).hexdigest(),
        "base_path": str(BASE.relative_to(ROOT)),
        "base_sha256": sha256(BASE),
        "candidate_path": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": sha256(CANDIDATE),
        "seeds": {},
    }
    for start_seed in SEEDS:
        counters = {
            name: {"right": 0, "wrong": 0, "runtime_errors": 0, "nonfinite": 0}
            for name in models
        }
        comparisons = {
            "disabled_raw_equal": 0,
            "default_raw_equal": 0,
            "candidate_dual_raw_equal": 0,
            "base_dual_raw_equal": 0,
            "disabled_decoded_equal": 0,
            "default_decoded_equal": 0,
        }
        max_raw_delta = {key: 0.0 for key in comparisons if "raw" in key}
        shapes: Counter[str] = Counter()
        first_failure: dict[str, object] | None = None
        valid = skipped = 0
        seed = start_seed
        while valid < COUNT:
            random.seed(seed)
            common.random.seed(seed)
            example = task.generate()
            inp = example["input"]
            expected = example["output"]
            if len(inp) > 30 or len(inp[0]) > 30:
                skipped += 1
                seed += 1
                continue
            x = onehot(inp)
            wanted = onehot(expected).astype(bool)
            shapes[f"{len(inp)}x{len(inp[0])}"] += 1
            outputs: dict[str, np.ndarray] = {}
            for name, sess in models.items():
                try:
                    raw = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
                    outputs[name] = raw
                    if not np.all(np.isfinite(raw)):
                        counters[name]["nonfinite"] += 1
                    if np.array_equal(raw > 0, wanted):
                        counters[name]["right"] += 1
                    else:
                        counters[name]["wrong"] += 1
                        if first_failure is None:
                            first_failure = {"seed": seed, "model": name, "kind": "wrong"}
                except Exception as exc:  # noqa: BLE001
                    counters[name]["runtime_errors"] += 1
                    if first_failure is None:
                        first_failure = {
                            "seed": seed,
                            "model": name,
                            "kind": "runtime_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            pairs = {
                "disabled": ("base_disabled", "candidate_disabled"),
                "default": ("base_default", "candidate_default"),
                "candidate_dual": ("candidate_disabled", "candidate_default"),
                "base_dual": ("base_disabled", "base_default"),
            }
            for label, (left, right) in pairs.items():
                if left not in outputs or right not in outputs:
                    continue
                a, b = outputs[left], outputs[right]
                raw_key = f"{label}_raw_equal"
                if np.array_equal(a, b):
                    comparisons[raw_key] += 1
                max_raw_delta[raw_key] = max(max_raw_delta[raw_key], float(np.max(np.abs(a - b))))
                decoded_key = f"{label}_decoded_equal"
                if decoded_key in comparisons and np.array_equal(a > 0, b > 0):
                    comparisons[decoded_key] += 1
            valid += 1
            seed += 1
        result["seeds"][str(start_seed)] = {
            "requested_valid": COUNT,
            "valid": valid,
            "skipped_over_30": skipped,
            "counters": counters,
            "comparisons": comparisons,
            "max_raw_delta": max_raw_delta,
            "shape_counts": dict(sorted(shapes.items())),
            "first_failure": first_failure,
        }
        (HERE / "fresh_dual_raw_2x5000.json").write_text(json.dumps(result, indent=2) + "\n")
        print(start_seed, counters, comparisons, max_raw_delta, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fresh platform/margin audit for the exact task333 sign rewrite."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
CANDIDATE = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS))
from lib import scoring  # noqa: E402


CONFIGS = {
    "disable_all_threads1": (True, 1),
    "disable_all_threads4": (True, 4),
    "default_threads1": (False, 1),
    "default_threads4": (False, 4),
}
SEEDS = (33308101, 33308102)


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_from_string(data)))
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--per-seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    disable, threads = CONFIGS[args.config]
    candidate_data = CANDIDATE.read_bytes()
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as archive:
        baseline_data = archive.read("task333.onnx")
    candidate = make_session(candidate_data, disable, threads)
    baseline = make_session(baseline_data, disable, threads)
    generator = importlib.import_module("task_d43fd935")

    totals = {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "sign_differences_vs_baseline": 0,
        "raw_different_values_vs_baseline": 0,
        "max_abs_raw_difference_vs_baseline": 0.0,
        "min_positive": None,
        "max_abs_raw": 0.0,
        "first_failure": None,
    }
    seed_rows = []
    started = time.time()
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        seed_right = seed_wrong = seed_runtime = 0
        for index in range(args.per_seed):
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise RuntimeError({"seed": seed, "index": index, "conversion": "failed"})
            try:
                feed = {candidate.get_inputs()[0].name: benchmark["input"]}
                candidate_raw = candidate.run([candidate.get_outputs()[0].name], feed)[0]
                base_feed = {baseline.get_inputs()[0].name: benchmark["input"]}
                baseline_raw = baseline.run([baseline.get_outputs()[0].name], base_feed)[0]
            except Exception as exc:  # noqa: BLE001
                seed_runtime += 1
                totals["runtime_errors"] += 1
                totals["first_failure"] = totals["first_failure"] or {
                    "seed": seed,
                    "index": index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            finite = np.isfinite(candidate_raw)
            totals["nonfinite_values"] += int(candidate_raw.size - np.count_nonzero(finite))
            safe = candidate_raw[finite]
            if safe.size:
                positive = safe[safe > 0]
                totals["near_positive_values"] += int(np.count_nonzero((safe > 0) & (safe < 0.25)))
                if positive.size:
                    value = float(positive.min())
                    totals["min_positive"] = value if totals["min_positive"] is None else min(totals["min_positive"], value)
                totals["max_abs_raw"] = max(totals["max_abs_raw"], float(np.abs(safe).max(initial=0.0)))
            unequal = candidate_raw != baseline_raw
            totals["raw_different_values_vs_baseline"] += int(np.count_nonzero(unequal))
            totals["sign_differences_vs_baseline"] += int(np.count_nonzero((candidate_raw > 0) != (baseline_raw > 0)))
            common_finite = finite & np.isfinite(baseline_raw)
            if common_finite.any():
                totals["max_abs_raw_difference_vs_baseline"] = max(
                    totals["max_abs_raw_difference_vs_baseline"],
                    float(np.abs(candidate_raw[common_finite] - baseline_raw[common_finite]).max(initial=0.0)),
                )
            expected = benchmark["output"].astype(bool)
            if np.array_equal(candidate_raw > 0, expected):
                totals["right"] += 1
                seed_right += 1
            else:
                totals["wrong"] += 1
                seed_wrong += 1
                totals["first_failure"] = totals["first_failure"] or {
                    "seed": seed,
                    "index": index,
                    "different_cells": int(np.count_nonzero((candidate_raw > 0) != expected)),
                }
            if (index + 1) % 500 == 0:
                print(args.config, seed, index + 1, "right", seed_right, "wrong", seed_wrong, flush=True)
        seed_rows.append({"seed": seed, "total": args.per_seed, "right": seed_right, "wrong": seed_wrong, "runtime_errors": seed_runtime})

    total = args.per_seed * len(SEEDS)
    result = {
        "task": 333,
        "config": args.config,
        "seeds": seed_rows,
        "total": total,
        **totals,
        "perfect": (
            totals["right"] == total
            and totals["wrong"] == totals["runtime_errors"] == totals["nonfinite_values"] == totals["near_positive_values"] == totals["sign_differences_vs_baseline"] == 0
        ),
        "elapsed_seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

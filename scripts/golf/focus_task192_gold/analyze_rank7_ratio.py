#!/usr/bin/env python3
"""Measure the exact decision-ratio gap of task192's current rank-7 kernel."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))


def planes(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(example["input"], dtype=np.int8)
    expected = np.asarray(example["output"], dtype=np.int8)
    height, width = grid.shape
    inside = np.zeros((30, 30), dtype=np.float64)
    nonzero = np.zeros((30, 30), dtype=np.float64)
    selected = np.zeros((30, 30), dtype=np.float64)
    keep = np.zeros((30, 30), dtype=np.bool_)
    color = max(range(1, 10), key=grid.reshape(-1).tolist().count)
    inside[:height, :width] = 1
    nonzero[:height, :width] = grid != 0
    selected[:height, :width] = grid == color
    keep[:height, :width] = expected == color
    return inside, nonzero, selected, keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=192_426_001)
    args = parser.parse_args()
    with zipfile.ZipFile(ROOT / "submission_base_8014.69.zip") as archive:
        model = onnx.load_model_from_string(archive.read("task192.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    factor = arrays["eig_factor"].astype(np.float64)
    kernel = factor @ factor.T
    generator = importlib.import_module("task_7e0986d6")
    random.seed(args.seed)
    positive_min = float("inf")
    negative_max = -float("inf")
    positive_count = negative_count = 0
    invalid_base = 0
    current_errors = 0
    current_threshold = float(-arrays["route_out"][0, 2] / arrays["route_out"][1, 2])
    examples = []
    known = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
    examples.extend(known[part] for part in ())
    stream = [generator.generate() for _ in range(args.count)]
    for example in stream:
        inside, nonzero, selected, keep = planes(example)
        h0 = inside @ kernel
        v0 = kernel @ inside
        h1 = selected @ kernel
        v1 = kernel @ selected
        base = inside * h0 * v0
        predicate = nonzero * h1 * v1
        valid = inside > 0
        bad_base = valid & (base <= 0)
        invalid_base += int(np.count_nonzero(bad_base))
        safe = valid & (base > 0)
        ratio = np.zeros_like(base)
        ratio[safe] = predicate[safe] / base[safe]
        positives = safe & keep
        negatives = safe & ~keep
        if np.any(positives):
            positive_min = min(positive_min, float(ratio[positives].min()))
            positive_count += int(np.count_nonzero(positives))
        if np.any(negatives):
            negative_max = max(negative_max, float(ratio[negatives].max()))
            negative_count += int(np.count_nonzero(negatives))
        predicted = valid & nonzero.astype(bool) & (ratio > current_threshold)
        current_errors += int(not np.array_equal(predicted, keep))
    report = {
        "seed": args.seed,
        "cases": args.count,
        "rank": int(factor.shape[1]),
        "current_threshold": current_threshold,
        "current_case_errors_exact_argmax": current_errors,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio_min": positive_min,
        "negative_ratio_max": negative_max,
        "strict_gap": positive_min - negative_max,
        "separable_by_threshold": bool(positive_min > negative_max and invalid_base == 0),
        "invalid_inside_base_cells": invalid_base,
        "kernel_frobenius_vs_radius1": float(np.linalg.norm(
            kernel - np.fromfunction(lambda i, j: (np.abs(i - j) <= 1).astype(float), (30, 30))
        )),
    }
    (HERE / "rank7_ratio.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

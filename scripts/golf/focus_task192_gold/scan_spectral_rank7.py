#!/usr/bin/env python3
"""Scan analytic signed rank-7 kernels for task192's generator support."""

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


def generate_dataset(count: int, seed: int) -> tuple[np.ndarray, ...]:
    generator = importlib.import_module("task_7e0986d6")
    random.seed(seed)
    inside = np.zeros((count, 30, 30), dtype=np.float32)
    nonzero = np.zeros_like(inside)
    selected = np.zeros_like(inside)
    keep = np.zeros_like(inside, dtype=np.bool_)
    for index in range(count):
        example = generator.generate()
        grid = np.asarray(example["input"], dtype=np.int8)
        expected = np.asarray(example["output"], dtype=np.int8)
        height, width = grid.shape
        color = max(range(1, 10), key=grid.reshape(-1).tolist().count)
        inside[index, :height, :width] = 1
        nonzero[index, :height, :width] = grid != 0
        selected[index, :height, :width] = grid == color
        keep[index, :height, :width] = expected == color
        if (index + 1) % 250 == 0:
            print(json.dumps({"generated": index + 1}), flush=True)
    return inside, nonzero, selected, keep


def best_threshold(ratio: np.ndarray, valid: np.ndarray, keep: np.ndarray) -> tuple[float, int]:
    values = ratio[valid].astype(np.float64)
    labels = keep[valid]
    order = np.argsort(values, kind="stable")
    values, labels = values[order], labels[order]
    positive_total = int(labels.sum())
    # threshold below the minimum predicts every cell positive.
    best_correct = positive_total
    best_value = float(np.nextafter(values[0], -np.inf))
    negative_seen = 0
    positive_seen = 0
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[end] == values[index]:
            end += 1
        chunk = labels[index:end]
        positive_seen += int(chunk.sum())
        negative_seen += int(len(chunk) - chunk.sum())
        correct = negative_seen + positive_total - positive_seen
        if correct > best_correct:
            best_correct = correct
            best_value = float(values[index])
        index = end
    return best_value, len(values) - best_correct


def evaluate(name: str, kernel: np.ndarray, inside: np.ndarray, nonzero: np.ndarray,
             selected: np.ndarray, keep: np.ndarray) -> dict[str, object]:
    h0 = inside @ kernel
    v0 = np.matmul(kernel, inside)
    h1 = selected @ kernel
    v1 = np.matmul(kernel, selected)
    base = inside * h0 * v0
    predicate = nonzero * h1 * v1
    logical = inside > 0
    safe = logical & (base > 1e-9)
    ratio = np.zeros_like(base)
    ratio[safe] = predicate[safe] / base[safe]
    threshold, cell_errors = best_threshold(ratio, safe, keep)
    predicted = safe & nonzero.astype(bool) & (ratio > threshold)
    differing = np.count_nonzero(predicted != keep, axis=(1, 2))
    positives = safe & keep
    negatives = safe & ~keep
    return {
        "name": name,
        "threshold": threshold,
        "cell_errors_at_best_threshold": cell_errors + int(np.count_nonzero(logical & ~safe)),
        "case_errors_at_best_threshold": int(np.count_nonzero(differing)),
        "positive_min": float(ratio[positives].min()),
        "negative_max": float(ratio[negatives].max()),
        "gap": float(ratio[positives].min() - ratio[negatives].max()),
        "invalid_base_cells": int(np.count_nonzero(logical & ~safe)),
        "kernel_frobenius_vs_radius1": float(np.linalg.norm(
            kernel - np.fromfunction(lambda i, j: (np.abs(i - j) <= 1).astype(float), (30, 30))
        )),
    }


def spectral(diagonal: float, positive: int, negative: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.diag(np.full(30, diagonal, dtype=np.float64))
    target += np.diag(np.ones(29), 1) + np.diag(np.ones(29), -1)
    values, vectors = np.linalg.eigh(target)
    pos = [index for index in np.argsort(values)[::-1] if values[index] > 0][:positive]
    neg = [index for index in np.argsort(values) if values[index] < 0][:negative]
    chosen = pos + neg
    factor = vectors[:, chosen] * np.sqrt(np.abs(values[chosen]))
    signs = np.sign(values[chosen])
    return factor.astype(np.float32), signs.astype(np.float32), (factor * signs) @ factor.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=192_427_001)
    args = parser.parse_args()
    inside, nonzero, selected, keep = generate_dataset(args.count, args.seed)
    rows = []
    factors: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(ROOT / "submission_base_8014.69.zip") as archive:
        current = onnx.load_model_from_string(archive.read("task192.onnx"))
    arrays = {item.name: numpy_helper.to_array(item) for item in current.graph.initializer}
    current_factor = arrays["eig_factor"].astype(np.float64)
    rows.append(evaluate("authority_psd7", current_factor @ current_factor.T,
                         inside, nonzero, selected, keep))
    for diagonal in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        for negative in range(0, 8):
            positive = 7 - negative
            factor, signs, kernel = spectral(diagonal, positive, negative)
            name = f"spectral_d{str(diagonal).replace('.', 'p')}_p{positive}n{negative}"
            row = evaluate(name, kernel, inside, nonzero, selected, keep)
            rows.append(row)
            factors[name] = {"factor": factor.tolist(), "signs": signs.tolist()}
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: (
        int(row["case_errors_at_best_threshold"]),
        int(row["cell_errors_at_best_threshold"]),
        -float(row["gap"]),
    ))
    result = {
        "seed": args.seed, "cases": args.count, "rows": rows,
        "best_factor": factors.get(rows[0]["name"]),
    }
    (HERE / "spectral_rank7_scan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"best": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()

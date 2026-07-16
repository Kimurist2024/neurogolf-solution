#!/usr/bin/env python3
"""Dual-ORT known and fresh-generator audit for the lane C29 task315 candidate."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = HERE / "task315_tied_color_factor.onnx"
BASELINE = HERE / "baseline" / "task315.onnx"
OUT = HERE / "task315_tied_color_factor_fresh5000.json"
FRESH_CASES = 5000
SEED = 315_290_071


def encoded(grid: np.ndarray) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            value[0, int(grid[row, col]), row, col] = 1.0
    return value


def truth(grid: np.ndarray) -> np.ndarray:
    numeric = np.zeros((9, 9), dtype=np.int64)
    for gate_row in range(3):
        for source_row in range(3):
            for gate_col in range(3):
                for source_col in range(3):
                    numeric[3 * gate_row + source_row, 3 * gate_col + source_col] = (
                        int(grid[source_row, source_col])
                        * int(grid[gate_row, gate_col] > 1)
                    )
    value = np.zeros((1, 10, 30, 30), dtype=bool)
    for row in range(9):
        for col in range(9):
            value[0, int(numeric[row, col]), row, col] = True
    return value


def options(level: ort.GraphOptimizationLevel) -> ort.SessionOptions:
    result = ort.SessionOptions()
    result.graph_optimization_level = level
    result.intra_op_num_threads = 1
    result.inter_op_num_threads = 1
    result.log_severity_level = 4
    return result


def session_audit(level: ort.GraphOptimizationLevel, known: list[np.ndarray], fresh: list[np.ndarray]) -> dict[str, Any]:
    session = ort.InferenceSession(
        str(CANDIDATE), options(level), providers=["CPUExecutionProvider"]
    )
    result: dict[str, Any] = {
        "mode": str(level),
        "known": {"right": 0, "wrong": 0, "errors": 0},
        "fresh": {"right": 0, "wrong": 0, "errors": 0},
        "min_positive": None,
        "max_nonpositive": None,
        "nonfinite_values": 0,
    }
    started = time.time()
    for label, grids in (("known", known), ("fresh", fresh)):
        for index, grid in enumerate(grids):
            expected = truth(grid)
            try:
                raw = session.run(["output"], {"input": encoded(grid)})[0]
                result["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(raw)))
                predicted = raw > 0.0
                if np.array_equal(predicted, expected):
                    result[label]["right"] += 1
                else:
                    result[label]["wrong"] += 1
                    result[label].setdefault(
                        "first_failure",
                        {
                            "index": index,
                            "different_cells": int(np.count_nonzero(predicted != expected)),
                            "grid": grid.tolist(),
                        },
                    )
                positive = float(raw[expected].min())
                nonpositive = float(raw[~expected].max())
                current_min = result["min_positive"]
                current_max = result["max_nonpositive"]
                result["min_positive"] = positive if current_min is None else min(current_min, positive)
                result["max_nonpositive"] = nonpositive if current_max is None else max(current_max, nonpositive)
            except Exception as exc:  # fail closed
                result[label]["errors"] += 1
                result[label].setdefault("first_error", f"{type(exc).__name__}: {exc}")
            if label == "fresh" and (index + 1) % 500 == 0:
                print(level, "fresh", index + 1, flush=True)
    result["elapsed_seconds"] = time.time() - started
    return result


def main() -> None:
    data = json.loads((ROOT / "inputs" / "neurogolf-2026" / "task315.json").read_text())
    known = [np.asarray(pair["input"], dtype=np.int64) for subset in ("train", "test", "arc-gen") for pair in data[subset]]
    if any(truth(grid).shape != (1, 10, 30, 30) for grid in known):
        raise RuntimeError("truth encoder failure")
    rng = np.random.default_rng(SEED)
    fresh = [rng.integers(0, 3, size=(3, 3), dtype=np.int64) for _ in range(FRESH_CASES)]

    model = onnx.load(CANDIDATE)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    base = onnx.load(BASELINE)
    payload: dict[str, Any] = {
        "task": 315,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "candidate_cost": 124,
        "baseline_cost": 128,
        "known_cases": len(known),
        "fresh_cases": FRESH_CASES,
        "fresh_seed": SEED,
        "checker_full": True,
        "strict_shape_inference": True,
        "declared_output_shape": [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim],
        "inferred_output_shape": [d.dim_value for d in inferred.graph.output[0].type.tensor_type.shape.dim],
        "candidate_einsum_operands": len(model.graph.node[0].input),
        "baseline_einsum_operands": len(base.graph.node[0].input),
        "modes": [],
    }
    for level in (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    ):
        payload["modes"].append(session_audit(level, known, fresh))
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

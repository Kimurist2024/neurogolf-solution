#!/usr/bin/env python3
"""Reproduce the task007 cost-68 known failure in both ORT modes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task007_r01_static68.onnx"


def make_input(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rr, cc = np.indices(values.shape)
    # Background is one-hot inside the true grid. Outside remains all-zero.
    result[0, values, rr, cc] = 1.0
    return result


def expected_onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros((10, values.shape[0], values.shape[1]), dtype=np.bool_)
    rr, cc = np.indices(values.shape)
    result[values, rr, cc] = True
    return result


def verify(disable_all: bool) -> dict[str, object]:
    options = ort.SessionOptions()
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(CANDIDATE), sess_options=options, providers=["CPUExecutionProvider"]
    )
    payload = json.loads((ROOT / "inputs/neurogolf-2026/task007.json").read_text())
    right = wrong = errors = 0
    failures = []
    near_margin = 0
    for split, examples in payload.items():
        if not isinstance(examples, list):
            continue
        for index, example in enumerate(examples):
            try:
                raw = session.run(None, {"input": make_input(example["input"])})[0]
                height = len(example["output"])
                width = len(example["output"][0])
                actual = raw[0, :, :height, :width] > 0
                expected = expected_onehot(example["output"])
                near_margin += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                if np.array_equal(actual, expected):
                    right += 1
                else:
                    wrong += 1
                    failures.append(
                        {
                            "split": split,
                            "index": index,
                            "actual_positive_count": int(actual.sum()),
                            "expected_positive_count": int(expected.sum()),
                            "different_cells": int(np.count_nonzero(actual != expected)),
                        }
                    )
            except Exception as exc:
                errors += 1
                failures.append(
                    {"split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
                )
    return {
        "disable_all": disable_all,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": right + wrong + errors,
        "near_margin_count": near_margin,
        "failures": failures,
    }


def main() -> None:
    data = CANDIDATE.read_bytes()
    result = {
        "task": 7,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "actual_cost": 68,
        "input_encoding": "channel 0 is one-hot for in-grid background; outside is all-zero",
        "default": verify(False),
        "disable_all": verify(True),
        "verdict": "REJECT_KNOWN_NOT_100",
    }
    assert result["default"]["right"] == 260
    assert result["disable_all"]["right"] == 260
    (HERE / "evidence/task007_cost68_known_dual.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

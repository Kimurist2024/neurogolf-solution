#!/usr/bin/env python3
"""Exhaustive task323 audit for the robust cost-104 reparameterization."""

from __future__ import annotations

import copy
import importlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.16.zip"
CANDIDATE = HERE / "task323_cost104_robust_u10.onnx"

import sys

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)
from scripts.lib import scoring  # noqa: E402


def make_cases() -> list[tuple[np.ndarray, np.ndarray]]:
    generator = importlib.import_module("task_d06dbe63")
    cases = []
    for row in range(13):
        for col in range(13):
            benchmark = scoring.convert_to_numpy(generator.generate(row=row, col=col))
            if benchmark is None:
                raise RuntimeError(f"generator conversion failed at {row},{col}")
            cases.append((benchmark["input"], benchmark["output"] > 0))
    return cases


def exhaustive(model: onnx.ModelProto, disable: bool, threads: int) -> dict[str, object]:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    session = ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    right = wrong = errors = nonfinite = 0
    min_positive = np.inf
    max_in_grid_off = -np.inf
    min_nonzero_abs = np.inf
    for inputs, expected in make_cases():
        try:
            raw = session.run(["output"], {"input": inputs})[0]
            mask = raw > 0
            if np.array_equal(mask, expected):
                right += 1
            else:
                wrong += 1
            nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
            min_positive = min(min_positive, float(np.min(raw[expected])))
            in_grid = np.zeros_like(expected)
            in_grid[:, :, :13, :13] = True
            off = raw[(~expected) & in_grid]
            max_in_grid_off = max(max_in_grid_off, float(np.max(off)))
            nz = raw[raw != 0]
            if nz.size:
                min_nonzero_abs = min(min_nonzero_abs, float(np.min(np.abs(nz))))
        except Exception:
            errors += 1
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total_generator_support": 169,
        "nonfinite_cells": nonfinite,
        "min_positive": min_positive,
        "max_in_grid_off": max_in_grid_off,
        "min_nonzero_abs": min_nonzero_abs,
        "perfect": right == 169 and wrong == errors == nonfinite == 0,
    }


def official(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix="task323_official_", dir="/tmp") as temp:
        return scoring.score_and_verify(model, 323, temp, label=label)


def main() -> None:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline = onnx.load_model_from_string(archive.read("task323.onnx"))
    candidate = onnx.load(CANDIDATE)
    result = {
        "baseline_zip": BASE_ZIP.name,
        "candidate": CANDIDATE.name,
        "generator_support_proof": {
            "size_default": 13,
            "row_domain": [0, 12],
            "col_domain": [0, 12],
            "complete_cases": 169,
        },
        "baseline_official": official(copy.deepcopy(baseline), "base"),
        "candidate_official": official(copy.deepcopy(candidate), "candidate"),
        "known_dual": {
            "disable_all": run_known(copy.deepcopy(candidate), 323, True),
            "default": run_known(copy.deepcopy(candidate), 323, False),
        },
        "structure": structure(copy.deepcopy(candidate), 323),
        "exhaustive": {
            "disable_threads1": exhaustive(copy.deepcopy(candidate), True, 1),
            "disable_threads4": exhaustive(copy.deepcopy(candidate), True, 4),
            "default_threads1": exhaustive(copy.deepcopy(candidate), False, 1),
            "default_threads4": exhaustive(copy.deepcopy(candidate), False, 4),
        },
    }
    (HERE / "task323_robust_audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Known four-config raw authority audit for the task205 fusion probe."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
CANDIDATE = HERE / "candidates/task205_reduce_scalar_fusion.onnx"
CONFIGS = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, level: Any, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def io_signature(model: onnx.ModelProto) -> dict[str, Any]:
    def values(items: Any) -> list[dict[str, Any]]:
        result = []
        for value in items:
            tensor = value.type.tensor_type
            result.append(
                {
                    "name": value.name,
                    "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                    "shape": [int(dim.dim_value) for dim in tensor.shape.dim],
                }
            )
        return result

    return {"inputs": values(model.graph.input), "outputs": values(model.graph.output)}


def known_cases() -> list[tuple[str, np.ndarray, np.ndarray]]:
    result = []
    for split, items in scoring.load_examples(205).items():
        for index, raw in enumerate(items):
            converted = scoring.convert_to_numpy(raw)
            if converted is None:
                raise RuntimeError(f"conversion failed for {split}[{index}]")
            result.append((f"{split}[{index}]", converted["input"], converted["output"]))
    return result


def run_config(
    authority: onnx.ModelProto,
    candidate: onnx.ModelProto,
    cases: list[tuple[str, np.ndarray, np.ndarray]],
    config: tuple[str, Any, int],
) -> dict[str, Any]:
    label, level, threads = config
    base_session = make_session(authority, level, threads)
    candidate_session = make_session(candidate, level, threads)
    row = {
        "config": label,
        "total": len(cases),
        "raw_equal_authority": 0,
        "threshold_equal_authority": 0,
        "shape_equal_authority": 0,
        "authority_right": 0,
        "candidate_right": 0,
        "authority_errors": 0,
        "candidate_errors": 0,
        "authority_nonfinite": 0,
        "candidate_nonfinite": 0,
        "authority_near_positive_0_0.25": 0,
        "candidate_near_positive_0_0.25": 0,
        "first_failure": None,
    }
    for case_id, x, expected in cases:
        try:
            base = base_session.run(["output"], {"input": x})[0]
        except Exception as exc:
            row["authority_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "case": case_id,
                "side": "authority",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        try:
            got = candidate_session.run(["output"], {"input": x})[0]
        except Exception as exc:
            row["candidate_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "case": case_id,
                "side": "candidate",
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        base_truth = base > 0
        got_truth = got > 0
        row["raw_equal_authority"] += int(np.array_equal(base, got))
        row["threshold_equal_authority"] += int(np.array_equal(base_truth, got_truth))
        row["shape_equal_authority"] += int(base.shape == got.shape)
        row["authority_right"] += int(np.array_equal(base_truth, expected))
        row["candidate_right"] += int(np.array_equal(got_truth, expected))
        row["authority_nonfinite"] += int(np.count_nonzero(~np.isfinite(base)))
        row["candidate_nonfinite"] += int(np.count_nonzero(~np.isfinite(got)))
        row["authority_near_positive_0_0.25"] += int(
            np.count_nonzero((base > 0) & (base < 0.25))
        )
        row["candidate_near_positive_0_0.25"] += int(
            np.count_nonzero((got > 0) & (got < 0.25))
        )
        if not np.array_equal(base, got) and row["first_failure"] is None:
            diff = np.argwhere(base != got)
            row["first_failure"] = {
                "case": case_id,
                "side": "raw_difference",
                "different_cells": int(diff.shape[0]),
                "first_index": diff[0].tolist() if diff.size else None,
                "max_abs": float(np.max(np.abs(base.astype(np.float64) - got.astype(np.float64)))),
            }
    row["pass"] = (
        row["raw_equal_authority"] == row["total"]
        and row["threshold_equal_authority"] == row["total"]
        and row["shape_equal_authority"] == row["total"]
        and row["authority_errors"] == row["candidate_errors"] == 0
        and row["authority_nonfinite"] == row["candidate_nonfinite"] == 0
    )
    return row


def main() -> None:
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task205.onnx")
    candidate_bytes = CANDIDATE.read_bytes()
    authority = onnx.load_from_string(authority_bytes)
    candidate = onnx.load_from_string(candidate_bytes)
    onnx.checker.check_model(copy.deepcopy(candidate), full_check=True)
    onnx.shape_inference.infer_shapes(copy.deepcopy(candidate), strict_mode=True, data_prop=True)
    cases = known_cases()
    runs = []
    for config in CONFIGS:
        row = run_config(authority, candidate, cases, config)
        runs.append(row)
        print(row["config"], row["raw_equal_authority"], "/", row["total"], flush=True)
    report = {
        "task": 205,
        "authority_member_sha256": hashlib.sha256(authority_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "known_cases": len(cases),
        "full_check": True,
        "strict_shape_inference_data_prop": True,
        "authority_io": io_signature(authority),
        "candidate_io": io_signature(candidate),
        "io_signature_equal": io_signature(authority) == io_signature(candidate),
        "authority_nodes": len(authority.graph.node),
        "candidate_nodes": len(candidate.graph.node),
        "fusion": "ReduceSum(row_mask) + Mul(rowpow_thr) -> Einsum('ab,->')",
        "runs": runs,
        "known_pass": all(row["pass"] for row in runs),
        "fresh": {
            "performed": False,
            "reason": "Parent coordination: another lane is already performing the all-input/fresh equivalence proof for this identical cost-1038 task205 candidate.",
        },
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("KNOWN_PASS", report["known_pass"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed structural and known-runtime audit for isolated task222 probes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


AUTHORITY = HERE / "authority" / "task222.onnx"
EVIDENCE = HERE / "evidence"
CONFIGS = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("default_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("default_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def params(model: onnx.ModelProto) -> int | None:
    return scoring.calculate_params(model)


def static_gate(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    row: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "params": params(model),
        "nodes": len(model.graph.node),
        "dense_initializers": len(model.graph.initializer),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "ops": dict(Counter(node.op_type for node in model.graph.node)),
        "functions": len(model.functions),
    }
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # fail-closed evidence
        row["full_check"] = False
        row["full_check_error"] = f"{type(exc).__name__}: {exc}"
    try:
        onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["strict_data_prop"] = True
    except Exception as exc:
        row["strict_data_prop"] = False
        row["strict_data_prop_error"] = f"{type(exc).__name__}: {exc}"
    return row


def session(model: onnx.ModelProto, level: Any, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known_cases() -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    rows = []
    examples = scoring.load_examples(222)
    for split in ("train", "test", "arc-gen"):
        for index, raw in enumerate(examples[split]):
            converted = scoring.convert_to_numpy(raw)
            assert converted is not None
            rows.append((split, index, converted["input"], converted["output"]))
    return rows


def runtime_gate(
    model: onnx.ModelProto,
    authority: onnx.ModelProto,
    cases: list[tuple[str, int, np.ndarray, np.ndarray]],
    config: tuple[str, Any, int],
) -> dict[str, Any]:
    label, level, threads = config
    row: dict[str, Any] = {
        "config": label,
        "total": len(cases),
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "raw_equal_authority": 0,
        "threshold_equal_authority": 0,
        "nonfinite": 0,
        "near_positive": 0,
        "output_shapes": [],
        "first_wrong": None,
        "first_raw_difference": None,
    }
    try:
        candidate_session = session(model, level, threads)
        authority_session = session(authority, level, threads)
    except Exception as exc:
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["errors"] = len(cases)
        return row
    shapes = set()
    for split, index, x, expected in cases:
        try:
            got = candidate_session.run(["output"], {"input": x})[0]
            base = authority_session.run(["output"], {"input": x})[0]
        except Exception as exc:
            row["errors"] += 1
            row.setdefault("first_error", f"{split}[{index}]: {type(exc).__name__}: {exc}")
            continue
        shapes.add(tuple(int(v) for v in got.shape))
        threshold = got > 0.0
        base_threshold = base > 0.0
        if np.array_equal(threshold, expected):
            row["right"] += 1
        else:
            row["wrong"] += 1
            if row["first_wrong"] is None:
                diff = np.argwhere(threshold != expected)
                row["first_wrong"] = {
                    "split": split,
                    "index": index,
                    "different_cells": int(diff.shape[0]),
                    "first_index": diff[0].tolist() if diff.size else None,
                }
        if np.array_equal(got, base):
            row["raw_equal_authority"] += 1
        elif row["first_raw_difference"] is None:
            diff = np.argwhere(got != base)
            row["first_raw_difference"] = {
                "split": split,
                "index": index,
                "different_cells": int(diff.shape[0]),
                "max_abs": float(np.max(np.abs(got - base))),
            }
        if np.array_equal(threshold, base_threshold):
            row["threshold_equal_authority"] += 1
        row["nonfinite"] += int(np.count_nonzero(~np.isfinite(got)))
        row["near_positive"] += int(np.count_nonzero((got > 0.0) & (got < 0.25)))
    row["output_shapes"] = [list(shape) for shape in sorted(shapes)]
    return row


def preprojection_diagnostic(
    authority: onnx.ModelProto,
    no_p: onnx.ModelProto,
    cases: list[tuple[str, int, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    base_session = session(authority, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1)
    pre_session = session(no_p, ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1)
    row: dict[str, Any] = {
        "formula": "out0=10*(s0-sum(s1..s9)); out_o=10*s_o for o>0",
        "cases": len(cases),
        "formula_raw_equal": 0,
        "background_sign_equivalent_cases": 0,
        "background_sign_difference_cells": 0,
        "preprojection_gold_right": 0,
        "first_background_counterexample": None,
    }
    for split, index, x, expected in cases:
        pre = pre_session.run(["output"], {"input": x})[0]
        final = base_session.run(["output"], {"input": x})[0]
        reconstructed = np.empty_like(pre)
        reconstructed[:, 0] = 10.0 * (
            pre[:, 0] - np.sum(pre[:, 1:], axis=1)
        )
        reconstructed[:, 1:] = 10.0 * pre[:, 1:]
        if np.array_equal(reconstructed, final):
            row["formula_raw_equal"] += 1
        old_bg = final[:, 0] > 0.0
        new_bg = pre[:, 0] > 0.0
        diff = np.argwhere(old_bg != new_bg)
        if diff.size == 0:
            row["background_sign_equivalent_cases"] += 1
        else:
            row["background_sign_difference_cells"] += int(diff.shape[0])
            if row["first_background_counterexample"] is None:
                b, r, c = (int(v) for v in diff[0])
                row["first_background_counterexample"] = {
                    "split": split,
                    "index": index,
                    "r": r,
                    "c": c,
                    "s0": float(pre[b, 0, r, c]),
                    "sum_nonbackground": float(np.sum(pre[b, 1:, r, c])),
                    "old_background": bool(old_bg[b, r, c]),
                    "new_background": bool(new_bg[b, r, c]),
                }
        if np.array_equal(pre > 0.0, expected):
            row["preprojection_gold_right"] += 1
    return row


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    paths = [AUTHORITY] + sorted((HERE / "candidates").glob("*.onnx"))
    authority = onnx.load(AUTHORITY)
    cases = known_cases()
    report: dict[str, Any] = {
        "task": 222,
        "authority_sha256": digest(AUTHORITY),
        "known_cases": len(cases),
        "models": [],
    }
    for path in paths:
        static = static_gate(path)
        model = onnx.load(path)
        runs = []
        if static["full_check"] and static["strict_data_prop"]:
            for config in CONFIGS:
                runs.append(runtime_gate(model, authority, cases, config))
        static["known_four_config"] = runs
        report["models"].append(static)
        print(path.name, static["params"], [(r["config"], r["right"]) for r in runs], flush=True)
    no_p = onnx.load(HERE / "candidates" / "task222_no_P_cost280.onnx")
    report["preprojection"] = preprojection_diagnostic(authority, no_p, cases)
    (EVIDENCE / "candidate_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

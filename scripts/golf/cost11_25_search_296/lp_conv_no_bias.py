#!/usr/bin/env python3
"""Search finite no-bias ConvTranspose kernels by exact sign feasibility."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
TASKS = (314, 322)


def load_support():
    spec = importlib.util.spec_from_file_location("lp_conv_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_support()


def session(data: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])


def basis_models(base: onnx.ModelProto) -> tuple[list[ort.InferenceSession], tuple[int, ...]]:
    model = copy.deepcopy(base)
    node = model.graph.node[0]
    if node.op_type != "ConvTranspose" or len(node.input) != 3:
        raise RuntimeError("expected one biased ConvTranspose")
    del node.input[2]
    bias_name = base.graph.node[0].input[2]
    keep = [item for item in model.graph.initializer if item.name != bias_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    kernel = model.graph.initializer[0]
    shape = tuple(int(dim) for dim in kernel.dims)
    result = []
    for index in range(int(np.prod(shape))):
        candidate = copy.deepcopy(model)
        values = np.zeros(shape, dtype=np.float32)
        values.reshape(-1)[index] = 1.0
        candidate.graph.initializer[0].CopyFrom(
            numpy_helper.from_array(values, candidate.graph.initializer[0].name)
        )
        result.append(session(candidate.SerializeToString()))
    return result, shape


def collect_constraints(task: int, base: onnx.ModelProto) -> dict[str, object]:
    sessions, shape = basis_models(base)
    cases, counts = SUPPORT.known_cases(task)
    positive: set[tuple[float, ...]] = set()
    nonpositive: set[tuple[float, ...]] = set()
    for case in cases:
        converted = SUPPORT.scoring.convert_to_numpy(case)
        if converted is None:
            raise RuntimeError("known conversion failed")
        inp = converted["input"]
        expected = converted["output"].astype(bool).reshape(-1)
        columns = [run.run(["output"], {"input": inp})[0].reshape(-1) for run in sessions]
        features = np.stack(columns, axis=1)
        for row in np.unique(features[expected], axis=0):
            positive.add(tuple(float(value) for value in row))
        for row in np.unique(features[~expected], axis=0):
            nonpositive.add(tuple(float(value) for value in row))
    conflict = positive & nonpositive
    p = len(sessions)
    result: dict[str, object] = {
        "task": task,
        "kernel_shape": list(shape),
        "params_without_bias": p,
        "known_counts": counts,
        "unique_positive_features": len(positive),
        "unique_nonpositive_features": len(nonpositive),
        "label_conflicts": len(conflict),
    }
    if conflict:
        result["decision"] = "INFEASIBLE_IDENTICAL_FEATURE_CONFLICT"
        result["first_conflict"] = list(next(iter(conflict)))
        return result
    pos = np.asarray(sorted(positive), dtype=np.float64).reshape(-1, p)
    neg = np.asarray(sorted(nonpositive), dtype=np.float64).reshape(-1, p)
    a_ub = np.concatenate((-pos, neg), axis=0)
    b_ub = np.concatenate((-np.ones(len(pos)), np.zeros(len(neg))), axis=0)
    solved = linprog(
        np.zeros(p), A_ub=a_ub, b_ub=b_ub,
        bounds=[(-1_000_000.0, 1_000_000.0)] * p,
        method="highs",
    )
    result["linprog_success"] = bool(solved.success)
    result["linprog_status"] = int(solved.status)
    result["linprog_message"] = solved.message
    if not solved.success:
        result["decision"] = "INFEASIBLE_LINEAR_SIGN_SYSTEM"
        return result
    weights = np.asarray(solved.x, dtype=np.float32)
    result["weights"] = weights.reshape(shape).tolist()
    result["positive_margin_min"] = float(np.min(pos @ weights))
    result["nonpositive_max"] = float(np.max(neg @ weights))
    result["decision"] = "FEASIBLE_PRE_RUNTIME"
    return result


def main() -> int:
    rows = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            base = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
            row = collect_constraints(task, base)
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "method": "known-complete exact sign LP after removing ConvTranspose bias",
        "results": rows,
    }
    (HERE / "lp_conv_no_bias.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

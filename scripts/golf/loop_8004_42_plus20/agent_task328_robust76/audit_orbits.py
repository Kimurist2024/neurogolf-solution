#!/usr/bin/env python3
"""Audit all task328 generator color-orbit representatives in four ORT configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
CORNERS = lambda size: ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))


def make_session(path: Path, disabled: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def empty_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "true_below_0_25": 0,
        "true_below_1": 0,
        "false_positive": 0,
        "min_true": None,
        "max_false": None,
        "max_abs": 0.0,
        "first_failure": None,
    }


def update(stats: dict[str, Any], raw: np.ndarray, expected: np.ndarray, case: dict[str, Any]) -> None:
    raw = np.asarray(raw)
    finite = np.isfinite(raw)
    stats["nonfinite_values"] += int(raw.size - np.count_nonzero(finite))
    safe = raw[finite]
    if safe.size:
        stats["max_abs"] = max(float(stats["max_abs"]), float(np.abs(safe).max(initial=0.0)))
    true_raw = raw[expected]
    false_raw = raw[~expected]
    stats["true_below_0_25"] += int(np.count_nonzero(true_raw < 0.25))
    stats["true_below_1"] += int(np.count_nonzero(true_raw < 1.0))
    stats["false_positive"] += int(np.count_nonzero(false_raw > 0))
    if true_raw.size and np.isfinite(true_raw).all():
        value = float(true_raw.min())
        stats["min_true"] = value if stats["min_true"] is None else min(float(stats["min_true"]), value)
    if false_raw.size and np.isfinite(false_raw).all():
        value = float(false_raw.max())
        stats["max_false"] = value if stats["max_false"] is None else max(float(stats["max_false"]), value)
    if np.array_equal(raw > 0, expected):
        stats["right"] += 1
    else:
        stats["wrong"] += 1
        if stats["first_failure"] is None:
            stats["first_failure"] = {
                **case,
                "different_cells": int(np.count_nonzero((raw > 0) != expected)),
            }


def canonical_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    generator = importlib.import_module("task_d22278a0")
    cases = []
    for size in range(6, 19):
        for count in range(2, 5):
            for selected in itertools.combinations(CORNERS(size), count):
                rows, cols = zip(*selected)
                colors = tuple(range(1, count + 1))
                example = generator.generate(size=size, rows=rows, cols=cols, colors=colors)
                cases.append((example, {"size": size, "corners": [list(item) for item in selected], "colors": list(colors)}))
    return cases


def audit_cases(path: Path) -> dict[str, Any]:
    cases = canonical_cases()
    sessions = {label: make_session(path, disabled, threads) for disabled, threads, label in CONFIGS}
    stats = {label: empty_stats() for _, _, label in CONFIGS}
    for index, (example, meta) in enumerate(cases, start=1):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"canonical generator case is not convertible: {meta}")
        expected = benchmark["output"].astype(bool)
        for label, session in sessions.items():
            try:
                raw = session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
                update(stats[label], raw, expected, {"orbit_case": index, **meta})
            except Exception as exc:  # noqa: BLE001
                stats[label]["runtime_errors"] += 1
                if stats[label]["first_failure"] is None:
                    stats[label]["first_failure"] = {"orbit_case": index, **meta, "error": f"{type(exc).__name__}: {exc}"}
        if index % 25 == 0:
            checkpoint = {
                "model": str(path.resolve().relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "completed_orbits": index,
                "total_orbits": len(cases),
                "configs": stats,
            }
            (HERE / "orbit_checkpoint.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
            print(f"orbit {index}/{len(cases)}", flush=True)
    for row in stats.values():
        row["perfect_sign"] = row["right"] == len(cases) and row["wrong"] == row["runtime_errors"] == 0
        row["margin_0_25"] = row["true_below_0_25"] == row["false_positive"] == row["nonfinite_values"] == 0
        row["margin_1"] = row["true_below_1"] == row["false_positive"] == row["nonfinite_values"] == 0
    return {
        "color_orbit_representatives": len(cases),
        "full_support_count": 13 * (6 * 9 * 8 + 4 * 9 * 8 * 7 + 9 * 8 * 7 * 6),
        "configs": stats,
        "all_configs_sign_perfect": all(row["perfect_sign"] for row in stats.values()),
        "all_configs_margin_0_25": all(row["margin_0_25"] for row in stats.values()),
        "all_configs_margin_1": all(row["margin_1"] for row in stats.values()),
    }


def audit_known(path: Path) -> dict[str, Any]:
    examples = scoring.load_examples(328)
    sessions = {label: make_session(path, disabled, threads) for disabled, threads, label in CONFIGS}
    stats = {label: empty_stats() for _, _, label in CONFIGS}
    total = 0
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[split]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            total += 1
            expected = benchmark["output"].astype(bool)
            for label, session in sessions.items():
                try:
                    raw = session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    update(stats[label], raw, expected, {"split": split, "index": index})
                except Exception as exc:  # noqa: BLE001
                    stats[label]["runtime_errors"] += 1
                    if stats[label]["first_failure"] is None:
                        stats[label]["first_failure"] = {"split": split, "index": index, "error": f"{type(exc).__name__}: {exc}"}
    for row in stats.values():
        row["perfect_sign"] = row["right"] == total and row["wrong"] == row["runtime_errors"] == 0
        row["margin_0_25"] = row["true_below_0_25"] == row["false_positive"] == row["nonfinite_values"] == 0
        row["margin_1"] = row["true_below_1"] == row["false_positive"] == row["nonfinite_values"] == 0
    return {"total": total, "configs": stats}


def static_audit(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    ops = Counter(node.op_type for node in model.graph.node)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    color_axis_initializers = []
    for name, array in arrays.items():
        for axis, size in enumerate(array.shape):
            if size == 10:
                color_axis_initializers.append({"name": name, "axis": axis})
    ssel = arrays["Ssel"]
    nonzero_color_columns_identical = bool(np.all(ssel[:, 1:] == ssel[:, 1:2]))
    with tempfile.TemporaryDirectory(prefix="task328_robust76_", dir="/tmp") as workdir:
        temp = Path(workdir) / "candidate.onnx"
        temp.write_bytes(path.read_bytes())
        memory, params, cost = cost_of(str(temp))
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    nonstatic = []
    for value in values:
        if not value.type.HasField("tensor_type"):
            continue
        dims = value.type.tensor_type.shape.dim
        if any(not dim.HasField("dim_value") or int(dim.dim_value) <= 0 for dim in dims):
            nonstatic.append(value.name)
    return {
        "full_checker": True,
        "strict_data_prop": True,
        "nonstatic_tensors": nonstatic,
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import) and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "op_histogram": dict(sorted(ops.items())),
        "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
        "conv_bias_findings": check_conv_bias(model),
        "finite_initializers": all(array.dtype.kind not in "fc" or np.isfinite(array).all() for array in arrays.values()),
        "lookup_ops": {name: ops.get(name, 0) for name in ("TfIdfVectorizer", "Gather", "GatherND", "ScatterND", "Hardmax") if ops.get(name, 0)},
        "profile": {"memory": int(memory), "params": int(params), "cost": int(cost)},
        "color_axis_initializers": color_axis_initializers,
        "nonzero_color_columns_identical": nonzero_color_columns_identical,
        "color_orbit_proof": "MaxPool is channelwise; ReduceL2 is invariant under channel permutation; the only initializer with a 10-color axis is Ssel and columns 1..9 are identical; the final free output color axis is therefore equivariant under every permutation of nonzero colors.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--known", action="store_true")
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    output = {
        "model": str(args.model.resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "static": static_audit(args.model),
        "orbits": audit_cases(args.model),
    }
    if args.known:
        output["known"] = audit_known(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"sha256": output["sha256"], "profile": output["static"]["profile"], "orbits": output["orbits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

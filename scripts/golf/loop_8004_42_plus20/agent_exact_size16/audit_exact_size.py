#!/usr/bin/env python3
"""Audit exact Size folds without promoting or modifying any root artifact."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
CANDIDATES = HERE / "candidates"
CURRENT_COSTS = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
TASKS = (69, 177, 367, 387)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def known_examples(task: int) -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    examples = scoring.load_examples(task)
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def structural(model: onnx.ModelProto) -> dict[str, object]:
    row: dict[str, object] = {
        "checker_full": False,
        "strict_data_prop": False,
        "static_positive_shapes": False,
        "standard_domains": all(opset.domain in {"", "ai.onnx"} for opset in model.opset_import),
        "banned_ops": [],
        "nested_graphs": [],
        "conv_bias_ub": [],
        "errors": [],
    }
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            row["banned_ops"].append(node.op_type)  # type: ignore[union-attr]
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                row["nested_graphs"].append(node.op_type)  # type: ignore[union-attr]
    inferred = None
    try:
        onnx.checker.check_model(model, full_check=True)
        row["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"full_check: {type(exc).__name__}: {exc}")  # type: ignore[union-attr]
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["errors"].append(f"strict_data_prop: {type(exc).__name__}: {exc}")  # type: ignore[union-attr]
    if inferred is not None:
        bad = []
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            if not value.type.HasField("tensor_type") or not value.type.tensor_type.HasField("shape"):
                bad.append(value.name)
                continue
            for dim in value.type.tensor_type.shape.dim:
                if not dim.HasField("dim_value") or dim.dim_value <= 0 or dim.HasField("dim_param"):
                    bad.append(value.name)
                    break
        row["static_positive_shapes"] = not bad
        row["nonstatic"] = bad

    spec = importlib.util.spec_from_file_location("check_conv_bias", ROOT / "scripts/golf/check_conv_bias.py")
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row["conv_bias_ub"] = [list(item) for item in module.check_model(model)]
    row["pass"] = bool(
        row["checker_full"]
        and row["strict_data_prop"]
        and row["static_positive_shapes"]
        and row["standard_domains"]
        and not row["banned_ops"]
        and not row["nested_graphs"]
        and not row["conv_bias_ub"]
    )
    return row


def make_session(model: onnx.ModelProto, disable_all: bool, profiling: bool = False, prefix: str = ""):
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.enable_profiling = profiling
    if profiling:
        options.profile_file_prefix = prefix
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return sanitized, ort.InferenceSession(sanitized.SerializeToString(), options)


def known_dual(base: onnx.ModelProto, candidate: onnx.ModelProto, examples: list[dict[str, np.ndarray]]):
    result: dict[str, object] = {}
    for mode, disable_all in (("disabled", True), ("default", False)):
        row: dict[str, object] = {
            "total": len(examples),
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "raw_bitwise_equal": 0,
            "decoded_equal": 0,
            "load_error": None,
        }
        try:
            _, base_session = make_session(base, disable_all)
            _, candidate_session = make_session(candidate, disable_all)
        except Exception as exc:  # noqa: BLE001
            row["load_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = len(examples)
            result[mode] = row
            continue
        for example in examples:
            try:
                base_raw = base_session.run(["output"], {"input": example["input"]})[0]
                candidate_raw = candidate_session.run(["output"], {"input": example["input"]})[0]
            except Exception:  # noqa: BLE001
                row["runtime_errors"] = int(row["runtime_errors"]) + 1
                continue
            base_decoded = base_raw > 0
            candidate_decoded = candidate_raw > 0
            right = np.array_equal(candidate_decoded, example["output"] > 0)
            row["right"] = int(row["right"]) + int(right)
            row["wrong"] = int(row["wrong"]) + int(not right)
            row["raw_bitwise_equal"] = int(row["raw_bitwise_equal"]) + int(
                np.array_equal(base_raw, candidate_raw, equal_nan=True)
            )
            row["decoded_equal"] = int(row["decoded_equal"]) + int(
                np.array_equal(base_decoded, candidate_decoded)
            )
        result[mode] = row
    return result


def declared_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    result = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        valid = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                valid = False
                break
            dims.append(dim.dim_value)
        if valid:
            result[value.name] = tuple(dims)
    return result


def runtime_shape_audit(model: onnx.ModelProto, examples: list[dict[str, np.ndarray]]) -> dict[str, object]:
    """Compare declared and profiled runtime shapes over up to five known cases."""
    with tempfile.TemporaryDirectory(prefix="shape_profile_", dir=HERE) as directory:
        prefix = str(Path(directory) / "trace")
        try:
            sanitized, session = make_session(model, True, profiling=True, prefix=prefix)
        except Exception as exc:  # noqa: BLE001
            return {"load_error": f"{type(exc).__name__}: {exc}", "mismatches": [], "pass": False}
        for example in examples[:5]:
            try:
                session.run(["output"], {"input": example["input"]})
            except Exception as exc:  # noqa: BLE001
                return {"runtime_error": f"{type(exc).__name__}: {exc}", "mismatches": [], "pass": False}
        trace_path = Path(session.end_profiling())
        trace = json.loads(trace_path.read_text())
        declared = declared_shapes(sanitized)
        node_outputs = {node.name: list(node.output) for node in sanitized.graph.node}
        actual: dict[str, set[tuple[int, ...]]] = {}
        for event in trace:
            args = event.get("args", {})
            if event.get("cat") != "Node" or "output_type_shape" not in args:
                continue
            node_name = event.get("name", "").replace("_kernel_time", "")
            outputs = node_outputs.get(node_name, [])
            for index, shape_dict in enumerate(args["output_type_shape"]):
                if index >= len(outputs) or not shape_dict:
                    continue
                dims = tuple(next(iter(shape_dict.values())))
                actual.setdefault(outputs[index], set()).add(dims)
        mismatches = []
        for name, shapes in actual.items():
            if name not in declared:
                continue
            if any(shape != declared[name] for shape in shapes):
                mismatches.append(
                    {"name": name, "declared": list(declared[name]), "runtime": [list(s) for s in sorted(shapes)]}
                )
        return {
            "profiled_examples": min(5, len(examples)),
            "profiled_outputs": len(actual),
            "mismatches": mismatches,
            "pass": not mismatches,
        }


def official_score(model: onnx.ModelProto, task: int, label: str):
    with tempfile.TemporaryDirectory(prefix=f"score_{task:03d}_", dir=HERE) as directory:
        try:
            return scoring.score_and_verify(copy.deepcopy(model), task, directory, label=label, require_correct=True)
        except Exception:  # The official scorer's final full-check is not wrapped.
            return None


def main() -> None:
    rows = []
    current = json.loads(CURRENT_COSTS.read_text())
    current_by_task = {int(row["task"]): row for row in current["ranked"]}
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            baseline_bytes = archive.read(name)
            candidate_bytes = (CANDIDATES / name).read_bytes()
            baseline = onnx.load_model_from_string(baseline_bytes)
            candidate = onnx.load_model_from_string(candidate_bytes)
            examples = known_examples(task)
            current_row = current_by_task[task]
            base_score = {
                "memory": int(current_row["memory"]),
                "params": int(current_row["params"]),
                "cost": int(current_row["cost"]),
                "score": max(1.0, 25.0 - math.log(int(current_row["cost"]))),
                "correct": True,
                "source": str(CURRENT_COSTS.relative_to(ROOT)),
            }
            dual = known_dual(baseline, candidate, examples)
            candidate_structure = structural(candidate)
            candidate_score = (
                official_score(candidate, task, "candidate")
                if candidate_structure["pass"]
                else None
            )
            base_shape = runtime_shape_audit(baseline, examples)
            candidate_shape = runtime_shape_audit(candidate, examples)
            old_cost = None if base_score is None else int(base_score["cost"])
            theoretical_cost = None if old_cost is None else old_cost - 7
            prerequisites = bool(
                candidate_structure["pass"]
                and candidate_shape["pass"]
                and candidate_score is not None
                and all(
                    mode["load_error"] is None
                    and mode["runtime_errors"] == 0
                    and mode["right"] == mode["total"]
                    and mode["raw_bitwise_equal"] == mode["total"]
                    for mode in dual.values()
                )
            )
            rows.append(
                {
                    "task": task,
                    "baseline_sha256": sha256(baseline_bytes),
                    "candidate": str((CANDIDATES / name).relative_to(ROOT)),
                    "candidate_sha256": sha256(candidate_bytes),
                    "rewrite": next(
                        row["rewrite"]
                        for row in json.loads((HERE / "build_manifest.json").read_text())
                        if row["task"] == task
                    ),
                    "baseline_score": base_score,
                    "candidate_score": candidate_score,
                    "theoretical_cost_if_the_fold_were_scorable": theoretical_cost,
                    "theoretical_gain": (
                        None if old_cost is None else math.log(old_cost / theoretical_cost)
                    ),
                    "candidate_structure": candidate_structure,
                    "baseline_runtime_shapes": base_shape,
                    "candidate_runtime_shapes": candidate_shape,
                    "known_dual": dual,
                    "fresh": {
                        "run": False,
                        "reason": "mandatory full-check/strict/runtime/known prerequisites failed",
                    },
                    "private_zero_lineage": False,
                    "accepted": prerequisites,
                    "verdict": "ACCEPT_EXACT" if prerequisites else "REJECT",
                }
            )
            print(f"task{task:03d}: {rows[-1]['verdict']}", flush=True)
    result = {
        "complete": True,
        "baseline_zip": BASE_ZIP.name,
        "baseline_zip_sha256": sha256(BASE_ZIP.read_bytes()),
        "tasks": rows,
        "accepted": [row["task"] for row in rows if row["accepted"]],
        "projected_gain": sum(
            float(row["theoretical_gain"]) for row in rows if row["accepted"]
        ),
        "verdict": "NO_SAFE_EXACT_CANDIDATE" if not any(row["accepted"] for row in rows) else "HAS_SAFE_EXACT_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": result["accepted"], "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed audit for the task209 residual candidates."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "others/71407/task209.onnx"
CANDIDATE = HERE / "candidates/task209_inherited_roundless.onnx"
TRUTHFUL_CONTROL = HERE / "candidates/task209_decloak_unsqueeze_roundless.onnx"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (218_209_001, 218_209_019)
FRESH_PER_SEED = 1000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text(encoding="utf-8"))


def make_session(model: onnx.ModelProto, disable: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])


def cases_known() -> list[dict[str, Any]]:
    payload = scoring.load_examples(209)
    return [item for split in ("train", "test", "arc-gen") for item in payload[split]]


def cases_fresh(seed: int, count: int) -> tuple[list[dict[str, Any]], int]:
    module = importlib.import_module(f"task_{TASK_MAP['209']}")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < count and attempts < count * 10:
        attempts += 1
        example = module.generate()
        if scoring.convert_to_numpy(example) is not None:
            rows.append(example)
    if len(rows) != count:
        raise RuntimeError(f"fresh generation incomplete: {len(rows)}/{count}")
    return rows, attempts


def raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.array_equal(
            np.ascontiguousarray(left).view(np.uint8),
            np.ascontiguousarray(right).view(np.uint8),
        )
    )


def evaluate(
    base: onnx.ModelProto,
    candidate: onnx.ModelProto,
    cases: list[dict[str, Any]],
    disable: bool,
    threads: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "base_right": 0,
        "candidate_right": 0,
        "runtime_errors": 0,
        "base_nonfinite": 0,
        "candidate_nonfinite": 0,
        "first_failure": None,
    }
    try:
        sessions = {
            "base": make_session(base, disable, threads),
            "candidate": make_session(candidate, disable, threads),
        }
    except Exception as exc:  # noqa: BLE001
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["pass"] = False
        return row
    for index, example in enumerate(cases):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        row["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                value = np.asarray(
                    session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: benchmark["input"]})[0]
                )
                outputs[label] = value
                row[f"{label}_nonfinite"] += int(value.size - np.count_nonzero(np.isfinite(value)))
                row[f"{label}_right"] += int(value.shape == expected.shape and np.array_equal(value > 0, expected))
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if len(outputs) == 2:
            equal = raw_equal(outputs["base"], outputs["candidate"])
            threshold_equal = np.array_equal(outputs["base"] > 0, outputs["candidate"] > 0)
            row["raw_equal"] += int(equal)
            row["threshold_equal"] += int(threshold_equal)
            if not equal:
                row["first_failure"] = row["first_failure"] or {
                    "index": index,
                    "comparison": "candidate_vs_authority",
                }
    valid = row["valid"]
    row["candidate_accuracy"] = row["candidate_right"] / valid if valid else None
    row["base_accuracy"] = row["base_right"] / valid if valid else None
    row["pass"] = bool(
        valid > 0
        and row["raw_equal"] == valid
        and row["threshold_equal"] == valid
        and row["runtime_errors"] == 0
        and row["candidate_nonfinite"] == row["base_nonfinite"]
    )
    return row


def log_index_model(gamma: float, with_round: bool) -> onnx.ModelProto:
    nodes = [
        helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT16),
        helper.make_node("Log", ["xf"], ["log"]),
        helper.make_node("Selu", ["log"], ["scaled"], alpha=1.0, gamma=float(gamma)),
    ]
    source = "scaled"
    if with_round:
        nodes.append(helper.make_node("Round", [source], ["rounded"]))
        source = "rounded"
    nodes.append(helper.make_node("Cast", [source], ["y"], to=TensorProto.UINT8))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "task209_log_index_support",
            [helper.make_tensor_value_info("x", TensorProto.UINT32, [])],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=10,
    )


def support_audit(base: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    old_node = next(node for node in base.graph.node if list(node.output) == ["pclog2"])
    new_node = next(node for node in candidate.graph.node if list(node.output) == ["pclog2"])
    old_gamma = float(next(attr.f for attr in old_node.attribute if attr.name == "gamma"))
    new_gamma = float(next(attr.f for attr in new_node.attribute if attr.name == "gamma"))
    old = log_index_model(old_gamma, True)
    new = log_index_model(new_gamma, False)
    values = [np.uint32(1 << bit) for bit in range(32)]
    modes: dict[str, Any] = {}
    for disable, threads, label in CONFIGS:
        sessions = {
            "old": make_session(old, disable, threads),
            "new": make_session(new, disable, threads),
        }
        rows = []
        equal = 0
        for bit, value in enumerate(values):
            outputs = {
                key: int(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: np.asarray(value, dtype=np.uint32)},
                    )[0]
                )
                for key, session in sessions.items()
            }
            equal += int(outputs["old"] == outputs["new"])
            rows.append({"bit": bit, "lowbit": int(value), **outputs})
        modes[label] = {"equal": equal, "total": len(values), "pass": equal == len(values), "rows": rows}
    return {
        "old_gamma": old_gamma,
        "new_gamma": new_gamma,
        "domain": "all 32 possible nonzero uint32 lowbit powers",
        "modes": modes,
        "pass": all(item["pass"] for item in modes.values()),
    }


def runtime_shape_truth(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {item.name for item in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if not name:
                continue
            names.append(name)
            if name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    example = scoring.convert_to_numpy(scoring.load_examples(209)["train"][0])
    if example is None:
        raise RuntimeError("known example conversion failed")
    arrays = session.run(names, {session.get_inputs()[0].name: example["input"]})
    mismatches = []
    nonfinite = 0
    for name, array in zip(names, arrays):
        declared = [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in typed[name].type.tensor_type.shape.dim
        ]
        actual = list(np.asarray(array).shape)
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if np.asarray(array).dtype.kind in "fc":
            nonfinite += int(np.asarray(array).size - np.count_nonzero(np.isfinite(array)))
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches,
    }


def standard_structure(model: onnx.ModelProto) -> dict[str, Any]:
    forbidden = []
    nested = []
    for node in model.graph.node:
        if node.op_type in {"Loop", "Scan", "NonZero", "Unique", "Compress"} or "Sequence" in node.op_type:
            forbidden.append(node.op_type)
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.output[0] if node.output else node.op_type)
    initializer_sizes = [int(np.prod(item.dims, dtype=np.int64)) for item in model.graph.initializer]
    return {
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "forbidden_ops": forbidden,
        "nested_graphs": nested,
        "lookup_abuse_ops": [node.op_type for node in model.graph.node if node.op_type in {"TfIdfVectorizer", "Hardmax"}],
        "largest_initializer": max(initializer_sizes, default=0),
        "center_crop_pad_count": sum(node.op_type == "CenterCropPad" for node in model.graph.node),
        "pass": not forbidden and not nested,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    base = onnx.load(BASE)
    candidate = onnx.load(CANDIDATE)
    truthful_control = onnx.load(TRUTHFUL_CONTROL)
    known = cases_known()
    fresh_streams = []
    all_fresh: list[dict[str, Any]] = []
    for seed in FRESH_SEEDS:
        rows, attempts = cases_fresh(seed, FRESH_PER_SEED)
        fresh_streams.append({"seed": seed, "valid": len(rows), "attempts": attempts})
        all_fresh.extend(rows)
    report: dict[str, Any] = {
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "truthful_control": str(TRUTHFUL_CONTROL.relative_to(ROOT)),
        "known_count": len(known),
        "fresh_streams": fresh_streams,
        "support": support_audit(base, candidate),
        "candidate_structure": standard_structure(candidate),
        "candidate_runtime_shape_truth": runtime_shape_truth(candidate),
        "truthful_control_structure": standard_structure(truthful_control),
        "truthful_control_runtime_shape_truth": runtime_shape_truth(truthful_control),
        "known_four_configs": {},
        "fresh_four_configs": {},
    }
    for disable, threads, label in CONFIGS:
        report["known_four_configs"][label] = evaluate(base, candidate, known, disable, threads)
        print("known", label, report["known_four_configs"][label]["pass"], flush=True)
        report["fresh_four_configs"][label] = evaluate(base, candidate, all_fresh, disable, threads)
        print("fresh", label, report["fresh_four_configs"][label]["pass"], flush=True)
        (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["pass"] = bool(
        report["support"]["pass"]
        and all(item["pass"] for item in report["known_four_configs"].values())
        and all(item["pass"] for item in report["fresh_four_configs"].values())
    )
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("overall", report["pass"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

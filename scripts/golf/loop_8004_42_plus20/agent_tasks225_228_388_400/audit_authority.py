#!/usr/bin/env python3
"""Independent latest-authority SOUND audit for tasks 225/228/388/400."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402

TASKS = (225, 228, 388, 400)
HASHES = {225: "93b581b8", 228: "952a094c", 388: "f5b8619d", 400: "ff805c23"}
AUTHORITY_COSTS = {225: 333, 228: 291, 388: 305, 400: 164}
FRESH_PER_SEED = 1000
SEEDS = (2252283884001, 2252283884002)
MODES = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("default_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("default_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sakana(task: int):
    path = ROOT / "inputs" / "sakana-gcg-2025" / "raw" / f"task{task}.py"
    spec = importlib.util.spec_from_file_location(f"audit_sakana_{task}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session(path: Path, level: ort.GraphOptimizationLevel, threads: int):
    model = scoring.sanitize_model(onnx.load(path))
    if model is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    result: dict[str, Any] = {
        "sha256": digest(path),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": int(scoring.calculate_params(model) or -1),
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "banned_ops": sorted(
            node.op_type for node in model.graph.node
            if node.op_type.upper() in scoring._EXCLUDED_OP_TYPES or "Sequence" in node.op_type
        ),
        "nonstandard_domains": sorted({node.domain for node in model.graph.node if node.domain}),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graph_attributes": sum(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute
        ),
        "hardmax": sum(node.op_type == "Hardmax" for node in model.graph.node),
        "tfidf": sum(node.op_type == "TfIdfVectorizer" for node in model.graph.node),
        "max_einsum_inputs": max(
            [len(node.input) for node in model.graph.node if node.op_type == "Einsum"] or [0]
        ),
        "giant_initializers": [
            item.name for item in model.graph.initializer
            if int(np.prod(item.dims, dtype=np.int64)) >= 10000
        ],
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_checker"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_checker=False, full_checker_error=f"{type(exc).__name__}: {exc}")
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
    result["conv_bias_findings"] = []
    values = {item.name: np.asarray(numpy_helper.to_array(item)) for item in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "ConvTranspose", "QLinearConv"}:
            continue
        # Only an explicit bias input is audited. QLinearConv has no bias input.
        position = 2 if node.op_type in {"Conv", "ConvTranspose"} else None
        if position is not None and len(node.input) > position and node.input[position]:
            weight = values.get(node.input[1])
            bias = values.get(node.input[position])
            if weight is not None and bias is not None:
                expected = int(weight.shape[1] if node.op_type == "ConvTranspose" else weight.shape[0])
                if bias.size != expected:
                    result["conv_bias_findings"].append(
                        {"node": index, "bias": node.input[position], "size": int(bias.size), "expected": expected}
                    )
    return result


def rule_equal(task: int, sakana: Any, example: dict[str, Any]) -> bool:
    grid = copy.deepcopy(example["input"])
    expected = copy.deepcopy(example["output"])
    actual = sakana.p(grid)
    return [list(row) for row in actual] == [list(row) for row in expected]


def stats() -> dict[str, Any]:
    return {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0, "raw_equal_reference": 0}


def run_case(sessions: dict[str, Any], benchmark: dict[str, np.ndarray]) -> dict[str, Any]:
    expected = benchmark["output"] > 0
    outputs: dict[str, np.ndarray] = {}
    result = {name: stats() for name in sessions}
    for name, sess in sessions.items():
        try:
            raw = sess.run(["output"], {"input": benchmark["input"]})[0]
            outputs[name] = raw
            result[name]["nonfinite"] += int(not np.isfinite(raw).all())
            result[name]["right" if np.array_equal(raw > 0, expected) else "wrong"] += 1
        except Exception:  # noqa: BLE001
            result[name]["errors"] += 1
    reference = outputs.get("disable_t1")
    if reference is not None:
        for name, raw in outputs.items():
            result[name]["raw_equal_reference"] += int(np.array_equal(raw, reference, equal_nan=True))
    return result


def add_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for mode in target:
        for key in target[mode]:
            target[mode][key] += source[mode][key]


def runtime_shape_audit(path: Path, benchmark_input: np.ndarray) -> dict[str, Any]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=False, data_prop=True)
    values = [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    elem_types = {value.name: value.type.tensor_type.elem_type for value in values}
    declared = {
        value.name: [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
        for value in values
    }
    for node in model.graph.node:
        for output in node.output:
            if output and output != "output":
                model.graph.output.append(
                    helper.make_tensor_value_info(output, elem_types.get(output, TensorProto.UNDEFINED), None)
                )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    try:
        sess = ort.InferenceSession(model.SerializeToString(), options)
        arrays = sess.run(None, {"input": benchmark_input})
        actual = {meta.name: list(array.shape) for meta, array in zip(sess.get_outputs(), arrays)}
        mismatches = [
            {"name": name, "declared": declared[name], "runtime": shape}
            for name, shape in actual.items()
            if name in declared and declared[name] != shape
        ]
        return {
            "outputs_observed": len(actual),
            "declared_runtime_mismatches": mismatches,
            "truthful_for_observed_outputs": not mismatches,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def audit_task(task: int) -> dict[str, Any]:
    path = HERE / "base" / f"task{task:03d}.onnx"
    generator = importlib.import_module(f"task_{HASHES[task]}")
    sakana = load_sakana(task)
    sessions = {name: session(path, level, threads) for name, level, threads in MODES}
    known = scoring.load_examples(task)
    known4 = known["train"] + known["test"]
    known_all = known4 + known["arc-gen"]
    known4_stats = {name: stats() for name in sessions}
    known_all_reference = {"right": 0, "wrong": 0, "errors": 0, "nonfinite": 0}
    known_rule_right = 0
    for index, example in enumerate(known_all):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        known_rule_right += int(rule_equal(task, sakana, example))
        if index < len(known4):
            add_stats(known4_stats, run_case(sessions, benchmark))
        try:
            raw = sessions["disable_t1"].run(["output"], {"input": benchmark["input"]})[0]
            known_all_reference["nonfinite"] += int(not np.isfinite(raw).all())
            known_all_reference["right" if np.array_equal(raw > 0, benchmark["output"] > 0) else "wrong"] += 1
        except Exception:  # noqa: BLE001
            known_all_reference["errors"] += 1

    fresh = {str(seed): {name: stats() for name in sessions} for seed in SEEDS}
    fresh_rule = {str(seed): {"right": 0, "wrong": 0, "generation_errors": 0} for seed in SEEDS}
    first_benchmark = None
    for seed in SEEDS:
        random.seed(seed + task)
        np.random.seed((seed + task) & 0xFFFFFFFF)
        valid = 0
        while valid < FRESH_PER_SEED:
            try:
                example = generator.generate()
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    continue
            except Exception:  # noqa: BLE001
                fresh_rule[str(seed)]["generation_errors"] += 1
                continue
            valid += 1
            if first_benchmark is None:
                first_benchmark = benchmark
            ok = rule_equal(task, sakana, example)
            fresh_rule[str(seed)]["right" if ok else "wrong"] += 1
            add_stats(fresh[str(seed)], run_case(sessions, benchmark))

    result = {
        "task": task,
        "generator": f"inputs/arc-gen-repo/tasks/task_{HASHES[task]}.py",
        "sakana_rule": f"inputs/sakana-gcg-2025/raw/task{task}.py",
        "structure": structure(path),
        "authority_cost": AUTHORITY_COSTS[task],
        "known_rule": {"right": known_rule_right, "total": len(known_all)},
        "known4_four_config": known4_stats,
        "known_all_disable_t1": known_all_reference,
        "fresh_per_seed": FRESH_PER_SEED,
        "fresh_rule": fresh_rule,
        "fresh_four_config": fresh,
    }
    if first_benchmark is not None:
        result["runtime_shape_audit"] = runtime_shape_audit(path, first_benchmark["input"])
    return result


def main() -> None:
    ort.set_default_logger_severity(4)
    payload = {
        "authority": "submission_base_8009.46.zip",
        "tasks": [audit_task(task) for task in TASKS],
    }
    (HERE / "evidence" / "authority_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        row["task"]: {
            "sha256": row["structure"]["sha256"],
            "known": row["known_all_disable_t1"],
            "fresh": row["fresh_four_config"],
            "rule": row["fresh_rule"],
        }
        for row in payload["tasks"]
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

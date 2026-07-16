#!/usr/bin/env python3
"""Independent fresh, default-ORT, cost, and structural audit for task185."""

from __future__ import annotations

import collections
import hashlib
import importlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 185
SOURCE = HERE / "base/task185.onnx"
CANDIDATE = HERE / "task185_padded_palette.onnx"
GENERATOR = "task_7837ac64"
SEED = 185_799_913
COUNT = 5000
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def session(path: Path, disable_optimizations: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disable_optimizations:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def fresh(path: Path, disable_optimizations: bool) -> dict[str, object]:
    generator = importlib.import_module(GENERATOR)
    random.seed(SEED)
    examples = [generator.generate() for _ in range(COUNT)]
    runtime_errors = 0
    wrong = 0
    first_failure: dict[str, object] | None = None
    try:
        runner = session(path, disable_optimizations)
    except Exception as exc:  # noqa: BLE001
        return {
            "seed": SEED,
            "requested": COUNT,
            "correct": 0,
            "wrong": 0,
            "runtime_errors": COUNT,
            "session_error": repr(exc),
        }
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = runner.run(["output"], {"input": benchmark["input"]})[0] > 0.0
        except Exception as exc:  # noqa: BLE001
            runtime_errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
            continue
        expected = benchmark["output"] > 0.0
        if not np.array_equal(actual, expected):
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "wrong",
                    "input_shape": list(np.asarray(example["input"]).shape),
                    "output_shape": list(np.asarray(example["output"]).shape),
                    "differing_elements": int(np.count_nonzero(actual != expected)),
                }
    return {
        "seed": SEED,
        "requested": COUNT,
        "correct": COUNT - wrong - runtime_errors,
        "wrong": wrong,
        "runtime_errors": runtime_errors,
        "first_failure": first_failure,
        "ort_graph_optimizations_disabled": disable_optimizations,
    }


def structure(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    non_static = []
    for value in values:
        if not value.type.HasField("tensor_type"):
            non_static.append(value.name)
            continue
        dimensions = value.type.tensor_type.shape.dim
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dimensions):
            non_static.append(value.name)
    banned = [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED]
    nested = [
        node.op_type
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    custom = [node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")]
    bias = check_conv_bias(model)
    return {
        "checker_full": True,
        "strict_shape_inference": True,
        "non_static_or_nonpositive_tensors": non_static,
        "standard_opset_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import),
        "custom_node_domains": custom,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_ops": banned,
        "sequence_ops": [node.op_type for node in model.graph.node if "SEQUENCE" in node.op_type.upper()],
        "conv_bias_issues": bias,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(collections.Counter(node.op_type for node in model.graph.node).items())),
        "output_shape": [dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim],
        "pass": not any((non_static, custom, model.functions, model.graph.sparse_initializer, nested, banned, bias)),
    }


def profile(path: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        result = scoring.score_and_verify(
            onnx.load(path), TASK, directory, label=path.stem, require_correct=True
        )
    assert result is not None
    return {
        "memory": result["memory"],
        "params": result["params"],
        "cost": result["cost"],
        "known_correct": bool(result["correct"]),
    }


def main() -> None:
    baseline = profile(SOURCE)
    candidate = profile(CANDIDATE)
    result = {
        "task": TASK,
        "baseline_path": str(SOURCE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "baseline": baseline,
        "candidate_path": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "candidate": candidate,
        "cost_reduction": baseline["cost"] - candidate["cost"],
        "score_gain": float(np.log(baseline["cost"] / candidate["cost"])),
        "fresh_disable_all": fresh(CANDIDATE, True),
        "fresh_default_ort": fresh(CANDIDATE, False),
        "structure": structure(CANDIDATE),
    }
    output = HERE / "task185_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

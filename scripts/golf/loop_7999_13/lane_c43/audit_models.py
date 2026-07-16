#!/usr/bin/env python3
"""Strict structure, known-data, and truthful-shape audit for task201."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 201
MODELS = {
    "baseline": HERE / "baseline" / "task201.onnx",
    "r01": HERE / "archive" / "task201_r01_static543.onnx",
    "r02": HERE / "archive" / "task201_r02_static674.onnx",
    "r03": HERE / "archive" / "task201_r03_static785.onnx",
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def make_truthful(model: onnx.ModelProto) -> onnx.ModelProto:
    repaired = copy.deepcopy(model)
    del repaired.graph.value_info[:]
    repaired = onnx.shape_inference.infer_shapes(
        repaired, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(repaired, full_check=True)
    return repaired


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def known(model: onnx.ModelProto, disabled: bool) -> dict[str, object]:
    runner = session(model, disabled)
    input_name = runner.get_inputs()[0].name
    output_name = runner.get_outputs()[0].name
    row = {"total": 0, "right": 0, "wrong": 0, "errors": 0}
    data = scoring.load_examples(TASK)
    for subset in ("train", "test", "arc-gen"):
        for example in data[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            row["total"] += 1
            try:
                raw = runner.run([output_name], {input_name: benchmark["input"]})[0]
                correct = np.array_equal(raw > 0, benchmark["output"] > 0)
                row["right"] += int(correct)
                row["wrong"] += int(not correct)
            except Exception:  # noqa: BLE001
                row["errors"] += 1
    row["perfect"] = bool(
        row["right"] == row["total"] and row["wrong"] == 0 and row["errors"] == 0
    )
    return row


def runtime_trace(
    model: onnx.ModelProto, truthful: onnx.ModelProto, disabled: bool
) -> dict[str, object]:
    typed = {
        value.name: value
        for value in list(truthful.graph.input)
        + list(truthful.graph.value_info)
        + list(truthful.graph.output)
    }
    declared = {
        value.name: shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    traced = copy.deepcopy(truthful)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runner = ort.InferenceSession(traced.SerializeToString(), options)
    benchmark = scoring.convert_to_numpy(scoring.load_examples(TASK)["train"][0])
    if benchmark is None:
        raise RuntimeError("known example is not convertible")
    outputs = runner.run(names, {"input": benchmark["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, outputs)}
    original_mismatches = [
        {"tensor": name, "declared": value, "actual": actual[name]}
        for name, value in declared.items()
        if name in actual and value != actual[name]
    ]
    truthful_mismatches = [
        {"tensor": name, "declared": shape(value), "actual": actual[name]}
        for name, value in typed.items()
        if name in actual and shape(value) != actual[name]
    ]
    return {
        "mode": "disable_all" if disabled else "default",
        "runtime_tensors": len(actual),
        "original_declared_mismatches": original_mismatches,
        "truthful_declared_mismatches": truthful_mismatches,
        "truthful": not truthful_mismatches,
    }


def audit(label: str, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    truthful = make_truthful(model)
    ops = Counter(node.op_type for node in model.graph.node)
    with tempfile.TemporaryDirectory(prefix=f"c43_{label}_", dir=HERE) as workdir:
        original_score = scoring.score_and_verify(
            copy.deepcopy(model), TASK, workdir, f"{label}_original", require_correct=False
        )
        truthful_score = scoring.score_and_verify(
            copy.deepcopy(truthful), TASK, workdir, f"{label}_truthful", require_correct=False
        )
    traces = [runtime_trace(model, truthful, True), runtime_trace(model, truthful, False)]
    known_rows = {
        "disable_all": known(model, True),
        "default": known(model, False),
    }
    nested = [
        node.op_type
        for node in model.graph.node
        for attribute in node.attribute
        if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "serialized_size": path.stat().st_size,
        "full_checker": True,
        "strict_shape_data_prop": True,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_histogram": dict(sorted(ops.items())),
        "original_score": original_score,
        "truthful_score": truthful_score,
        "known_dual": known_rows,
        "runtime_shapes": traces,
        "shape_cloak": any(row["original_declared_mismatches"] for row in traces),
        "truthful_cost_below_793": bool(truthful_score and truthful_score["cost"] < 793),
        "lookup_red_flags": {
            "tfidf_vectorizers": ops.get("TfIdfVectorizer", 0),
            "hardmax": ops.get("Hardmax", 0),
            "argmax": ops.get("ArgMax", 0),
            "gather": ops.get("Gather", 0) + ops.get("GatherND", 0),
            "scatter": ops.get("ScatterElements", 0) + ops.get("ScatterND", 0),
        },
        "canonical_io": [value.name for value in model.graph.input] == ["input"]
        and [value.name for value in model.graph.output] == ["output"],
        "input_shape": shape(model.graph.input[0]),
        "output_shape": shape(model.graph.output[0]),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in model.graph.node)
        and all(item.domain in ("", "ai.onnx") for item in model.opset_import),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": nested,
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "Sequence" in node.op_type
        ],
        "conv_bias_issues": check_conv_bias(model),
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    rows = [audit(label, path) for label, path in MODELS.items()]
    output = HERE / "model_audit.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            row["label"],
            "original", row["original_score"]["cost"] if row["original_score"] else None,
            "truthful", row["truthful_score"]["cost"] if row["truthful_score"] else None,
            "known", [item["right"] for item in row["known_dual"].values()],
            "shape_cloak", row["shape_cloak"],
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

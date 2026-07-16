#!/usr/bin/env python3
"""Actual-cost, known, structure, UB, and runtime-shape audit for C11 leads."""

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
from lib import scoring  # noqa: E402


BASES = {task: HERE / "base" / f"task{task:03d}.onnx" for task in (77, 79, 90, 96, 143, 153)}
CANDIDATES = {
    "task077_r07": (77, HERE / "candidates" / "task077_r07_static3345.onnx"),
    "task079_r02": (79, HERE / "candidates" / "task079_r02_static173.onnx"),
    "task090_r02": (90, HERE / "candidates" / "task090_r02_static174.onnx"),
    "task090_r04": (90, HERE / "candidates" / "task090_r04_static226.onnx"),
    "task090_r06": (90, HERE / "candidates" / "task090_r06_static418.onnx"),
    "task090_r07": (90, HERE / "candidates" / "task090_r07_static430.onnx"),
    "task096_r01": (96, HERE / "candidates" / "task096_r01_static1111.onnx"),
    "task096_r02": (96, HERE / "candidates" / "task096_r02_static1151.onnx"),
    "task143_r02": (143, HERE / "candidates" / "task143_r02_static148.onnx"),
    "task153_r02": (153, HERE / "candidates" / "task153_r02_static236.onnx"),
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def declared_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(task: int, session: ort.InferenceSession) -> dict[str, object]:
    rows: dict[str, dict[str, int]] = {}
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for subset in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in scoring.load_examples(task)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                actual = session.run([output_name], {input_name: benchmark["input"]})[0] > 0
                if np.array_equal(actual, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
        rows[subset] = {"right": right, "wrong": wrong, "errors": errors}
    rows["total"] = {
        key: sum(row[key] for name, row in rows.items() if name != "total")
        for key in ("right", "wrong", "errors")
    }
    return rows


def conv_bias_findings(model: onnx.ModelProto) -> list[dict[str, object]]:
    inits = {item.name: item for item in model.graph.initializer}
    rows: list[dict[str, object]] = []
    for node in model.graph.node:
        if node.op_type == "Conv":
            weight_index, bias_index = 1, 2
            weight = inits.get(node.input[weight_index]) if len(node.input) > weight_index else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        elif node.op_type == "ConvTranspose":
            weight_index, bias_index = 1, 2
            weight = inits.get(node.input[weight_index]) if len(node.input) > weight_index else None
            group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
            expected = int(weight.dims[1] * group) if weight is not None and len(weight.dims) > 1 else None
        elif node.op_type == "QLinearConv":
            weight_index, bias_index = 3, 8
            weight = inits.get(node.input[weight_index]) if len(node.input) > weight_index else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        else:
            continue
        if len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = inits.get(node.input[bias_index])
        count = int(math.prod(bias.dims)) if bias is not None and bias.dims else None
        rows.append(
            {
                "node": node.name or node.output[0],
                "op": node.op_type,
                "bias": node.input[bias_index],
                "bias_is_initializer": bias is not None,
                "bias_shape": list(bias.dims) if bias is not None else None,
                "expected_channels": expected,
                "safe": bias is not None and len(bias.dims) == 1 and count == expected,
            }
        )
    return rows


def runtime_shape_trace(task: int, original: onnx.ModelProto) -> dict[str, object]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(original), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    declared = {
        value.name: declared_shape(value)
        for value in list(original.graph.output) + list(original.graph.value_info)
    }
    traced = copy.deepcopy(original)
    original_outputs = {value.name for value in traced.graph.output}
    del traced.graph.output[:]
    output_names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in output_names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                output_names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert benchmark is not None
    input_name = traced.graph.input[0].name
    outputs = session.run(output_names, {input_name: benchmark["input"]})
    arrays = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    mismatches = [
        {"tensor": name, "declared": value, "actual": shapes[name]}
        for name, value in declared.items()
        if name in shapes and value != shapes[name]
    ]
    return {
        "runtime_tensors": len(arrays),
        "declared_tensors": len(declared),
        "declared_actual_mismatches": mismatches,
        "undeclared_intermediate_count": sum(
            name not in declared and name not in original_outputs for name in arrays
        ),
        "single_example_intermediate_bytes": sum(
            int(value.nbytes) for name, value in arrays.items() if name not in original_outputs
        ),
    }


def audit(label: str, task: int, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    ops = Counter(node.op_type for node in model.graph.node)
    record: dict[str, object] = {
        "label": label,
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": scoring.calculate_params(model),
        "op_histogram": dict(ops.most_common()),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "lookup_red_flags": {
            "tfidf": ops.get("TfIdfVectorizer", 0),
            "hardmax": ops.get("Hardmax", 0),
            "giant_einsum_nodes": sum(node.op_type == "Einsum" and len(node.input) >= 8 for node in model.graph.node),
            "params_over_10000": scoring.calculate_params(model) > 10_000,
        },
        "nonstandard_domains": [item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}],
        "banned_ops": [node.op_type for node in model.graph.node if node.op_type.upper() in BANNED or "Sequence" in node.op_type],
        "nested_graph_attributes": sum(
            1
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "conv_bias_findings": conv_bias_findings(model),
    }
    try:
        onnx.checker.check_model(model, full_check=True)
        record["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(full_check=False, full_check_error=repr(exc))
    try:
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        record["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        record.update(strict_shape_data_prop=False, strict_shape_error=repr(exc))
    try:
        record["runtime_shape_trace"] = runtime_shape_trace(task, model)
    except Exception as exc:  # noqa: BLE001
        record["runtime_shape_trace"] = {"error": f"{type(exc).__name__}: {exc}"}
    with tempfile.TemporaryDirectory(prefix=f"c11_{task}_", dir="/tmp") as workdir:
        try:
            record["official_like_score"] = scoring.score_and_verify(
                copy.deepcopy(model), task, workdir, label, require_correct=False
            )
        except Exception as exc:  # noqa: BLE001
            record["official_like_score_error"] = f"{type(exc).__name__}: {exc}"
    for disable_all, key in ((True, "known_disable_all"), (False, "known_default")):
        try:
            record[key] = known(task, make_session(model, disable_all))
        except Exception as exc:  # noqa: BLE001
            record[key] = {"session_error": f"{type(exc).__name__}: {exc}"}
    return record


def main() -> None:
    ort.set_default_logger_severity(4)
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = {}
    for task, path in BASES.items():
        label = f"base_task{task:03d}"
        output[label] = audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score")
        print(label, None if score is None else score.get("cost"), None if score is None else score.get("correct"), flush=True)
    for label, (task, path) in CANDIDATES.items():
        output[label] = audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score")
        print(label, None if score is None else score.get("cost"), None if score is None else score.get("correct"), flush=True)


if __name__ == "__main__":
    main()

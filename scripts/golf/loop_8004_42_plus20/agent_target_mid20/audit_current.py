#!/usr/bin/env python3
"""Audit immutable 8005.16 incumbents without promoting or editing shared files."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CURRENT = HERE / "current"
TASKS = (51, 64, 185, 200, 245, 264, 394, 397)
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def initializer_params(model: onnx.ModelProto) -> int:
    return sum(max(1, math.prod(init.dims)) for init in model.graph.initializer)


def bias_audit(model: onnx.ModelProto) -> list[dict[str, object]]:
    inits = {init.name: init for init in model.graph.initializer}
    rows = []
    for index, node in enumerate(model.graph.node):
        expected = None
        actual = None
        bias_name = None
        if node.op_type == "Conv" and len(node.input) >= 3:
            weight = inits.get(node.input[1])
            bias = inits.get(node.input[2])
            if weight is not None and weight.dims:
                expected = int(weight.dims[0])
            if bias is not None and bias.dims:
                actual = int(bias.dims[0])
                bias_name = bias.name
        elif node.op_type == "ConvTranspose" and len(node.input) >= 3:
            weight = inits.get(node.input[1])
            bias = inits.get(node.input[2])
            group = next(
                (int(attr.i) for attr in node.attribute if attr.name == "group"), 1
            )
            if weight is not None and len(weight.dims) >= 2:
                expected = int(weight.dims[1]) * group
            if bias is not None and bias.dims:
                actual = int(bias.dims[0])
                bias_name = bias.name
        elif node.op_type == "QLinearConv" and len(node.input) >= 9:
            weight = inits.get(node.input[3])
            bias = inits.get(node.input[8])
            if weight is not None and weight.dims:
                expected = int(weight.dims[0])
            if bias is not None and bias.dims:
                actual = int(bias.dims[0])
                bias_name = bias.name
        if expected is not None or actual is not None:
            rows.append(
                {
                    "node": index,
                    "op": node.op_type,
                    "bias": bias_name,
                    "expected_channels": expected,
                    "actual_bias_len": actual,
                    "ub_free": expected == actual if actual is not None else True,
                }
            )
    return rows


def main() -> int:
    ort.set_default_logger_severity(3)
    report: dict[str, object] = {}
    for task in TASKS:
        path = CURRENT / f"task{task:03d}.onnx"
        data = path.read_bytes()
        model = onnx.load_from_string(data)
        entry: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "file_size": len(data),
            "opsets": {item.domain: item.version for item in model.opset_import},
            "standard_domains_only": all(
                item.domain in {"", "ai.onnx"} for item in model.opset_import
            )
            and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
            "ops": [node.op_type for node in model.graph.node],
            "initializer_params_static": initializer_params(model),
            "lookup_ops": [
                {"node": index, "op": node.op_type}
                for index, node in enumerate(model.graph.node)
                if node.op_type in {"TfIdfVectorizer"}
            ],
            "center_crop_pad_nodes": sum(
                node.op_type == "CenterCropPad" for node in model.graph.node
            ),
            "einsums": [
                {
                    "node": index,
                    "inputs": len(node.input),
                    "equation": next(
                        (
                            onnx.helper.get_attribute_value(attr).decode(
                                "utf-8", errors="replace"
                            )
                            for attr in node.attribute
                            if attr.name == "equation"
                        ),
                        "",
                    ),
                }
                for index, node in enumerate(model.graph.node)
                if node.op_type == "Einsum"
            ],
            "bias_audit": bias_audit(model),
        }
        try:
            onnx.checker.check_model(model, full_check=True)
            entry["checker_full"] = True
        except Exception as exc:
            entry["checker_full"] = False
            entry["checker_error"] = repr(exc)
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
            entry["strict_shape_inference"] = True
            entry["declared_output_shapes"] = [dims(v) for v in inferred.graph.output]
        except Exception as exc:
            entry["strict_shape_inference"] = False
            entry["shape_error"] = repr(exc)

        examples = scoring.load_examples(task)
        known = [
            example
            for subset in ("train", "test", "arc-gen")
            for example in examples.get(subset, [])
        ]
        entry["known_examples"] = len(known)
        for disabled, label in ((True, "disable_all"), (False, "default")):
            try:
                session = make_session(model, disabled)
                right, wrong, _ = scoring.verify_subset(session, known)
                sample = next(
                    scoring.convert_to_numpy(example)
                    for example in known
                    if scoring.convert_to_numpy(example) is not None
                )
                raw = scoring._raw_output(session, sample["input"])
                entry[label] = {
                    "session": True,
                    "right": right,
                    "wrong": wrong,
                    "runtime_failures": wrong,
                    "observed_output_shape": list(raw.shape),
                    "perfect": right == len(known) and wrong == 0,
                }
            except Exception as exc:
                entry[label] = {"session": False, "error": repr(exc), "perfect": False}

        with tempfile.TemporaryDirectory(prefix=f"mid20_task{task:03d}_") as workdir:
            try:
                scored = scoring.score_and_verify(
                    model,
                    task,
                    workdir,
                    label="incumbent",
                    require_correct=False,
                )
                entry["score"] = (
                    None
                    if scored is None
                    else {
                        "cost": int(scored["cost"]),
                        "memory": int(scored["memory"]),
                        "params": int(scored["params"]),
                        "correct": bool(scored["correct"]),
                    }
                )
            except Exception as exc:
                entry["score"] = None
                entry["score_error"] = repr(exc)
        report[f"task{task:03d}"] = entry
        print(
            f"task{task:03d} cost={entry.get('score')} "
            f"known_disabled={entry.get('disable_all')} default={entry.get('default')}",
            flush=True,
        )
    output = HERE / "current_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

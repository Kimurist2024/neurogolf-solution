#!/usr/bin/env python3
"""Reproduce the rejection audit of the task200 cost-344 experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PATH = HERE / "rejected/task200_zero_background_cost344.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(model))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(model.SerializeToString(), options)


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> int:
    data = PATH.read_bytes()
    model = onnx.load_from_string(data)
    result: dict[str, object] = {
        "task": 200,
        "path": str(PATH.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "file_size": len(data),
        "standard_domain": all(
            item.domain in {"", "ai.onnx"} for item in model.opset_import
        )
        and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "lookup_ops": [
            node.op_type for node in model.graph.node if node.op_type == "TfIdfVectorizer"
        ],
        "center_crop_pad_nodes": sum(
            node.op_type == "CenterCropPad" for node in model.graph.node
        ),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    }
    onnx.checker.check_model(model, full_check=True)
    result["checker_full"] = True
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    result["strict_data_prop"] = True
    result["declared_output_shapes"] = [shape(value) for value in inferred.graph.output]
    inits = {init.name: init for init in model.graph.initializer}
    conv = next(node for node in model.graph.node if node.op_type == "Conv")
    expected = int(inits[conv.input[1]].dims[0])
    actual = int(inits[conv.input[2]].dims[0]) if len(conv.input) >= 3 else None
    result["bias_ub"] = {
        "expected_channels": expected,
        "actual_bias_len": actual,
        "bias_present": actual is not None,
        "ub_free": actual is None or expected == actual,
    }
    examples = scoring.load_examples(200)
    known = [
        example
        for subset in ("train", "test", "arc-gen")
        for example in examples.get(subset, [])
    ]
    for disabled, label in ((True, "disable_all"), (False, "default")):
        sess = session(model, disabled)
        right, wrong, _ = scoring.verify_subset(sess, known)
        sample = scoring.convert_to_numpy(known[0])
        if sample is None:
            raise AssertionError("first known case not convertible")
        raw = scoring._raw_output(sess, sample["input"])
        result[label] = {
            "right": right,
            "wrong": wrong,
            "runtime_failures": wrong,
            "observed_output_shape": list(raw.shape),
            "perfect": right == len(known) and wrong == 0,
        }
    margin_ok, margin_min = scoring.model_margin_stable(model, 200)
    result["margin"] = {"stable": margin_ok, "min_nonzero_abs": margin_min}
    with tempfile.TemporaryDirectory(prefix="mid20_task200_final_") as workdir:
        scored = scoring.score_and_verify(
            model, 200, workdir, label="final", require_correct=True
        )
    if scored is None:
        raise AssertionError("score_and_verify rejected finalist")
    result["score"] = {
        "cost": int(scored["cost"]),
        "memory": int(scored["memory"]),
        "params": int(scored["params"]),
        "correct": bool(scored["correct"]),
    }
    result["incumbent_cost"] = 346
    result["strictly_cheaper"] = int(scored["cost"]) < 346
    result["passed_pre_fresh"] = bool(
        result["standard_domain"]
        and not result["lookup_ops"]
        and result["center_crop_pad_nodes"] == 0
        and result["max_einsum_inputs"] < 8
        and result["bias_ub"]["ub_free"]
        and result["disable_all"]["perfect"]
        and result["default"]["perfect"]
        and result["margin"]["stable"]
        and result["strictly_cheaper"]
    )
    output = HERE / "task200_candidate_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed_pre_fresh"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

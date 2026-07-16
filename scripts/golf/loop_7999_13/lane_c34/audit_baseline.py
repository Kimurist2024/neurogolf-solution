#!/usr/bin/env python3
"""Run a bounded strict structural audit for the C34 task009 baseline."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def one_runtime_shape(model: onnx.ModelProto, disabled: bool) -> list[int]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    assert sanitized is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(sanitized.SerializeToString(), options)
    example = scoring.load_examples(9)["train"][0]
    benchmark = scoring.convert_to_numpy(example)
    assert benchmark is not None
    return list(np.asarray(session.run(None, {"input": benchmark["input"]})[0]).shape)


def main() -> None:
    path = HERE / "baseline" / "task009.onnx"
    model = onnx.load(path)
    ops = Counter(node.op_type for node in model.graph.node)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed_names = {
        value.name
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    node_outputs = {
        name for node in inferred.graph.node for name in node.output if name
    }
    record = {
        "label": "base_task009",
        "task": 9,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "params": scoring.calculate_params(model),
        "op_histogram": dict(ops.most_common()),
        "full_check": True,
        "strict_shape_data_prop": True,
        "all_node_outputs_statically_typed": node_outputs <= typed_names,
        "missing_static_outputs": sorted(node_outputs - typed_names),
        "declared_input_shape": shape(model.graph.input[0]),
        "declared_output_shape": shape(model.graph.output[0]),
        "runtime_output_shape_disabled": one_runtime_shape(model, True),
        "runtime_output_shape_default": one_runtime_shape(model, False),
        "nonstandard_domains": [
            item.domain
            for item in model.opset_import
            if item.domain not in {"", "ai.onnx"}
        ],
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper() in BANNED or "Sequence" in node.op_type
        ],
        "nested_graph_attributes": sum(
            1
            for node in model.graph.node
            for attr in node.attribute
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
        ),
        "function_count": len(model.functions),
        "sparse_initializer_count": len(model.graph.sparse_initializer),
        "lookup_red_flags": {
            "tfidf": ops.get("TfIdfVectorizer", 0),
            "hardmax": ops.get("Hardmax", 0),
            "einsum": ops.get("Einsum", 0),
        },
        "conv_count": ops.get("Conv", 0),
        "conv_with_bias_count": sum(
            node.op_type == "Conv" and len(node.input) >= 3 and bool(node.input[2])
            for node in model.graph.node
        ),
        "cost_authority": {
            "source": "external_validator_task009.json",
            "memory": 2567,
            "params": 52,
            "cost": 2619,
        },
    }
    (HERE / "candidate_audit.json").write_text(
        json.dumps({"base_task009": record}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

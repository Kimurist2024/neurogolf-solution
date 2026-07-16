#!/usr/bin/env python3
"""Final structural and dual-known audit for the fixed task192 evidence model."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402
import verify_fix  # noqa: E402


CANDIDATE = HERE / "candidates/task192_large2k_k4l_k4l_p0p0.onnx"
EXACT_CONTROL = ROOT / "scripts/golf/scratch_codex/task192/pad_axes_probe.onnx"
BASE = ROOT / "submission_base_8005.17.zip"
BANNED = {
    "Loop",
    "Scan",
    "NonZero",
    "Unique",
    "Script",
    "Function",
    "Compress",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory() as directory:
        result = scoring.score_and_verify(
            model, 192, directory, label=label, require_correct=False
        )
    if result is None:
        return None
    return {
        key: result.get(key)
        for key in ("memory", "params", "cost", "score", "correct")
    }


def all_static(inferred: onnx.ModelProto) -> bool:
    values = list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for value in values
        for dimension in value.type.tensor_type.shape.dim
    )


def nested_graph_count(model: onnx.ModelProto) -> int:
    return sum(
        attribute.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
        for node in model.graph.node
        for attribute in node.attribute
    )


def make_input(example: dict[str, object]) -> np.ndarray:
    grid = np.asarray(example["input"], dtype=np.int8)
    height, width = grid.shape
    tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for color in range(10):
        tensor[0, color, :height, :width] = grid == color
    return tensor


def runtime_shapes(model: onnx.ModelProto, inferred: onnx.ModelProto) -> dict[str, object]:
    types = {value.name: value for value in inferred.graph.value_info}
    exposed = copy.deepcopy(model)
    names = []
    for node in exposed.graph.node:
        for name in node.output:
            if name == "output" or name not in types:
                continue
            exposed.graph.output.append(copy.deepcopy(types[name]))
            names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    inference = ort.InferenceSession(exposed.SerializeToString(), options)
    known = scoring.load_examples(192)
    example = (known["train"] + known["test"] + known["arc-gen"])[0]
    outputs = inference.run(None, {"input": make_input(example)})
    actual = {name: list(value.shape) for name, value in zip(names, outputs[1:], strict=True)}
    declared = {
        name: [dimension.dim_value for dimension in types[name].type.tensor_type.shape.dim]
        for name in names
    }
    mismatches = [
        {"name": name, "declared": declared[name], "actual": actual[name]}
        for name in names
        if declared[name] != actual[name]
    ]
    return {"declared": declared, "actual": actual, "mismatches": mismatches, "truthful": not mismatches}


def known_mode(model: onnx.ModelProto, optimized: bool) -> dict[str, int]:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        return {"right": 0, "wrong": 0, "errors": 265}
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    inference = ort.InferenceSession(sanitized.SerializeToString(), options)
    examples = scoring.load_examples(192)
    right = wrong = errors = 0
    for subset in ("train", "test", "arc-gen"):
        subset_right, subset_wrong, _ = scoring.verify_subset(
            inference, examples[subset]
        )
        right += subset_right
        wrong += subset_wrong
    return {"right": right, "wrong": wrong, "errors": errors}


def structural(model: onnx.ModelProto) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    domains = sorted({node.domain for node in model.graph.node})
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED or "Sequence" in node.op_type
        }
    )
    conv_bias = []
    inferred_shapes = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in inferred.graph.value_info
    }
    initializer_shapes = {item.name: list(item.dims) for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        weight = initializer_shapes.get(node.input[1])
        bias_shape = inferred_shapes.get(node.input[2], initializer_shapes.get(node.input[2]))
        output_channels = weight[0] if weight else None
        conv_bias.append(
            {
                "node_output": list(node.output),
                "weight_shape": weight,
                "bias_shape": bias_shape,
                "output_channels": output_channels,
                "safe": bias_shape == [output_channels],
            }
        )
    return {
        "full_check": True,
        "strict_shape_inference_data_prop": True,
        "all_shapes_static_positive": all_static(inferred),
        "runtime_shapes": runtime_shapes(model, inferred),
        "domains": domains,
        "standard_domain_only": domains in ([], [""]),
        "banned_ops": banned,
        "nested_graph_attributes": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node)),
        "max_node_inputs": max(len(node.input) for node in model.graph.node),
        "conv_bias": conv_bias,
        "conv_bias_ub_count": sum(not item["safe"] for item in conv_bias),
        "initializer_elements": sum(numpy_helper.to_array(item).size for item in model.graph.initializer),
    }


def main() -> None:
    candidate = onnx.load(CANDIDATE)
    exact = onnx.load(EXACT_CONTROL)
    with zipfile.ZipFile(BASE) as archive:
        baseline_member = archive.read("task192.onnx")
    stable, margin = scoring.model_margin_stable(candidate, 192)
    independent = json.loads((HERE / "large2k_audit_2x500.json").read_text())
    candidate_fresh = next(
        row for row in independent["models"] if row["sha256"] == sha(CANDIDATE)
    )
    result = {
        "task": 192,
        "baseline": {
            "zip": BASE.name,
            "zip_sha256": sha(BASE),
            "member_sha256": hashlib.sha256(baseline_member).hexdigest(),
            "cost": 1609,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha(CANDIDATE),
            "score": score(candidate, "candidate"),
            "known_disabled": known_mode(candidate, False),
            "known_default": known_mode(candidate, True),
            "official_gold": verify_fix.official_gold(CANDIDATE, 192),
            "margin_stable": bool(stable),
            "margin_min": margin,
            "fresh_independent": candidate_fresh["fresh"],
            "fresh_two_seed_min_accuracy": candidate_fresh[
                "fresh_two_seed_min_accuracy"
            ],
            "structural": structural(candidate),
            "exact_true_rule": False,
            "private_zero_policy_eligible": False,
            "decision": "EVIDENCE_ONLY_REJECT_PRIVATE_ZERO_NOT_GUARANTEED",
        },
        "exact_control": {
            "path": str(EXACT_CONTROL.relative_to(ROOT)),
            "sha256": sha(EXACT_CONTROL),
            "score": score(exact, "exact_control"),
            "known_disabled": known_mode(exact, False),
            "known_default": known_mode(exact, True),
            "historical_fresh_evidence": "5000/5000 in scripts/golf/scratch_codex_plus10/review_task192/REPORT.md",
            "exact_true_rule_on_generator_support": True,
            "strictly_cheaper_than_1609": False,
            "decision": "REJECT_COST_INFERIOR",
        },
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "verdict": "NO_ADOPTION_FAIL_CLOSED_PRIVATE_ZERO_POLICY",
        "zip_modified": False,
        "protected_files_modified": False,
    }
    (HERE / "final_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

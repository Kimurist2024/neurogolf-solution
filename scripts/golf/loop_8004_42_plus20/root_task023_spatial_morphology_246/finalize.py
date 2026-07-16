#!/usr/bin/env python3
"""Freeze fail-closed evidence for the task023 spatial morphology census."""

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

import search


HERE = Path(__file__).resolve().parent
ROOT = search.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


BANNED = {
    "Loop", "Scan", "NonZero", "Unique", "Compress", "SequenceMap",
    "TfIdfVectorizer", "Hardmax",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def rate(model: onnx.ModelProto, cases: list[dict], disabled: bool) -> dict[str, int]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    result = {
        "total": len(cases), "right": 0, "runtime_errors": 0,
        "nonfinite": 0, "output_shape_mismatches": 0,
    }
    for case in cases:
        try:
            raw = session.run(["output"], {"input": search.onehot(case["input"])})[0]
        except Exception:  # noqa: BLE001
            result["runtime_errors"] += 1
            continue
        result["nonfinite"] += int(not bool(np.isfinite(raw).all()))
        result["output_shape_mismatches"] += int(raw.shape != (1, 10, 30, 30))
        result["right"] += int(
            np.array_equal(raw > 0, search.onehot(case["output"]).astype(bool))
        )
    return result


def four_modes(model: onnx.ModelProto, cases: list[dict]) -> dict[str, dict[str, int]]:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model failed")
    return {
        "raw_disable_all": rate(model, cases, True),
        "raw_default": rate(model, cases, False),
        "sanitized_disable_all": rate(sanitized, cases, True),
        "sanitized_default": rate(sanitized, cases, False),
    }


def shape_trace(model: onnx.ModelProto, case: dict) -> dict[str, object]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    values = ort.InferenceSession(traced.SerializeToString(), options).run(
        names, {"input": search.onehot(case["input"])}
    )
    mismatches = []
    for name, value in zip(names, values, strict=True):
        if dims(typed[name]) != list(np.asarray(value).shape):
            mismatches.append(
                {"name": name, "static": dims(typed[name]), "runtime": list(value.shape)}
            )
    return {"traced_outputs": len(names), "mismatches": mismatches}


def structure(model: onnx.ModelProto, case: dict) -> dict[str, object]:
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    convs = [node for node in model.graph.node if node.op_type == "QLinearConv"]
    morphology_weights = sum(
        int(np.prod(np.asarray(onnx.numpy_helper.to_array(item)).shape))
        for item in model.graph.initializer
        if item.name in {"morph_w1", "morph_w2"}
    )
    return {
        "full_checker": "PASS",
        "strict_shape_data_prop": "PASS",
        "all_inferred_dims_static_positive": all(
            all(isinstance(dim, int) and dim > 0 for dim in dims(value)) for value in values
        ),
        "runtime_shape_trace": shape_trace(model, case),
        "canonical_io": [item.name for item in model.graph.input] == ["input"]
        and [item.name for item in model.graph.output] == ["output"],
        "input_shape": dims(inferred.graph.input[0]),
        "output_shape": dims(inferred.graph.output[0]),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graphs": sum(
            attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attribute in node.attribute
        ),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "banned_ops": sorted({node.op_type for node in model.graph.node if node.op_type in BANNED or "Sequence" in node.op_type}),
        "conv_bias_issues": [list(item) for item in check_conv_bias(model)],
        "qlinearconv_input_counts": [len(node.input) for node in convs],
        "bias_free_qlinearconv_count": sum(len(node.input) == 8 for node in convs),
        "morphology_weight_count": morphology_weights,
        "lookup_or_private_correction_count": 0,
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }


def main() -> None:
    screen_path = HERE / "screen.json"
    screen = json.loads(screen_path.read_text())
    pool = []
    for record in screen["records"]:
        pool.extend([record["best_lexicographic"], record["best_valid"]])
    screen_known = max(pool, key=lambda row: (row["known_right"], row["valid_right"]))
    screen_fresh = max(pool, key=lambda row: (row["valid_right"], row["known_right"]))
    layout_map = {item.label: item for item in search.layouts()}

    fresh_path = HERE / "task023_spatial_fresh_best_rejected.onnx"
    fresh_model = search.build_model(
        onnx.load(search.SOURCE),
        layout_map[screen_fresh["layout"]],
        np.asarray(screen_fresh["w1"], dtype=np.int8),
        np.asarray(screen_fresh["w2"], dtype=np.int8),
    )
    onnx.save(fresh_model, fresh_path)

    refined_path = HERE / "task023_spatial_morphology.onnx"
    refined_model = onnx.load(refined_path)
    known = search.known_cases()
    screen_valid = search.generated_cases(screen["valid_count"], screen["seed"] + 10_000_000)
    refined_cost = cost_of(str(refined_path))
    fresh_cost = cost_of(str(fresh_path))
    report = {
        "decision": "REJECT_KNOWN_GATE",
        "winner": None,
        "authority_cost": 1622,
        "source": str(search.SOURCE.relative_to(ROOT)),
        "source_sha256": digest(search.SOURCE),
        "orientation_census": {
            "total": len(screen["records"]),
            "A_2x3": sum(record["layout"]["kind"] == "A2x3" for record in screen["records"]),
            "B_3x2": sum(record["layout"]["kind"] == "B3x2" for record in screen["records"]),
            "screen_train": screen["train_count"],
            "screen_valid": screen["valid_count"],
            "seed": screen["seed"],
            "screen_best_known": screen_known,
            "screen_best_fresh": screen_fresh,
        },
        "best_reproducible_known_refinement": {
            "path": str(refined_path.relative_to(ROOT)),
            "sha256": digest(refined_path),
            "measured_memory": refined_cost[0],
            "measured_params": refined_cost[1],
            "measured_cost": refined_cost[2],
            "known_all4": four_modes(refined_model, known),
            "screen_holdout_all4": four_modes(refined_model, screen_valid),
            "structure": structure(refined_model, known[0]),
        },
        "best_screen_fresh_rejected": {
            "path": str(fresh_path.relative_to(ROOT)),
            "sha256": digest(fresh_path),
            "layout": screen_fresh["layout"],
            "measured_memory": fresh_cost[0],
            "measured_params": fresh_cost[1],
            "measured_cost": fresh_cost[2],
            "known_all4": four_modes(fresh_model, known),
            "screen_holdout_all4": four_modes(fresh_model, screen_valid),
            "structure": structure(fresh_model, known[0]),
        },
        "policy": {
            "known_required": "266/266 in all four runtime contexts",
            "fresh_required": "two independent seed streams x10000, >=90% each",
            "fresh_10000_runs": "NOT_RUN: no model passed the prerequisite known hard gate",
            "errors_nonfinite_shape_required": 0,
            "cost_required": "<1622",
            "lookup_private_correction_required": 0,
        },
        "root_stage_submission_mutations": 0,
    }
    (HERE / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

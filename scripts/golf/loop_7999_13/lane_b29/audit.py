#!/usr/bin/env python3
"""Final B29 structural and metadata-equivalence audit."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.loop_7999_13.lane_c11 import audit_candidates as c11  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


BASELINE = HERE / "baseline_task366.onnx"
SOURCE = HERE / "source_task366_cost7646_shape_cloak.onnx"
REPAIRED = HERE / "task366_cost7646_truthful_metadata.onnx"
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_shape_ok(value: onnx.ValueInfoProto) -> bool:
    return all(
        dimension.HasField("dim_value") and dimension.dim_value > 0
        for dimension in value.type.tensor_type.shape.dim
    )


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    ops = Counter(node.op_type for node in model.graph.node)
    trace = c11.runtime_shape_trace(366, copy.deepcopy(model))
    memory, params, cost = (int(value) for value in cost_of(str(path)))
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path),
        "checker_full": True,
        "strict_shape_data_prop": True,
        "static_positive_shapes": all(static_shape_ok(value) for value in values),
        "runtime_shape_trace": trace,
        "shape_truthful": not trace["declared_actual_mismatches"]
        and trace["undeclared_intermediate_count"] == 0,
        "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
        and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
        "no_banned_or_nested": not model.functions
        and not model.graph.sparse_initializer
        and all(
            node.op_type.upper() not in BANNED
            and "SEQUENCE" not in node.op_type.upper()
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for attr in node.attribute
            )
            for node in model.graph.node
        ),
        "lookup_red_flags": {
            "tfidf": ops.get("TfIdfVectorizer", 0),
            "hardmax": ops.get("Hardmax", 0),
            "giant_einsum_nodes": sum(
                node.op_type == "Einsum" and len(node.input) >= 8
                for node in model.graph.node
            ),
            "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        },
        "conv_bias_findings": check_conv_bias(model),
        "memory": memory,
        "params": params,
        "cost": cost,
    }


def same_computation(source: onnx.ModelProto, repaired: onnx.ModelProto) -> dict[str, bool]:
    return {
        "nodes_byte_identical": [node.SerializeToString(deterministic=True) for node in source.graph.node]
        == [node.SerializeToString(deterministic=True) for node in repaired.graph.node],
        "initializers_byte_identical": [item.SerializeToString(deterministic=True) for item in source.graph.initializer]
        == [item.SerializeToString(deterministic=True) for item in repaired.graph.initializer],
        "opsets_byte_identical": [item.SerializeToString(deterministic=True) for item in source.opset_import]
        == [item.SerializeToString(deterministic=True) for item in repaired.opset_import],
        "input_names_identical": [value.name for value in source.graph.input]
        == [value.name for value in repaired.graph.input],
        "output_names_identical": [value.name for value in source.graph.output]
        == [value.name for value in repaired.graph.output],
    }


def make_session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def probe_differential(source: onnx.ModelProto, repaired: onnx.ModelProto) -> dict[str, Any]:
    examples = scoring.load_examples(366)
    executable_arc = [
        example for example in examples["arc-gen"] if scoring.convert_to_numpy(example) is not None
    ]
    probes = [examples["train"][0], examples["train"][-1], examples["test"][0], executable_arc[0], executable_arc[-1]]
    rows = {}
    for label, disabled in (("disable_all", True), ("default", False)):
        left = make_session(source, disabled)
        right = make_session(repaired, disabled)
        raw_equal = threshold_equal = errors = 0
        max_abs_difference = 0.0
        for example in probes:
            benchmark = scoring.convert_to_numpy(example)
            assert benchmark is not None
            try:
                left_raw = np.asarray(left.run(["output"], {"input": benchmark["input"]})[0])
                right_raw = np.asarray(right.run(["output"], {"input": benchmark["input"]})[0])
                if np.array_equal(left_raw, right_raw):
                    raw_equal += 1
                if np.array_equal(left_raw > 0, right_raw > 0):
                    threshold_equal += 1
                max_abs_difference = max(
                    max_abs_difference, float(np.max(np.abs(left_raw - right_raw)))
                )
            except Exception:
                errors += 1
        rows[label] = {
            "requested": len(probes),
            "raw_equal": raw_equal,
            "threshold_equal": threshold_equal,
            "errors": errors,
            "max_abs_difference": max_abs_difference,
        }
    return rows


def main() -> int:
    baseline = structure(BASELINE)
    source = structure(SOURCE)
    repaired = structure(REPAIRED)
    source_model = onnx.load(SOURCE)
    repaired_model = onnx.load(REPAIRED)
    computation = same_computation(source_model, repaired_model)
    if not all(computation.values()):
        raise RuntimeError(computation)
    differential = probe_differential(source_model, repaired_model)
    payload = {
        "task": 366,
        "baseline_score_label": 8000.46,
        "baseline": baseline,
        "source": source,
        "repaired": repaired,
        "metadata_only_equivalence": computation,
        "five_probe_raw_differential": differential,
        "historical_fresh_evidence": {
            "requested": 5000,
            "executable": 4757,
            "right": 4685,
            "wrong": 72,
            "rate": 0.9848644103426529,
            "source": "scripts/golf/loop_7999_13/lane_c17/fresh_evidence.json",
        },
        "decision": {
            "status": "REJECT_TRUTHFUL_COST_ABOVE_BASELINE",
            "winner_count": 0,
            "verified_gain": 0.0,
            "cost_delta": repaired["cost"] - baseline["cost"],
            "reasons": [
                "all 107 source mismatches and eight undeclared intermediates were repaired without changing nodes or initializers",
                "repaired cost is 9465, which is 1478 above baseline 7987",
                "fresh5000 dual-mode and external validator500 are skipped because the prerequisite repaired cost below baseline failed",
            ],
        },
    }
    (HERE / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: {field: row[field] for field in ("sha256", "cost", "shape_truthful", "conv_bias_findings")} for key, row in (("baseline", baseline), ("source", source), ("repaired", repaired))}, indent=2))
    print(json.dumps(payload["five_probe_raw_differential"], indent=2))
    print(json.dumps(payload["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Structural and dual-known differential audit for the A37 task105 candidate."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import audit_task013_candidate as common


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK = 105
BASELINE = HERE.parent / "lane_initializer_contraction_wave17" / "task105_combined.onnx"
CANDIDATE = HERE / "task105_remove_output_one.onnx"
OUTPUT = HERE / "task105_candidate_audit.json"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "SequenceMap"}


def main() -> None:
    common.TASK = TASK
    baseline = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(candidate, full_check=True)
    except Exception as exc:  # noqa: BLE001
        checker = False
        checker_error = repr(exc)
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(candidate), strict_mode=True, data_prop=True
        )
    except Exception as exc:  # noqa: BLE001
        strict = False
        strict_error = repr(exc)
        inferred = candidate
    base_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in baseline.graph.initializer
    }
    candidate_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in candidate.graph.initializer
    }
    base_inputs = [len(node.input) for node in baseline.graph.node if node.op_type == "Einsum"]
    candidate_inputs = [len(node.input) for node in candidate.graph.node if node.op_type == "Einsum"]
    with tempfile.TemporaryDirectory(prefix="a37_task105_") as workdir:
        score = common.scoring.score_and_verify(
            copy.deepcopy(candidate), TASK, workdir, "a37", require_correct=False
        )
    known = {
        "disable_all": common.known_and_differential(baseline, candidate, True),
        "default": common.known_and_differential(baseline, candidate, False),
    }
    runtime_shapes = common.runtime_shape_audit(candidate)
    all_values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    equations = [
        next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")
        for node in candidate.graph.node
        if node.op_type == "Einsum"
    ]
    rewritten = [equation for equation in equations if equation.endswith("->b")]
    structure = {
        "full_checker": checker,
        "checker_error": checker_error,
        "strict_shape_data_prop": strict,
        "strict_error": strict_error,
        "all_static_positive": all(
            all(isinstance(dim, int) and dim > 0 for dim in common.shape(value))
            for value in all_values
        ),
        "runtime_shapes": runtime_shapes,
        "canonical_io": (
            [value.name for value in candidate.graph.input] == ["input"]
            and [value.name for value in candidate.graph.output] == ["output"]
        ),
        "standard_domains": all(node.domain in ("", "ai.onnx") for node in candidate.graph.node)
        and all(item.domain in ("", "ai.onnx") for item in candidate.opset_import),
        "no_functions_sparse_nested": (
            not candidate.functions
            and not candidate.graph.sparse_initializer
            and all(
                attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                for node in candidate.graph.node
                for attr in node.attribute
            )
        ),
        "no_banned_ops": not [node.op_type for node in candidate.graph.node if node.op_type in BANNED],
        "no_external_initializers": all(
            item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
            for item in candidate.graph.initializer
        ),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or np.isfinite(array).all()
            for array in candidate_arrays.values()
        ),
        "conv_bias_issues": common.check_conv_bias(candidate),
        "node_count_unchanged": len(candidate.graph.node) == len(baseline.graph.node),
        "einsum_operand_counts_never_increased": (
            len(candidate_inputs) == len(base_inputs)
            and all(after <= before for before, after in zip(base_inputs, candidate_inputs))
        ),
        "six_operands_removed": sum(base_inputs) - sum(candidate_inputs) == 6,
        "six_output_axes_reused_batch": len(rewritten) == 6,
        "removed_initializers": sorted(set(base_arrays) - set(candidate_arrays)),
        "added_initializers": sorted(set(candidate_arrays) - set(base_arrays)),
        "remaining_initializers_unchanged": all(
            name in candidate_arrays and np.array_equal(array, candidate_arrays[name], equal_nan=True)
            for name, array in base_arrays.items()
            if name != "one_f"
        ),
        "ops": dict(Counter(node.op_type for node in candidate.graph.node)),
    }
    report = {
        "task": TASK,
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
        "score": score,
        "structure": structure,
        "known_dual_and_raw_differential": known,
        "eligible_for_user_95_percent_gate": bool(
            score
            and score.get("correct")
            and score.get("cost") == 194
            and all(value["perfect"] for value in known.values())
            and all(value["raw_bitwise_equal"] == value["total"] for value in known.values())
            and runtime_shapes["truthful"]
            and checker
            and strict
            and structure["all_static_positive"]
            and structure["canonical_io"]
            and structure["standard_domains"]
            and structure["no_functions_sparse_nested"]
            and structure["no_banned_ops"]
            and structure["no_external_initializers"]
            and structure["finite_initializers"]
            and not structure["conv_bias_issues"]
            and structure["einsum_operand_counts_never_increased"]
            and structure["six_operands_removed"]
            and structure["six_output_axes_reused_batch"]
            and structure["removed_initializers"] == ["one_f"]
            and not structure["added_initializers"]
            and structure["remaining_initializers_unchanged"]
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=bool) + "\n")
    print(json.dumps(report, indent=2, default=bool))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Complete structural, known-dual, and baseline-differential audit for A37 task013."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 13
BASELINE = HERE.parent / "lane_initializer_contraction_wave17" / "task013_combined.onnx"
CANDIDATE = HERE / "task013_qch_from_qor_shared_reduction.onnx"
OUTPUT = HERE / "task013_candidate_audit.json"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "SequenceMap"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


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


def known_and_differential(
    baseline: onnx.ModelProto,
    candidate: onnx.ModelProto,
    disabled: bool,
) -> dict[str, object]:
    base_session = session(baseline, disabled)
    candidate_session = session(candidate, disabled)
    row = {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "raw_bitwise_equal": 0,
        "decoded_equal": 0,
        "total": 0,
        "max_abs_raw_difference": 0.0,
    }
    for subset in ("train", "test", "arc-gen"):
        for example in scoring.load_examples(TASK)[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            row["total"] += 1
            try:
                raw_base = base_session.run(["output"], {"input": benchmark["input"]})[0]
                raw_candidate = candidate_session.run(["output"], {"input": benchmark["input"]})[0]
            except Exception:  # noqa: BLE001
                row["runtime_errors"] += 1
                continue
            correct = np.array_equal(raw_candidate > 0, benchmark["output"] > 0)
            row["right"] += int(correct)
            row["wrong"] += int(not correct)
            row["raw_bitwise_equal"] += int(np.array_equal(raw_base, raw_candidate, equal_nan=True))
            row["decoded_equal"] += int(np.array_equal(raw_base > 0, raw_candidate > 0))
            difference = np.abs(
                np.nan_to_num(raw_base).astype(np.float64, copy=False)
                - np.nan_to_num(raw_candidate).astype(np.float64, copy=False)
            )
            row["max_abs_raw_difference"] = max(
                float(row["max_abs_raw_difference"]),
                float(difference.max(initial=0.0)),
            )
    row["perfect"] = bool(row["wrong"] == 0 and row["runtime_errors"] == 0)
    return row


def runtime_shape_audit(model: onnx.ModelProto) -> dict[str, object]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    runner = ort.InferenceSession(traced.SerializeToString(), options)
    benchmark = scoring.convert_to_numpy(scoring.load_examples(TASK)["train"][0])
    if benchmark is None:
        raise RuntimeError("missing trace benchmark")
    values = runner.run(names, {"input": benchmark["input"]})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, values)}
    expected = {name: shape(typed[name]) for name in names}
    mismatches = [
        {"name": name, "static": expected[name], "runtime": actual[name]}
        for name in names
        if expected[name] != actual[name]
    ]
    return {
        "traced_outputs": len(names),
        "mismatches": mismatches,
        "truthful": not mismatches,
        "output_static": shape(inferred.graph.output[0]),
        "output_runtime": actual.get("output"),
    }


def main() -> None:
    ort.set_default_logger_severity(4)
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
    unchanged_initializers = all(
        name in candidate_arrays and np.array_equal(array, candidate_arrays[name], equal_nan=True)
        for name, array in base_arrays.items()
        if name != "Qch"
    )
    base_einsum_inputs = [len(node.input) for node in baseline.graph.node if node.op_type == "Einsum"]
    candidate_einsum_inputs = [len(node.input) for node in candidate.graph.node if node.op_type == "Einsum"]
    final_equation = next(
        attr.s.decode("ascii")
        for node in candidate.graph.node
        if "output" in node.output
        for attr in node.attribute
        if attr.name == "equation"
    )
    with tempfile.TemporaryDirectory(prefix="a37_task013_") as workdir:
        score = scoring.score_and_verify(
            copy.deepcopy(candidate), TASK, workdir, "a37", require_correct=False
        )
    shapes = runtime_shape_audit(candidate)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    structure = {
        "full_checker": checker,
        "checker_error": checker_error,
        "strict_shape_data_prop": strict,
        "strict_error": strict_error,
        "all_static_positive": all(
            all(isinstance(dim, int) and dim > 0 for dim in shape(value))
            for value in values
        ),
        "runtime_shapes": shapes,
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
        "conv_bias_issues": check_conv_bias(candidate),
        "node_count_unchanged": len(candidate.graph.node) == len(baseline.graph.node),
        "einsum_operand_counts_unchanged": candidate_einsum_inputs == base_einsum_inputs,
        "no_new_or_enlarged_giant_einsum": candidate_einsum_inputs == base_einsum_inputs,
        "shared_reduction_equation_exact_pattern": final_equation.startswith("bshw,")
        and final_equation.endswith("->bkhw")
        and all(term in final_equation for term in ("cXlmm", "cXnpp", "rXABB", "rXCDD", "GXTUU", "GXVWW")),
        "removed_initializers": sorted(set(base_arrays) - set(candidate_arrays)),
        "added_initializers": sorted(set(candidate_arrays) - set(base_arrays)),
        "remaining_initializers_unchanged": unchanged_initializers,
    }
    known = {
        "disable_all": known_and_differential(baseline, candidate, True),
        "default": known_and_differential(baseline, candidate, False),
    }
    report = {
        "task": TASK,
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": digest(BASELINE),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": digest(CANDIDATE),
        "score": score,
        "structure": structure,
        "known_dual_and_raw_differential": known,
        "eligible": bool(
            score
            and score.get("correct")
            and score.get("cost") == 731
            and all(value["perfect"] for value in known.values())
            and all(value["raw_bitwise_equal"] == value["total"] for value in known.values())
            and all(
                value is True
                for key, value in structure.items()
                if key
                not in {
                    "checker_error",
                    "strict_error",
                    "runtime_shapes",
                    "conv_bias_issues",
                    "removed_initializers",
                    "added_initializers",
                }
            )
            and structure["runtime_shapes"]["truthful"]
            and not structure["conv_bias_issues"]
            and structure["removed_initializers"] == ["Qch"]
            and not structure["added_initializers"]
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, default=bool) + "\n")
    print(json.dumps(report, indent=2, default=bool))


if __name__ == "__main__":
    main()

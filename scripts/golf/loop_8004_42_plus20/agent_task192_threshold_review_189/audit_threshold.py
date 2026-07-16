#!/usr/bin/env python3
"""Independent fail-closed audit of task192_hardsigmoid_k31.onnx.

This lane is evidence-only.  It reads the immutable 8009.46 authority, the
staged ArgMax+OneHot control, and root188's k31 candidate.  It never promotes
or modifies any of those inputs.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/root_task192_threshold_188/candidates"
    / "task192_hardsigmoid_k31.onnx"
)
STAGED = ROOT / "others/71407/task192.onnx"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ALL_SCORES = ROOT / "all_scores.csv"

EXPECTED = {
    "authority_zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "root_submission": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
    "authority_task192": "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c",
    "staged": "19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c",
    "candidate": "91315f9982649a65341134c541f904dc5398168600475a4d4f916b09b2f41941",
}
FRESH_SEEDS = (192_189_031, 192_189_977)
FRESH_PER_SEED = 5000
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
LOOKUP_INDEX_OPS = {
    "TFIDFVECTORIZER",
    "HARDMAX",
    "GATHER",
    "GATHERELEMENTS",
    "GATHERND",
    "SCATTER",
    "SCATTERELEMENTS",
    "SCATTERND",
    "ONEHOT",
    "ARGMAX",
    "ARGMIN",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def structural_audit(data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {
        "full_check": False,
        "strict_shape_inference_data_prop": False,
        "reasons": [],
    }
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row["full_check_error"] = f"{type(exc).__name__}: {exc}"
        row["reasons"].append("full_check_failed")
        row["pass"] = False
        return row
    try:
        inferred = shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["strict_shape_inference_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row["strict_shape_inference_error"] = f"{type(exc).__name__}: {exc}"
        row["reasons"].append("strict_shape_inference_data_prop_failed")

    op_histogram = Counter(node.op_type for node in model.graph.node)
    upper_ops = {node.op_type.upper() for node in model.graph.node}
    banned = sorted(
        op
        for op in upper_ops
        if op in BANNED or "SEQUENCE" in op
    )
    lookup = sorted(upper_ops.intersection(LOOKUP_INDEX_OPS))
    hardmax = "HARDMAX" in upper_ops
    standard_domains = all(
        item.domain in ("", "ai.onnx") for item in model.opset_import
    ) and all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    nested_graphs = sum(
        attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    finite_initializers = True
    largest_initializer = 0
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        largest_initializer = max(largest_initializer, int(array.size))
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            finite_initializers = False

    typed = {
        value.name: value
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    graph_outputs = {value.name for value in inferred.graph.output}
    nonstatic: list[str] = []
    inferred_memory = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in graph_outputs:
                continue
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                nonstatic.append(name)
                continue
            shape = dims(value)
            if any(dim is None or dim <= 0 for dim in shape):
                nonstatic.append(name)
                continue
            dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            inferred_memory += math.prod(shape) * np.dtype(dtype).itemsize

    try:
        conv_findings = check_conv_bias(model)
    except Exception as exc:  # noqa: BLE001
        conv_findings = [{"check_error": f"{type(exc).__name__}: {exc}"}]
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    row.update(
        {
            "serialized_bytes": len(data),
            "opset_imports": [
                {"domain": item.domain, "version": int(item.version)}
                for item in model.opset_import
            ],
            "node_count": len(model.graph.node),
            "op_histogram": dict(sorted(op_histogram.items())),
            "initializer_count": len(model.graph.initializer),
            "largest_initializer_elements": largest_initializer,
            "max_einsum_inputs": max(
                (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
                default=0,
            ),
            "banned_ops": banned,
            "lookup_or_index_ops": lookup,
            "hardmax": hardmax,
            "hard_sigmoid_is_not_hardmax": bool(
                op_histogram.get("HardSigmoid") == 1 and not hardmax
            ),
            "standard_domains": standard_domains,
            "nested_graph_count": nested_graphs,
            "function_count": len(model.functions),
            "sparse_initializer_count": len(model.graph.sparse_initializer),
            "finite_initializers": finite_initializers,
            "nonstatic_node_outputs": sorted(set(nonstatic)),
            "all_node_outputs_static_positive": not nonstatic,
            "inferred_memory": int(inferred_memory),
            "params": int(scoring.calculate_params(model) or 0),
            "conv_family_bias_findings": conv_findings,
            "conv_family_bias_ub0": not conv_findings,
            "sanitize_model_pass": sanitized is not None,
        }
    )
    if banned:
        row["reasons"].append("banned_ops")
    if lookup:
        row["reasons"].append("lookup_or_index_ops")
    if hardmax:
        row["reasons"].append("hardmax")
    if not standard_domains:
        row["reasons"].append("nonstandard_domain")
    if nested_graphs or model.functions or model.graph.sparse_initializer:
        row["reasons"].append("nested_function_or_sparse")
    if not finite_initializers:
        row["reasons"].append("nonfinite_initializer")
    if nonstatic:
        row["reasons"].append("nonstatic_node_output")
    if conv_findings:
        row["reasons"].append("conv_family_bias_ub")
    if sanitized is None:
        row["reasons"].append("sanitize_model_failed")
    row["reasons"] = sorted(set(row["reasons"]))
    row["pass"] = not row["reasons"]
    return row


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def runtime_shape_trace(data: bytes, benchmark_input: np.ndarray) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    traced = copy.deepcopy(model)
    existing = {value.name for value in traced.graph.output}
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if not name or name in names or name not in typed:
                continue
            names.append(name)
            if name not in existing:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing.add(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    arrays = session.run(names, {session.get_inputs()[0].name: benchmark_input})
    mismatches = []
    nonfinite = 0
    observed = []
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        declared = dims(typed[name])
        actual = list(value.shape)
        observed.append({"name": name, "declared": declared, "actual": actual})
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "observed": observed,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def official_score(data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"task192_189_{label}_") as directory:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data),
            192,
            directory,
            label=label,
            require_correct=True,
        )


def update_range(stats: dict[str, dict[str, int | None]], name: str, value: int) -> None:
    item = stats.setdefault(name, {"min": None, "max": None})
    item["min"] = value if item["min"] is None else min(int(item["min"]), value)
    item["max"] = value if item["max"] is None else max(int(item["max"]), value)


def color_stats(examples: list[dict[str, Any]]) -> dict[str, Any]:
    ranges: dict[str, dict[str, int | None]] = {}
    box_color_is_argmax = 0
    unique_argmax = 0
    threshold_exact_one = 0
    for example in examples:
        grid = example["input"]
        output = example["output"]
        height, width = len(grid), len(grid[0])
        flat = [value for row in grid for value in row]
        out_flat = [value for row in output for value in row]
        counts = [flat.count(color) for color in range(10)]
        dominant = max(range(1, 10), key=lambda color: counts[color])
        ranked = sorted((counts[color], color) for color in range(1, 10))
        runner_up = ranked[-2][0]
        output_colors = sorted(set(out_flat) - {0})
        box_color = output_colors[0] if len(output_colors) == 1 else -1
        threshold_channels = sum(count >= 32 for count in counts[1:])
        positive_counts = [count for count in counts[1:] if count > 0]
        update_range(ranges, "height", height)
        update_range(ranges, "width", width)
        update_range(ranges, "dominant_color_count", counts[dominant])
        update_range(ranges, "runner_up_color_count", runner_up)
        update_range(ranges, "dominant_minus_runner_up", counts[dominant] - runner_up)
        update_range(ranges, "threshold_selected_channel_count", threshold_channels)
        update_range(ranges, "positive_nonzero_color_channel_count", len(positive_counts))
        update_range(ranges, "minimum_positive_nonzero_color_count", min(positive_counts))
        update_range(ranges, "maximum_positive_nonzero_color_count", max(positive_counts))
        update_range(ranges, "box_color_input_count", counts[box_color] if box_color > 0 else -1)
        update_range(ranges, "output_box_cell_count", sum(value != 0 for value in out_flat))
        box_color_is_argmax += int(box_color == dominant)
        unique_argmax += int(counts[dominant] > runner_up)
        threshold_exact_one += int(threshold_channels == 1)
    total = len(examples)
    return {
        "total": total,
        "ranges": ranges,
        "box_color_is_argmax": box_color_is_argmax,
        "unique_nonzero_argmax": unique_argmax,
        "hard_sigmoid_selects_exactly_one_channel": threshold_exact_one,
        "threshold_condition_perfect": bool(
            total
            and box_color_is_argmax == total
            and unique_argmax == total
            and threshold_exact_one == total
            and int(ranges["dominant_color_count"]["min"] or 0) >= 32
            and int(ranges["runner_up_color_count"]["max"] or 10**9) <= 31
        ),
        "threshold_semantics": (
            "HardSigmoid(alpha=1,beta=-31) maps integer hist count <=31 to 0 "
            "and >=32 to 1. Exact ArgMax equivalence therefore requires one and "
            "only one nonzero-color channel >=32 and all others <=31."
        ),
    }


def task192_rule(grid: list[list[int]]) -> list[list[int]]:
    counts = [sum(row.count(color) for row in grid) for color in range(10)]
    selected = max(range(1, 10), key=lambda color: counts[color])
    height, width = len(grid), len(grid[0])
    output = [[0] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            horizontal = any(
                grid[row][other] == selected
                for other in range(max(0, col - 1), min(width, col + 2))
            )
            vertical = any(
                grid[other][col] == selected
                for other in range(max(0, row - 1), min(height, row + 2))
            )
            if grid[row][col] != 0 and horizontal and vertical:
                output[row][col] = selected
    return output


def empty_compare(total: int) -> dict[str, Any]:
    return {
        "total": total,
        "converted": 0,
        "conversion_errors": 0,
        "candidate_right": 0,
        "staged_right": 0,
        "candidate_runtime_errors": 0,
        "staged_runtime_errors": 0,
        "one_sided_runtime_errors": 0,
        "candidate_nonfinite_values": 0,
        "staged_nonfinite_values": 0,
        "candidate_near_positive_values": 0,
        "raw_bitwise_equal": 0,
        "sign_equal": 0,
        "max_abs_raw_difference": 0.0,
        "candidate_min_positive": None,
        "candidate_max_nonpositive": None,
        "candidate_output_shapes": [],
        "staged_output_shapes": [],
        "first_failure": None,
    }


def run_cases(
    examples: list[dict[str, Any]],
    candidate_session: ort.InferenceSession,
    staged_session: ort.InferenceSession,
    *,
    check_rule: bool,
    progress_label: str | None = None,
) -> dict[str, Any]:
    row = empty_compare(len(examples))
    row["readable_rule_right"] = 0
    for index, example in enumerate(examples):
        if progress_label and (index + 1) % 500 == 0:
            print(f"{progress_label}: {index + 1}/{len(examples)}", flush=True)
        if check_rule:
            row["readable_rule_right"] += int(
                task192_rule(example["input"]) == example["output"]
            )
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            row["conversion_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "index": index,
                "error": "convert_to_numpy returned None",
            }
            continue
        row["converted"] += 1
        candidate_raw = staged_raw = None
        try:
            candidate_raw = np.asarray(
                candidate_session.run(
                    [candidate_session.get_outputs()[0].name],
                    {candidate_session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            row["candidate_runtime_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "index": index,
                "candidate_error": f"{type(exc).__name__}: {exc}",
            }
        try:
            staged_raw = np.asarray(
                staged_session.run(
                    [staged_session.get_outputs()[0].name],
                    {staged_session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            row["staged_runtime_errors"] += 1
            row["first_failure"] = row["first_failure"] or {
                "index": index,
                "staged_error": f"{type(exc).__name__}: {exc}",
            }
        if (candidate_raw is None) != (staged_raw is None):
            row["one_sided_runtime_errors"] += 1
        if candidate_raw is None or staged_raw is None:
            continue
        candidate_shape = list(candidate_raw.shape)
        staged_shape = list(staged_raw.shape)
        if candidate_shape not in row["candidate_output_shapes"]:
            row["candidate_output_shapes"].append(candidate_shape)
        if staged_shape not in row["staged_output_shapes"]:
            row["staged_output_shapes"].append(staged_shape)
        candidate_finite = np.isfinite(candidate_raw)
        staged_finite = np.isfinite(staged_raw)
        row["candidate_nonfinite_values"] += int(
            candidate_raw.size - np.count_nonzero(candidate_finite)
        )
        row["staged_nonfinite_values"] += int(
            staged_raw.size - np.count_nonzero(staged_finite)
        )
        safe = candidate_raw[candidate_finite]
        if safe.size:
            positive = safe[safe > 0]
            nonpositive = safe[safe <= 0]
            row["candidate_near_positive_values"] += int(
                np.count_nonzero((safe > 0) & (safe < 0.25))
            )
            if positive.size:
                value = float(positive.min())
                current = row["candidate_min_positive"]
                row["candidate_min_positive"] = value if current is None else min(current, value)
            if nonpositive.size:
                value = float(nonpositive.max())
                current = row["candidate_max_nonpositive"]
                row["candidate_max_nonpositive"] = value if current is None else max(current, value)
        expected = benchmark["output"].astype(bool)
        candidate_sign = candidate_raw > 0
        staged_sign = staged_raw > 0
        candidate_right = np.array_equal(candidate_sign, expected)
        staged_right = np.array_equal(staged_sign, expected)
        raw_equal = np.array_equal(candidate_raw, staged_raw, equal_nan=True)
        sign_equal = np.array_equal(candidate_sign, staged_sign)
        row["candidate_right"] += int(candidate_right)
        row["staged_right"] += int(staged_right)
        row["raw_bitwise_equal"] += int(raw_equal)
        row["sign_equal"] += int(sign_equal)
        if candidate_finite.all() and staged_finite.all():
            row["max_abs_raw_difference"] = max(
                float(row["max_abs_raw_difference"]),
                float(np.max(np.abs(candidate_raw.astype(np.float64) - staged_raw.astype(np.float64)), initial=0.0)),
            )
        if not (candidate_right and staged_right and raw_equal and sign_equal):
            row["first_failure"] = row["first_failure"] or {
                "index": index,
                "candidate_right": bool(candidate_right),
                "staged_right": bool(staged_right),
                "raw_bitwise_equal": bool(raw_equal),
                "sign_equal": bool(sign_equal),
                "candidate_vs_expected_different_cells": int(
                    np.count_nonzero(candidate_sign != expected)
                ),
                "candidate_vs_staged_different_values": int(
                    np.count_nonzero(candidate_raw != staged_raw)
                ),
                "height": len(example["input"]),
                "width": len(example["input"][0]),
            }
    executable = int(row["converted"]) - int(row["candidate_runtime_errors"])
    row["candidate_accuracy"] = (
        int(row["candidate_right"]) / executable if executable > 0 else None
    )
    row["perfect"] = bool(
        row["converted"] == row["total"]
        and row["conversion_errors"] == 0
        and row["candidate_right"] == row["total"]
        and row["staged_right"] == row["total"]
        and row["candidate_runtime_errors"] == 0
        and row["staged_runtime_errors"] == 0
        and row["one_sided_runtime_errors"] == 0
        and row["candidate_nonfinite_values"] == 0
        and row["staged_nonfinite_values"] == 0
        and row["raw_bitwise_equal"] == row["total"]
        and row["sign_equal"] == row["total"]
        and (not check_rule or row["readable_rule_right"] == row["total"])
    )
    return row


def known_examples() -> list[dict[str, Any]]:
    loaded = scoring.load_examples(192)
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in loaded.get(split, [])
    ]


def fresh_examples(seed: int, count: int) -> tuple[list[dict[str, Any]], int, int]:
    generator = importlib.import_module("task_7e0986d6")
    random.seed(seed)
    rows: list[dict[str, Any]] = []
    errors = 0
    attempts = 0
    while len(rows) < count and attempts < count * 10:
        attempts += 1
        try:
            rows.append(generator.generate())
        except Exception:  # noqa: BLE001
            errors += 1
    return rows, errors, attempts


def hash_snapshot(authority_task_data: bytes) -> dict[str, Any]:
    observed = {
        "authority_zip": sha256_path(AUTHORITY),
        "root_submission": sha256_path(ROOT_SUBMISSION),
        "all_scores": sha256_path(ALL_SCORES),
        "authority_task192": sha256_bytes(authority_task_data),
        "staged": sha256_path(STAGED),
        "candidate": sha256_path(CANDIDATE),
    }
    return {
        "observed": observed,
        "expected": EXPECTED,
        "all_match": observed == EXPECTED,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    candidate_data = CANDIDATE.read_bytes()
    staged_data = STAGED.read_bytes()
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task192.onnx")
    before = hash_snapshot(authority_data)
    if not before["all_match"]:
        raise RuntimeError(f"immutable input hash mismatch: {before}")

    print("structural and official scoring", flush=True)
    structural = structural_audit(candidate_data)
    official = {
        "immutable_8009_46": official_score(authority_data, "immutable_8009_46"),
        "staged_argmax_onehot": official_score(staged_data, "staged_argmax_onehot"),
        "candidate_hardsigmoid_k31": official_score(candidate_data, "candidate_k31"),
    }

    known = known_examples()
    first_benchmark = scoring.convert_to_numpy(known[0])
    if first_benchmark is None:
        raise RuntimeError("first known example did not convert")
    shape_trace = runtime_shape_trace(candidate_data, first_benchmark["input"])
    staged_shape_trace = runtime_shape_trace(staged_data, first_benchmark["input"])
    known_stats = color_stats(known)
    known_four: dict[str, Any] = {}
    for disable, threads, label in CONFIGS:
        print(f"known config {label}", flush=True)
        known_four[label] = run_cases(
            known,
            make_session(candidate_data, disable, threads),
            make_session(staged_data, disable, threads),
            check_rule=True,
        )

    fresh_rows = []
    for seed in FRESH_SEEDS:
        print(f"fresh seed {seed}: generating", flush=True)
        examples, generation_errors, attempts = fresh_examples(seed, FRESH_PER_SEED)
        comparison = run_cases(
            examples,
            make_session(candidate_data, True, 1),
            make_session(staged_data, True, 1),
            check_rule=True,
            progress_label=f"fresh seed {seed}",
        )
        stats = color_stats(examples)
        fresh_rows.append(
            {
                "seed": seed,
                "requested": FRESH_PER_SEED,
                "generated": len(examples),
                "generation_attempts": attempts,
                "generation_errors": generation_errors,
                "color_count_statistics": stats,
                "comparison": comparison,
                "perfect": bool(
                    len(examples) == FRESH_PER_SEED
                    and generation_errors == 0
                    and stats["threshold_condition_perfect"]
                    and comparison["perfect"]
                ),
            }
        )

    after = hash_snapshot(authority_data)
    reasons: list[str] = []
    if not structural["pass"]:
        reasons.append("structural_gate_failed")
    expected_costs = {
        "immutable_8009_46": 1609,
        "staged_argmax_onehot": 1149,
        "candidate_hardsigmoid_k31": 1138,
    }
    for label, expected_cost in expected_costs.items():
        row = official[label]
        if row is None or not row.get("correct") or int(row.get("cost", -1)) != expected_cost:
            reasons.append(f"official_score_failed_{label}")
    if not shape_trace["truthful"]:
        reasons.append("candidate_runtime_shapes_not_truthful")
    if not staged_shape_trace["truthful"]:
        reasons.append("staged_runtime_shapes_not_truthful")
    if not known_stats["threshold_condition_perfect"]:
        reasons.append("known_threshold_condition_failed")
    if not all(row["perfect"] for row in known_four.values()):
        reasons.append("known_four_not_raw_exact_perfect")
    if not all(row["perfect"] for row in fresh_rows):
        reasons.append("fresh_two_seed_not_raw_exact_perfect")
    if not before["all_match"] or not after["all_match"] or before != after:
        reasons.append("immutable_hash_changed")

    candidate_cost = int(official["candidate_hardsigmoid_k31"]["cost"])
    staged_cost = int(official["staged_argmax_onehot"]["cost"])
    immutable_cost = int(official["immutable_8009_46"]["cost"])
    result = {
        "status": "ACCEPT_INDEPENDENT_AUDIT" if not reasons else "REJECT_FAIL_CLOSED",
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "policy": {
            "known_configs": [label for _, _, label in CONFIGS],
            "fresh_seeds": list(FRESH_SEEDS),
            "fresh_per_seed": FRESH_PER_SEED,
            "fresh_required_accuracy": 1.0,
            "runtime_errors_allowed": 0,
            "nonfinite_values_allowed": 0,
            "raw_bitwise_equality_to_staged_required": True,
            "sign_equality_to_staged_required": True,
            "fail_closed": True,
        },
        "inputs": {
            "authority_zip": str(AUTHORITY.relative_to(ROOT)),
            "authority_member": "task192.onnx",
            "staged": str(STAGED.relative_to(ROOT)),
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "hashes_before": before,
            "hashes_after": after,
        },
        "structural": structural,
        "official_scores": official,
        "cost_comparison": {
            "immutable_cost": immutable_cost,
            "staged_cost": staged_cost,
            "candidate_cost": candidate_cost,
            "candidate_reduction_vs_immutable": immutable_cost - candidate_cost,
            "candidate_reduction_vs_staged": staged_cost - candidate_cost,
            "projected_gain_vs_immutable": math.log(immutable_cost / candidate_cost),
            "projected_gain_vs_staged": math.log(staged_cost / candidate_cost),
        },
        "candidate_runtime_shape_trace": shape_trace,
        "staged_runtime_shape_trace": staged_shape_trace,
        "known_color_count_statistics": known_stats,
        "known_four_configs": known_four,
        "fresh_two_independent_seeds": fresh_rows,
        "semantic_assessment": {
            "candidate_selector": (
                "HardSigmoid(hist, alpha=1, beta=-31): integer count <=31 -> 0; "
                "integer count >=32 -> 1"
            ),
            "staged_selector": "ArgMax(hist) followed by OneHot(depth=10, values=[0,1])",
            "equivalence_scope": (
                "Exact for every input whose nonzero-color histogram has exactly one "
                "channel >=32 and all other channels <=31. The audit tests and records "
                "that condition on all known and both independent 5000-case streams."
            ),
            "not_a_universal_input_identity": True,
            "hard_sigmoid_not_hardmax": True,
            "lookup_or_private_table": False,
        },
        "root_or_other_stage_modified": False,
    }
    output = HERE / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={result['status']} costs={immutable_cost}->{staged_cost}->{candidate_cost} "
        f"known4={all(row['perfect'] for row in known_four.values())} "
        f"fresh={[row['comparison']['candidate_right'] for row in fresh_rows]} "
        f"raw={[row['comparison']['raw_bitwise_equal'] for row in fresh_rows]}",
        flush=True,
    )
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())

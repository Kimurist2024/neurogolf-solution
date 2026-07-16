#!/usr/bin/env python3
"""Try cost-neutral finite repairs for the exact task379 cost-1947 model.

The lane is non-promoting.  It writes only below its own directory and proves
why exact reciprocal rescaling cannot remove the final float16 overflow when
the mathematical output itself is outside float16's finite range.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8005.17.zip"
SOURCE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_algebraic20_83/candidates/"
    "task379_exact_initializer_absorption.onnx"
)
CANDIDATES = HERE / "candidates"
TASK = 379
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
CONFIGS = (
    ("disable_all_t1", "disable_all", 1),
    ("disable_all_t4", "disable_all", 4),
    ("default_t1", "default", 1),
    ("default_t4", "default", 4),
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def equation(node: onnx.NodeProto) -> str:
    return next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")


def set_equation(node: onnx.NodeProto, text: str) -> None:
    next(attr for attr in node.attribute if attr.name == "equation").s = text.encode("ascii")


def array_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def replace_array(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for item in model.graph.initializer:
        if item.name == name:
            item.CopyFrom(numpy_helper.from_array(np.asarray(value), name))
            return
    raise KeyError(name)


def value_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def make_session(
    model: onnx.ModelProto,
    optimization: str,
    threads: int,
) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected graph")
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_cases() -> list[dict[str, np.ndarray]]:
    raw = scoring.load_examples(TASK)
    result: list[dict[str, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for example in raw.get(split, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                result.append(converted)
    return result


def reference_outputs(
    authority: onnx.ModelProto,
    cases: list[dict[str, np.ndarray]],
) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    for label, optimization, threads in CONFIGS:
        runtime = make_session(authority, optimization, threads)
        result[label] = [
            runtime.run(
                [runtime.get_outputs()[0].name],
                {runtime.get_inputs()[0].name: case["input"]},
            )[0]
            for case in cases
        ]
    return result


def known_four(
    model: onnx.ModelProto,
    cases: list[dict[str, np.ndarray]],
    references: dict[str, list[np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, optimization, threads in CONFIGS:
        row: dict[str, Any] = {
            "total": len(cases),
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "raw_sign_equal_to_authority": 0,
            "raw_equal_to_authority": 0,
            "nonfinite_values": 0,
            "nan_values": 0,
            "positive_infinity_values": 0,
            "negative_infinity_values": 0,
            "near_positive_values_0_to_0_25": 0,
            "near_positive_examples": 0,
            "min_positive": None,
            "max_finite_raw_difference": 0.0,
            "first_failure": None,
        }
        try:
            runtime = make_session(model, optimization, threads)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_errors"] = len(cases) or 1
            row["pass"] = False
            result[label] = row
            continue
        positives: list[float] = []
        for index, (case, reference) in enumerate(zip(cases, references[label]), start=1):
            try:
                raw = runtime.run(
                    [runtime.get_outputs()[0].name],
                    {runtime.get_inputs()[0].name: case["input"]},
                )[0]
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
                continue
            correct = np.array_equal(raw > 0.0, case["output"].astype(bool))
            row["right" if correct else "wrong"] += 1
            sign_equal = np.array_equal(raw > 0.0, reference > 0.0)
            row["raw_sign_equal_to_authority"] += int(sign_equal)
            row["raw_equal_to_authority"] += int(
                np.array_equal(raw, reference, equal_nan=True)
            )
            if (not correct or not sign_equal) and row["first_failure"] is None:
                row["first_failure"] = {"case": index, "kind": "sign_mismatch"}
            finite = np.isfinite(raw)
            row["nonfinite_values"] += int((~finite).sum())
            row["nan_values"] += int(np.isnan(raw).sum())
            row["positive_infinity_values"] += int(np.isposinf(raw).sum())
            row["negative_infinity_values"] += int(np.isneginf(raw).sum())
            near = finite & (raw > 0.0) & (raw < 0.25)
            near_count = int(near.sum())
            row["near_positive_values_0_to_0_25"] += near_count
            row["near_positive_examples"] += int(near_count > 0)
            positive = raw[finite & (raw > 0.0)]
            if positive.size:
                positives.append(float(positive.min()))
            jointly_finite = finite & np.isfinite(reference)
            if np.any(jointly_finite):
                difference = np.abs(
                    raw[jointly_finite].astype(np.float64)
                    - reference[jointly_finite].astype(np.float64)
                )
                row["max_finite_raw_difference"] = max(
                    float(row["max_finite_raw_difference"]),
                    float(difference.max(initial=0.0)),
                )
        row["min_positive"] = min(positives) if positives else None
        row["pass"] = bool(
            row["right"] == row["total"]
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
            and row["raw_sign_equal_to_authority"] == row["total"]
            and row["nonfinite_values"] == 0
            and row["near_positive_values_0_to_0_25"] == 0
        )
        result[label] = row
    return result


def runtime_shapes(model: onnx.ModelProto, optimization: str) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
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
    options.log_severity_level = 4
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    runtime = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    case = known_cases()[0]
    values = runtime.run(names, {runtime.get_inputs()[0].name: case["input"]})
    mismatches = [
        {
            "tensor": name,
            "declared": value_shape(typed[name]),
            "runtime": list(np.asarray(value).shape),
        }
        for name, value in zip(names, values)
        if value_shape(typed[name]) != list(np.asarray(value).shape)
    ]
    return {
        "optimization": optimization,
        "traced_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def structure(model: onnx.ModelProto) -> dict[str, Any]:
    errors: list[str] = []
    try:
        onnx.checker.check_model(model, full_check=True)
        checker = True
    except Exception as exc:  # noqa: BLE001
        checker = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        strict = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        strict = False
        errors.append(f"shape:{type(exc).__name__}:{exc}")
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    shape_rows: list[dict[str, Any]] = []
    for optimization in ("disable_all", "default"):
        try:
            shape_rows.append(runtime_shapes(model, optimization))
        except Exception as exc:  # noqa: BLE001
            shape_rows.append(
                {
                    "optimization": optimization,
                    "truthful": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    domains = {item.domain for item in model.opset_import} | {
        node.domain for node in model.graph.node
    }
    nested = [
        node.op_type
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    banned = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper() in BANNED or "SEQUENCE" in node.op_type.upper()
    ]
    init_arrays = array_map(model).values()
    checks = {
        "checker_full": checker,
        "strict_shape_data_prop": strict,
        "static_positive_shapes": all(
            value_shape(value) is not None
            for value in values
            if value.type.HasField("tensor_type")
        ),
        "truthful_runtime_shapes_both_ort": all(row.get("truthful") for row in shape_rows),
        "canonical_io": (
            len(model.graph.input) == 1
            and len(model.graph.output) == 1
            and model.graph.input[0].name == "input"
            and model.graph.output[0].name == "output"
            and value_shape(model.graph.input[0]) == [1, 10, 30, 30]
            and value_shape(model.graph.output[0]) == [1, 10, 30, 30]
        ),
        "standard_domains": all(domain in ("", "ai.onnx") for domain in domains),
        "no_banned_ops": not banned,
        "no_nested_functions_sparse": (
            not nested and not model.functions and not model.graph.sparse_initializer
        ),
        "conv_bias_ub0": not check_conv_bias(model),
        "finite_initializers": all(
            array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
            for array in init_arrays
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "errors": errors,
        "runtime_shape_evidence": shape_rows,
        "conv_bias_findings": check_conv_bias(model),
        "banned_ops": banned,
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    }


def measured_cost(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def reorder_final(model: onnx.ModelProto, order: list[int]) -> None:
    node = model.graph.node[-1]
    if node.op_type != "Einsum":
        raise ValueError("final node is not Einsum")
    lhs, rhs = equation(node).split("->", 1)
    terms = lhs.split(",")
    inputs = list(node.input)
    if sorted(order) != list(range(len(inputs))):
        raise ValueError("not a permutation")
    del node.input[:]
    node.input.extend(inputs[index] for index in order)
    set_equation(node, ",".join(terms[index] for index in order) + "->" + rhs)


def build_variants(source: onnx.ModelProto) -> list[dict[str, Any]]:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    base_arrays = array_map(source)
    variants: list[tuple[str, onnx.ModelProto, dict[str, Any]]] = []
    variants.append(
        (
            "source_exact_absorption",
            copy.deepcopy(source),
            {
                "family": "source",
                "all_input_real_algebra_equal": True,
                "proof": "Exact QRow1_2 = einsum('bca->a', WBasis) substitution.",
            },
        )
    )
    for exponent in (1, 2, 3):
        candidate = copy.deepcopy(source)
        replace_array(
            candidate,
            "WBasis",
            (base_arrays["WBasis"] * (2**exponent)).astype(np.float16),
        )
        replace_array(
            candidate,
            "CWCoef",
            (base_arrays["CWCoef"] / (2 ** (2 * exponent))).astype(np.float16),
        )
        variants.append(
            (
                f"reciprocal_wbasis_x{2**exponent}_cwcoef_div{2**(2*exponent)}",
                candidate,
                {
                    "family": "power_of_two_reciprocal",
                    "wbasis_occurrences_in_final_einsum": 2,
                    "cwcoef_occurrences_in_final_einsum": 1,
                    "wbasis_scale": float(2**exponent),
                    "cwcoef_scale": float(2 ** (-2 * exponent)),
                    "total_real_scale": 1.0,
                    "all_input_real_algebra_equal": True,
                    "proof": "(2^k)^2 * 2^(-2k) = 1 for every product term.",
                },
            )
        )
    operand_count = len(source.graph.node[-1].input)
    candidate = copy.deepcopy(source)
    reorder_final(candidate, list(reversed(range(operand_count))))
    variants.append(
        (
            "exact_operand_order_reverse",
            candidate,
            {
                "family": "contraction_order",
                "permutation": "reverse",
                "all_input_real_algebra_equal": True,
                "proof": "The Einsum operand list and corresponding LHS terms are permuted together.",
            },
        )
    )
    candidate = copy.deepcopy(source)
    connected_first = [23, 24, 25, 26, 27] + [
        index for index in range(operand_count) if index not in {23, 24, 25, 26, 27}
    ]
    reorder_final(candidate, connected_first)
    variants.append(
        (
            "exact_operand_order_cw_component_first",
            candidate,
            {
                "family": "contraction_order",
                "permutation": connected_first,
                "all_input_real_algebra_equal": True,
                "proof": "The Einsum operand list and corresponding LHS terms are permuted together.",
            },
        )
    )
    candidate = copy.deepcopy(source)
    replace_array(candidate, "CWCoef", (base_arrays["CWCoef"] / 2).astype(np.float16))
    variants.append(
        (
            "diagnostic_uniform_cwcoef_div2",
            candidate,
            {
                "family": "non_exact_diagnostic",
                "real_output_scale": 0.5,
                "all_input_real_algebra_equal": False,
                "all_input_real_sign_equal": True,
                "rejection": "Changes raw algebra and moves positive 0.25 to forbidden 0.125.",
            },
        )
    )
    candidate = copy.deepcopy(source)
    changed = base_arrays["CWCoef"].copy()
    changed[changed == np.float16(-32768)] = np.float16(-16384)
    replace_array(candidate, "CWCoef", changed)
    variants.append(
        (
            "diagnostic_sentinel_minus16384",
            candidate,
            {
                "family": "non_exact_diagnostic",
                "changed_coefficients": 2,
                "from": -32768.0,
                "to": -16384.0,
                "all_input_real_algebra_equal": False,
                "rejection": "Finite and known-safe, but changes two real coefficients.",
            },
        )
    )
    result: list[dict[str, Any]] = []
    for label, model, proof in variants:
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        path = CANDIDATES / f"task379_{label}.onnx"
        onnx.save(model, path)
        result.append({"label": label, "model": model, "path": path, "algebra": proof})
    return result


def trace_overflow(model: onnx.ModelProto, case: dict[str, np.ndarray]) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    existing_outputs = {value.name for value in traced.graph.output}
    appended: list[tuple[int, str, str]] = []
    for index, node in enumerate(traced.graph.node):
        for name in node.output:
            if name and name in typed and name not in existing_outputs:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                existing_outputs.add(name)
                appended.append((index, node.op_type, name))
    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runtime = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    runtime_values = runtime.run(None, {runtime.get_inputs()[0].name: case["input"]})
    original_output = np.asarray(runtime_values[0])
    dynamic = {
        name: np.asarray(value)
        for (_, _, name), value in zip(appended, runtime_values[1:])
    }
    runtime_map = {model.graph.output[0].name: original_output, **dynamic}
    node_rows: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        for name in node.output:
            if name not in runtime_map:
                continue
            value = runtime_map[name]
            finite = value[np.isfinite(value)]
            node_rows.append(
                {
                    "node_index": index,
                    "op_type": node.op_type,
                    "tensor": name,
                    "shape": list(value.shape),
                    "nonfinite": int((~np.isfinite(value)).sum()),
                    "nan": int(np.isnan(value).sum()),
                    "positive_inf": int(np.isposinf(value).sum()),
                    "negative_inf": int(np.isneginf(value).sum()),
                    "finite_min": float(finite.min()) if finite.size else None,
                    "finite_max": float(finite.max()) if finite.size else None,
                }
            )
    final = model.graph.node[-1]
    constants = array_map(model)
    operands = [constants.get(name, dynamic.get(name)) for name in final.input]
    if any(value is None for value in operands):
        missing = [name for name, value in zip(final.input, operands) if value is None]
        raise RuntimeError(f"missing final operands: {missing}")
    exact = np.einsum(
        equation(final),
        *[np.asarray(value, dtype=np.float64) for value in operands],
        optimize="greedy",
    )
    outside = np.abs(exact) > np.finfo(np.float16).max
    coordinates = np.argwhere(outside)
    first_nonfinite = next((row for row in node_rows if row["nonfinite"]), None)
    dynamic_final = {
        name: {
            "shape": list(dynamic[name].shape),
            "elements": int(dynamic[name].size),
            "dtype": str(dynamic[name].dtype),
        }
        for name in ("B", "O", "Hs", "ac")
    }
    dynamic_elements = sum(row["elements"] for row in dynamic_final.values())
    return {
        "first_nonfinite_node": first_nonfinite,
        "all_pre_final_node_outputs_finite": all(
            row["nonfinite"] == 0 for row in node_rows if row["node_index"] < len(model.graph.node) - 1
        ),
        "node_output_stats": node_rows,
        "final_einsum_equation": equation(final),
        "final_output_dtype": TensorProto.DataType.Name(
            model.graph.output[0].type.tensor_type.elem_type
        ),
        "float16_finite_limit": float(np.finfo(np.float16).max),
        "float64_real_evaluation": {
            "minimum": float(exact.min()),
            "maximum": float(exact.max()),
            "outside_float16_finite_range": int(outside.sum()),
            "first_outside_coordinates_and_values": [
                {"coordinate": coordinate.tolist(), "value": float(exact[tuple(coordinate)])}
                for coordinate in coordinates[:20]
            ],
        },
        "runtime_final": {
            "negative_inf": int(np.isneginf(original_output).sum()),
            "positive_inf": int(np.isposinf(original_output).sum()),
            "nan": int(np.isnan(original_output).sum()),
        },
        "dynamic_final_operands": dynamic_final,
        "minimum_float32_promotion_bound": {
            "dynamic_elements_requiring_common_Einsum_dtype": dynamic_elements,
            "additional_bytes_f16_to_f32": dynamic_elements * 2,
            "current_cost": 1947,
            "lower_bound_cost": 1947 + dynamic_elements * 2,
            "reason": (
                "ONNX Einsum has one tensor type T for all inputs and output; at minimum "
                "B/O/Hs/ac must widen from counted float16 to float32."
            ),
        },
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    HERE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    authority_zip_sha = digest(AUTHORITY_ZIP)
    source_sha = digest(SOURCE)
    expected_source_sha = "854c63d966310949803391cf4c019b02a9c0f2a53578257fee5898386e53cf64"
    if source_sha != expected_source_sha:
        raise RuntimeError(f"source SHA mismatch: {source_sha}")
    source = onnx.load(SOURCE)
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_data = archive.read("task379.onnx")
    authority = onnx.load_model_from_string(authority_data)
    cases = known_cases()
    references = reference_outputs(authority, cases)
    variants = build_variants(source)
    overflow = trace_overflow(source, cases[0])
    with tempfile.TemporaryDirectory(prefix="task379_finite85_", dir=HERE) as tmp:
        authority_path = Path(tmp) / "task379.onnx"
        authority_path.write_bytes(authority_data)
        authority_cost = measured_cost(authority_path)
    rows: list[dict[str, Any]] = []
    for item in variants:
        known = known_four(item["model"], cases, references)
        static = structure(item["model"])
        cost = measured_cost(item["path"])
        numeric_pass = all(row.get("pass") for row in known.values())
        cost_pass = bool(0 < cost["cost"] <= 1947 and cost["cost"] < authority_cost["cost"])
        accepted = bool(
            item["algebra"].get("all_input_real_algebra_equal")
            and numeric_pass
            and static["pass"]
            and cost_pass
        )
        reasons: list[str] = []
        if not item["algebra"].get("all_input_real_algebra_equal"):
            reasons.append("not_all_input_real_algebra_equal")
        if not numeric_pass:
            if any(row.get("nonfinite_values", 0) for row in known.values()):
                reasons.append("nonfinite_output")
            if any(row.get("near_positive_values_0_to_0_25", 0) for row in known.values()):
                reasons.append("near_positive_output")
            if any(row.get("runtime_errors", 0) for row in known.values()):
                reasons.append("runtime_error")
            if any(
                row.get("right", 0) != row.get("total", -1)
                or row.get("raw_sign_equal_to_authority", 0) != row.get("total", -1)
                for row in known.values()
            ):
                reasons.append("raw_sign_or_gold_mismatch")
        if not static["pass"]:
            reasons.extend(name for name, passed in static["checks"].items() if not passed)
        if not cost_pass:
            reasons.append("cost_above_1947_or_not_below_authority")
        rows.append(
            {
                "label": item["label"],
                "path": str(item["path"].relative_to(ROOT)),
                "sha256": digest(item["path"]),
                "cost": cost,
                "algebra": item["algebra"],
                "known_complete_four_configs": known,
                "structure": static,
                "accepted": accepted,
                "reasons": sorted(set(reasons)),
            }
        )
        compact = known["disable_all_t1"]
        print(
            f"{item['label']}: cost={cost['cost']} correct={compact['right']}/{compact['total']} "
            f"nonfinite={compact['nonfinite_values']} near={compact['near_positive_values_0_to_0_25']} "
            f"exact={item['algebra'].get('all_input_real_algebra_equal')} accepted={accepted}",
            flush=True,
        )
    accepted = [row for row in rows if row["accepted"]]
    result = {
        "task": TASK,
        "authority": {
            "archive": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "archive_sha256": authority_zip_sha,
            "member": "task379.onnx",
            "member_sha256": digest_bytes(authority_data),
            "cost": authority_cost,
        },
        "source_exact_candidate": {
            "path": str(SOURCE.relative_to(ROOT)),
            "sha256": source_sha,
            "cost": {"memory": 1570, "params": 377, "cost": 1947},
        },
        "policy": {
            "maximum_cost": 1947,
            "strictly_below_authority": True,
            "all_input_real_algebra_equal": True,
            "known_complete_four_configs": True,
            "raw_sign_equal_to_authority": True,
            "runtime_errors": 0,
            "nonfinite_values": 0,
            "near_positive_values_open_interval_0_0_25": 0,
            "strict_shape_data_prop": True,
            "truthful_runtime_shapes": True,
            "conv_bias_ub": 0,
        },
        "overflow_root_cause": overflow,
        "variants": rows,
        "accepted": accepted,
        "accepted_count": len(accepted),
        "aggregate_gain": sum(
            math.log(authority_cost["cost"] / row["cost"]["cost"]) for row in accepted
        ),
        "impossibility_conclusion": {
            "exact_finite_float16_repair_at_cost_1947": False,
            "reason": (
                "A known valid input has exact real final values below -65504. "
                "Every algebra-identical float16 final output must therefore be nonfinite; "
                "widening the common Einsum type has a cost lower bound above 1947."
            ),
            "finite_nonexact_control": "diagnostic_sentinel_minus16384",
        },
        "protected_files_modified": [],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    winner_manifest = {
        "authority": str(AUTHORITY_ZIP.relative_to(ROOT)),
        "authority_sha256": authority_zip_sha,
        "winners": [
            {
                "task": TASK,
                "path": row["path"],
                "sha256": row["sha256"],
                "cost": row["cost"]["cost"],
                "gain": math.log(authority_cost["cost"] / row["cost"]["cost"]),
            }
            for row in accepted
        ],
        "merge_performed": False,
    }
    (HERE / "winner_manifest.json").write_text(
        json.dumps(winner_manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "variants": len(rows),
                "accepted": [row["label"] for row in accepted],
                "overflow_minimum": overflow["float64_real_evaluation"]["minimum"],
                "float16_limit": overflow["float16_finite_limit"],
                "f32_lower_bound_cost": overflow["minimum_float32_promotion_bound"][
                    "lower_bound_cost"
                ],
                "result": str((HERE / "result.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

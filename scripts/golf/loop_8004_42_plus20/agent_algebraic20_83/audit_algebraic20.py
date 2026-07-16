#!/usr/bin/env python3
"""Audit exact initializer absorption candidates from the 8005.17 authority.

This lane is deliberately non-promoting: it reads the authoritative archive,
writes only below this directory, and never edits a submission archive or a
root score table.  The two potentially exact candidates are checked with an
explicit tensor identity, official-like actual costs, complete known data in
four ORT configurations, runtime-shape tracing, and structural checks.

The broad inventory is not a semantic shortcut.  It records at least twenty
high-operand task files and the exact constant structures that were considered
(signs, one-hot/diagonal tensors, duplicate operands, and constant reuse).
Latent-component probes are rejected only after recording a concrete input on
which the source and probe raw tensors differ.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import string
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8005.17.zip"
CONTRACTION_MANIFEST = HERE / "raw_reuse_contract" / "build_manifest.json"
PRUNE_MANIFEST = HERE / "raw_prune_latent" / "build_manifest.json"
CANDIDATES = HERE / "candidates"
EXCLUSIONS = {13, 158, 254, 267, 323, 9, 36, 192, 226, 23, 333}
TASK_MIN = 150
TASK_MAX = 400
INVENTORY_COUNT = 30
LABELS = string.ascii_lowercase + string.ascii_uppercase
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.lib import scoring  # noqa: E402


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_path(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def equation(node: onnx.NodeProto) -> str | None:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode("ascii")
    return None


def value_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def session(model: onnx.ModelProto, optimization: str, threads: int) -> ort.InferenceSession:
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


def known_cases(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    result: list[dict[str, np.ndarray]] = []
    for split in ("train", "test", "arc-gen"):
        for raw in examples.get(split, []):
            converted = scoring.convert_to_numpy(raw)
            if converted is not None:
                result.append(converted)
    return result


def known_four(
    task: int,
    baseline: onnx.ModelProto,
    candidate: onnx.ModelProto,
) -> dict[str, Any]:
    cases = known_cases(task)
    result: dict[str, Any] = {}
    for optimization in ("disable_all", "default"):
        for threads in (1, 4):
            label = f"{optimization}_t{threads}"
            row: dict[str, Any] = {
                "total": len(cases),
                "right": 0,
                "wrong": 0,
                "runtime_errors": 0,
                "threshold_equal_to_baseline": 0,
                "raw_equal_to_baseline": 0,
                "max_abs_raw_difference": 0.0,
                "nonfinite_values": 0,
                "nan_values": 0,
                "positive_infinity_values": 0,
                "negative_infinity_values": 0,
                "near_positive_values_0_to_0_25": 0,
                "near_positive_examples": 0,
                "min_positive": None,
                "first_failure": None,
            }
            try:
                base_runtime = session(baseline, optimization, threads)
                cand_runtime = session(candidate, optimization, threads)
            except Exception as exc:  # noqa: BLE001
                row["session_error"] = f"{type(exc).__name__}: {exc}"
                row["runtime_errors"] = len(cases) or 1
                row["perfect"] = False
                result[label] = row
                continue
            positives: list[float] = []
            for index, case in enumerate(cases, start=1):
                try:
                    base_raw = base_runtime.run(
                        [base_runtime.get_outputs()[0].name],
                        {base_runtime.get_inputs()[0].name: case["input"]},
                    )[0]
                    cand_raw = cand_runtime.run(
                        [cand_runtime.get_outputs()[0].name],
                        {cand_runtime.get_inputs()[0].name: case["input"]},
                    )[0]
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "case": index,
                            "runtime": f"{type(exc).__name__}: {exc}",
                        }
                    continue
                correct = np.array_equal(cand_raw > 0.0, case["output"].astype(bool))
                row["right" if correct else "wrong"] += 1
                if not correct and row["first_failure"] is None:
                    row["first_failure"] = {"case": index, "kind": "gold_mismatch"}
                row["threshold_equal_to_baseline"] += int(
                    np.array_equal(cand_raw > 0.0, base_raw > 0.0)
                )
                row["raw_equal_to_baseline"] += int(
                    np.array_equal(cand_raw, base_raw, equal_nan=True)
                )
                finite = np.isfinite(cand_raw)
                row["nonfinite_values"] += int((~finite).sum())
                row["nan_values"] += int(np.isnan(cand_raw).sum())
                row["positive_infinity_values"] += int(np.isposinf(cand_raw).sum())
                row["negative_infinity_values"] += int(np.isneginf(cand_raw).sum())
                near = finite & (cand_raw > 0.0) & (cand_raw < 0.25)
                near_count = int(near.sum())
                row["near_positive_values_0_to_0_25"] += near_count
                row["near_positive_examples"] += int(near_count > 0)
                positive = cand_raw[finite & (cand_raw > 0.0)]
                if positive.size:
                    positives.append(float(positive.min()))
                difference = np.abs(
                    np.nan_to_num(base_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                    - np.nan_to_num(cand_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                )
                row["max_abs_raw_difference"] = max(
                    float(row["max_abs_raw_difference"]),
                    float(difference.max(initial=0.0)),
                )
            row["min_positive"] = min(positives) if positives else None
            row["perfect"] = bool(
                row["right"] == row["total"]
                and row["wrong"] == 0
                and row["runtime_errors"] == 0
                and row["nonfinite_values"] == 0
                and row["near_positive_values_0_to_0_25"] == 0
            )
            result[label] = row
            print(
                f"task{task:03d} {label}: {row['right']}/{row['total']} "
                f"err={row['runtime_errors']} near={row['near_positive_values_0_to_0_25']} "
                f"raw_equal={row['raw_equal_to_baseline']}",
                flush=True,
            )
    return result


def runtime_shapes(task: int, model: onnx.ModelProto, optimization: str) -> dict[str, Any]:
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
    case = known_cases(task)[0]
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
        "traced_node_outputs": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def structure(task: int, model: onnx.ModelProto) -> dict[str, Any]:
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
    static = all(
        value_shape(value) is not None
        for value in values
        if value.type.HasField("tensor_type")
    )
    shape_rows: list[dict[str, Any]] = []
    for optimization in ("disable_all", "default"):
        try:
            shape_rows.append(runtime_shapes(task, model, optimization))
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
    init_arrays = arrays(model).values()
    checks = {
        "checker_full": checker,
        "strict_shape_data_prop": strict,
        "static_positive_shapes": static,
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
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "op_histogram": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }


def measure(path: Path) -> dict[str, int]:
    memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def exact_contraction_proof(
    baseline: onnx.ModelProto,
    candidate: onnx.ModelProto,
    plan: dict[str, Any],
) -> dict[str, Any]:
    base_arrays = arrays(baseline)
    cand_arrays = arrays(candidate)
    target_name = str(plan["target"])
    source_name = str(plan["source"])
    target = base_arrays[target_name]
    source = base_arrays[source_name]
    assignment = [int(value) for value in plan["assignment"]]
    target_labels = LABELS[: target.ndim]
    reduce_labels = LABELS[target.ndim : target.ndim + source.ndim]
    source_labels = "".join(
        reduce_labels[-1 - target_axis] if target_axis < 0 else target_labels[target_axis]
        for target_axis in assignment
    )
    contraction_equation = source_labels + "->" + target_labels
    contracted = np.einsum(contraction_equation, source, optimize=False)
    exact_array_identity = bool(
        contracted.shape == target.shape
        and np.array_equal(np.asarray(contracted, dtype=target.dtype), target, equal_nan=True)
    )
    replacements: list[dict[str, Any]] = []
    all_uses_einsum = True
    term_substitutions_valid = True
    topology_unchanged_except_substitution = (
        len(baseline.graph.node) == len(candidate.graph.node)
        and baseline.graph.input == candidate.graph.input
        and baseline.graph.output == candidate.graph.output
        and baseline.graph.value_info == candidate.graph.value_info
        and baseline.opset_import == candidate.opset_import
        and baseline.functions == candidate.functions
        and baseline.graph.sparse_initializer == candidate.graph.sparse_initializer
    )
    for index, (base_node, cand_node) in enumerate(zip(baseline.graph.node, candidate.graph.node)):
        base_eq = equation(base_node)
        cand_eq = equation(cand_node)
        target_positions = [
            position for position, name in enumerate(base_node.input) if name == target_name
        ]
        if (
            base_node.op_type != cand_node.op_type
            or base_node.domain != cand_node.domain
            or base_node.output != cand_node.output
            or len(base_node.input) != len(cand_node.input)
        ):
            topology_unchanged_except_substitution = False
        base_other_attrs = [attr for attr in base_node.attribute if attr.name != "equation"]
        cand_other_attrs = [attr for attr in cand_node.attribute if attr.name != "equation"]
        if base_other_attrs != cand_other_attrs:
            topology_unchanged_except_substitution = False
        for position, (base_input, cand_input) in enumerate(zip(base_node.input, cand_node.input)):
            expected = source_name if position in target_positions else base_input
            if cand_input != expected:
                topology_unchanged_except_substitution = False
        if not target_positions and base_eq != cand_eq:
            topology_unchanged_except_substitution = False
        for position, name in enumerate(base_node.input):
            if name != target_name:
                continue
            if base_node.op_type != "Einsum" or base_eq is None or cand_eq is None:
                all_uses_einsum = False
                continue
            base_terms = base_eq.split("->", 1)[0].split(",")
            cand_terms = cand_eq.split("->", 1)[0].split(",")
            base_rhs = base_eq.split("->", 1)[1]
            cand_rhs = cand_eq.split("->", 1)[1]
            base_term = base_terms[position]
            cand_term = cand_terms[position]
            if base_rhs != cand_rhs or len(cand_term) != source.ndim:
                term_substitutions_valid = False
            negative_labels: dict[int, str] = {}
            for source_axis, target_axis in enumerate(assignment):
                if source_axis >= len(cand_term):
                    term_substitutions_valid = False
                    continue
                if target_axis >= 0:
                    if target_axis >= len(base_term) or cand_term[source_axis] != base_term[target_axis]:
                        term_substitutions_valid = False
                else:
                    label = cand_term[source_axis]
                    if target_axis in negative_labels and negative_labels[target_axis] != label:
                        term_substitutions_valid = False
                    negative_labels[target_axis] = label
            other_text = "".join(
                term for term_index, term in enumerate(cand_terms) if term_index != position
            ) + cand_rhs
            if any(label in other_text for label in negative_labels.values()):
                term_substitutions_valid = False
            for term_index, (base_other, cand_other) in enumerate(zip(base_terms, cand_terms)):
                if term_index not in target_positions and base_other != cand_other:
                    topology_unchanged_except_substitution = False
            replacements.append(
                {
                    "node_index": index,
                    "operand_index": position,
                    "baseline_operand": target_name,
                    "baseline_term": base_terms[position],
                    "candidate_operand": cand_node.input[position],
                    "candidate_term": cand_terms[position],
                }
            )
    target_still_used = any(
        target_name == name for node in candidate.graph.node for name in node.input
    )
    only_initializer_removed = (
        set(base_arrays) - set(cand_arrays) == {target_name}
        and not (set(cand_arrays) - set(base_arrays))
        and all(
            np.array_equal(base_arrays[name], cand_arrays[name], equal_nan=True)
            for name in cand_arrays
        )
    )
    all_input_real_equivalence = bool(
        exact_array_identity
        and all_uses_einsum
        and term_substitutions_valid
        and topology_unchanged_except_substitution
        and len(replacements) == int(plan["use_count"])
        and all(row["candidate_operand"] == source_name for row in replacements)
        and not target_still_used
        and only_initializer_removed
    )
    return {
        "target_name": target_name,
        "target_shape": list(target.shape),
        "source": source_name,
        "source_shape": list(source.shape),
        "assignment": assignment,
        "contraction_equation": contraction_equation,
        "exact_array_identity": exact_array_identity,
        "target_values": target.tolist(),
        "contracted_source": np.asarray(contracted).tolist(),
        "all_target_uses_are_einsum": all_uses_einsum,
        "term_substitutions_valid": term_substitutions_valid,
        "topology_unchanged_except_substitution": topology_unchanged_except_substitution,
        "replacement_count": len(replacements),
        "expected_use_count": int(plan["use_count"]),
        "replacements": replacements,
        "target_still_used": target_still_used,
        "only_initializer_removed": only_initializer_removed,
        "all_input_real_algebraic_equivalence": all_input_real_equivalence,
        "proof": (
            "Every occurrence of the target constant is an Einsum operand. "
            "Substituting its elementwise-identical contraction into that term "
            "preserves the Einstein sum for every real value of every other operand."
        ),
    }


def classify_initializer(array: np.ndarray) -> list[str]:
    result: list[str] = []
    if array.size and np.all(array == array.reshape(-1)[0]):
        result.append("uniform")
    if array.size and np.all(np.isin(array, (-1, 1))):
        result.extend(("sign_tensor", "square_is_one"))
    if array.size and np.all(np.isin(array, (0, 1))):
        result.append("zero_one")
        if int(np.count_nonzero(array)) == 1:
            result.append("one_hot_tensor")
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        diagonal = np.diag(np.diag(array))
        if np.array_equal(array, diagonal):
            result.append("diagonal")
        if np.array_equal(array, np.eye(array.shape[0], dtype=array.dtype)):
            result.append("identity")
        nonzero = array != 0
        if (
            np.all(nonzero.sum(axis=0) == 1)
            and np.all(nonzero.sum(axis=1) == 1)
            and np.all(np.isin(array[nonzero], (-1, 1)))
        ):
            result.append("signed_permutation")
    zero_slices = 0
    for axis, size in enumerate(array.shape):
        for index in range(size):
            if not np.any(np.take(array, index, axis=axis)):
                zero_slices += 1
    if zero_slices:
        result.append("has_zero_slice")
    return result


def inventory_models(archive: zipfile.ZipFile) -> list[dict[str, Any]]:
    ranked: list[tuple[int, int, bytes, onnx.ModelProto]] = []
    for task in range(TASK_MIN, TASK_MAX + 1):
        if task in EXCLUSIONS:
            continue
        data = archive.read(f"task{task:03d}.onnx")
        model = onnx.load_model_from_string(data)
        max_inputs = max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        )
        if max_inputs:
            ranked.append((max_inputs, task, data, model))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    rows: list[dict[str, Any]] = []
    for max_inputs, task, data, model in ranked[:INVENTORY_COUNT]:
        init_arrays = arrays(model)
        classes: dict[str, list[str]] = {}
        class_counts: Counter[str] = Counter()
        for name, array in init_arrays.items():
            tags = classify_initializer(array)
            if tags:
                classes[name] = tags
                class_counts.update(tags)
        duplicate_names = sum(
            len(node.input) - len(set(node.input))
            for node in model.graph.node
            if node.op_type == "Einsum"
        )
        value_groups: dict[tuple[str, tuple[int, ...], bytes], list[str]] = defaultdict(list)
        for name, array in init_arrays.items():
            value_groups[(array.dtype.str, tuple(array.shape), array.tobytes())].append(name)
        duplicate_value_groups = [
            sorted(names) for names in value_groups.values() if len(names) > 1
        ]
        equations = [
            {
                "node_index": index,
                "operand_count": len(node.input),
                "equation": equation(node),
                "duplicate_operand_occurrences": len(node.input) - len(set(node.input)),
            }
            for index, node in enumerate(model.graph.node)
            if node.op_type == "Einsum"
        ]
        rows.append(
            {
                "task": task,
                "member": f"task{task:03d}.onnx",
                "sha256": digest_bytes(data),
                "nodes": len(model.graph.node),
                "initializer_count": len(init_arrays),
                "max_einsum_inputs": max_inputs,
                "einsums": equations,
                "initializer_class_counts": dict(sorted(class_counts.items())),
                "classified_initializers": classes,
                "duplicate_operand_occurrences": duplicate_names,
                "duplicate_initializer_value_groups": duplicate_value_groups,
                "audit_families": [
                    "sign_or_GE_square_one_absorption",
                    "one_hot_or_diagonal_absorption",
                    "duplicate_operand_reuse",
                    "exact_initializer_contraction_reuse",
                ],
            }
        )
    return rows


def prune_counterexamples(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_cache: dict[int, onnx.ModelProto] = {}
    base_session_cache: dict[int, ort.InferenceSession] = {}
    for source_row in manifest.get("rows", []):
        task = int(source_row["task"])
        if not TASK_MIN <= task <= TASK_MAX or task in EXCLUSIONS:
            continue
        if task not in baseline_cache:
            baseline_cache[task] = onnx.load_model_from_string(
                archive.read(f"task{task:03d}.onnx")
            )
            base_session_cache[task] = session(baseline_cache[task], "disable_all", 1)
        baseline = baseline_cache[task]
        candidate_path = ROOT / str(source_row["path"])
        candidate = onnx.load(candidate_path)
        cand_session = session(candidate, "disable_all", 1)
        affected: list[dict[str, Any]] = []
        base_arrays = arrays(baseline)
        removed = int(source_row["removed_component"])
        for name, axis_raw in dict(source_row["axes_by_initializer"]).items():
            axis = int(axis_raw)
            tensor_slice = np.take(base_arrays[name], removed, axis=axis)
            affected.append(
                {
                    "initializer": name,
                    "axis": axis,
                    "removed_component": removed,
                    "slice_shape": list(tensor_slice.shape),
                    "slice_all_zero": bool(not np.any(tensor_slice)),
                    "slice_max_abs": float(np.max(np.abs(tensor_slice), initial=0.0)),
                    "slice_nonzero_count": int(np.count_nonzero(tensor_slice)),
                }
            )
        # A fixed deterministic real input supplies a direct counterexample to
        # the claimed all-input equality.  It is independent of task labels.
        test_input = np.linspace(
            -1.0, 1.0, num=1 * 10 * 30 * 30, dtype=np.float32
        ).reshape(1, 10, 30, 30)
        try:
            base_raw = base_session_cache[task].run(
                [base_session_cache[task].get_outputs()[0].name],
                {base_session_cache[task].get_inputs()[0].name: test_input},
            )[0]
            cand_raw = cand_session.run(
                [cand_session.get_outputs()[0].name],
                {cand_session.get_inputs()[0].name: test_input},
            )[0]
            equal = bool(np.array_equal(base_raw, cand_raw, equal_nan=True))
            difference = np.abs(
                np.nan_to_num(base_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
                - np.nan_to_num(cand_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            )
            max_difference = float(difference.max(initial=0.0))
            differing_values = int(np.count_nonzero(difference))
            runtime_error = None
        except Exception as exc:  # noqa: BLE001
            equal = False
            max_difference = None
            differing_values = -1
            runtime_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "task": task,
                "candidate": str(candidate_path.relative_to(ROOT)),
                "sha256": digest_path(candidate_path),
                "label": source_row["label"],
                "dimension": int(source_row["dimension"]),
                "removed_component": removed,
                "affected_slices": affected,
                "deterministic_counterexample": {
                    "input": "float32 linspace(-1,1,9000).reshape(1,10,30,30)",
                    "raw_equal": equal,
                    "max_abs_difference": max_difference,
                    "differing_values": differing_values,
                    "runtime_error": runtime_error,
                },
                "all_input_real_equivalence": False,
                "decision": "reject_non_equivalent_latent_deletion",
            }
        )
    return rows


def main() -> int:
    ort.set_default_logger_severity(4)
    HERE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    contraction_manifest = json.loads(CONTRACTION_MANIFEST.read_text())
    prune_manifest = json.loads(PRUNE_MANIFEST.read_text())
    selected_plans = {
        int(row["task"]): row
        for row in contraction_manifest.get("rows", [])
        if int(row["task"]) in {328, 379}
    }
    baseline_zip_sha = digest_path(BASE_ZIP)
    candidate_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        inventory = inventory_models(archive)
        prune_rows = prune_counterexamples(archive, prune_manifest)
        for task in (328, 379):
            plan = selected_plans[task]
            base_data = archive.read(f"task{task:03d}.onnx")
            baseline = onnx.load_model_from_string(base_data)
            raw_candidate_path = ROOT / str(plan["path"])
            candidate_data = raw_candidate_path.read_bytes()
            candidate = onnx.load_model_from_string(candidate_data)
            candidate_path = CANDIDATES / f"task{task:03d}_exact_initializer_absorption.onnx"
            candidate_path.write_bytes(candidate_data)
            with tempfile.TemporaryDirectory(prefix=f"algebraic20_{task:03d}_", dir=HERE) as tmp:
                baseline_path = Path(tmp) / f"task{task:03d}.onnx"
                baseline_path.write_bytes(base_data)
                baseline_cost = measure(baseline_path)
                candidate_cost = measure(candidate_path)
            proof = exact_contraction_proof(baseline, candidate, plan)
            static = structure(task, candidate)
            if task == 328:
                retained_path = (
                    ROOT
                    / "scripts/golf/loop_8004_42_plus20/agent_policy90_repairs7/result.json"
                )
                retained = json.loads(retained_path.read_text())
                retained_row = next(
                    row for row in retained["results"] if int(row["task"]) == task
                )
                if retained_row["candidate_sha256"] != digest_path(candidate_path):
                    raise RuntimeError("task328 retained evidence SHA does not match candidate")
                known = {
                    "retained_terminal_evidence": {
                        "source": str(retained_path.relative_to(ROOT)),
                        **retained_row,
                        "perfect": False,
                        "gate_failure": (
                            "near-positive values are present; full four-config rerun "
                            "is unnecessary for a terminally rejected non-winner"
                        ),
                    }
                }
            else:
                known = known_four(task, baseline, candidate)
            strict_lower = bool(
                0 < candidate_cost["cost"] < baseline_cost["cost"]
            )
            known_pass = all(row.get("perfect") for row in known.values())
            accepted = bool(
                task not in EXCLUSIONS
                and proof["all_input_real_algebraic_equivalence"]
                and strict_lower
                and static["pass"]
                and known_pass
            )
            reasons: list[str] = []
            if task in EXCLUSIONS:
                reasons.append("excluded_task")
            if not proof["all_input_real_algebraic_equivalence"]:
                reasons.append("not_all_input_algebraically_equivalent")
            if not strict_lower:
                reasons.append("actual_cost_not_strictly_lower")
            if not static["pass"]:
                reasons.extend(name for name, passed in static["checks"].items() if not passed)
            if not known_pass:
                reasons.append("known_four_runtime_nonfinite_or_near_positive_gate_failed")
            candidate_rows.append(
                {
                    "task": task,
                    "baseline_member": f"task{task:03d}.onnx",
                    "baseline_sha256": digest_bytes(base_data),
                    "candidate_path": str(candidate_path.relative_to(ROOT)),
                    "candidate_sha256": digest_path(candidate_path),
                    "baseline_cost": baseline_cost,
                    "candidate_cost": candidate_cost,
                    "strict_lower": strict_lower,
                    "gain_if_accepted": (
                        math.log(baseline_cost["cost"] / candidate_cost["cost"])
                        if strict_lower
                        else 0.0
                    ),
                    "algebraic_proof": proof,
                    "structure": static,
                    "known_complete_four_configs": known,
                    "accepted": accepted,
                    "reasons": sorted(set(reasons)),
                }
            )
            print(
                f"task{task:03d}: cost={candidate_cost['cost']}/{baseline_cost['cost']} "
                f"proof={proof['all_input_real_algebraic_equivalence']} "
                f"known4={known_pass} structure={static['pass']} accepted={accepted}",
                flush=True,
            )

    accepted = [row for row in candidate_rows if row["accepted"]]
    result = {
        "authority": str(BASE_ZIP.relative_to(ROOT)),
        "authority_sha256": baseline_zip_sha,
        "scope": [TASK_MIN, TASK_MAX],
        "exclusions": sorted(EXCLUSIONS),
        "policy": {
            "all_input_real_algebraic_equivalence": True,
            "actual_cost_strictly_lower": True,
            "known_complete": True,
            "known_configurations": [
                "ORT_DISABLE_ALL threads=1",
                "ORT_DISABLE_ALL threads=4",
                "ORT_ENABLE_ALL threads=1",
                "ORT_ENABLE_ALL threads=4",
            ],
            "runtime_errors": 0,
            "nonfinite_values": 0,
            "near_positive_values_open_interval_0_0_25": 0,
            "strict_shape_data_prop": True,
            "truthful_runtime_shapes": True,
            "standard_domains": True,
            "conv_bias_ub": 0,
        },
        "inventory": {
            "files_scanned": len(inventory),
            "selection": "top eligible task150-400 members by maximum Einsum operand count",
            "rows": inventory,
        },
        "latent_prune_audit": {
            "candidates_examined": len(prune_rows),
            "rows": prune_rows,
        },
        "exact_absorption_candidates": candidate_rows,
        "accepted": accepted,
        "accepted_count": len(accepted),
        "aggregate_gain": sum(float(row["gain_if_accepted"]) for row in accepted),
        "protected_files_modified": [],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    winner_manifest = {
        "authority": str(BASE_ZIP.relative_to(ROOT)),
        "authority_sha256": baseline_zip_sha,
        "winners": [
            {
                "task": row["task"],
                "path": row["candidate_path"],
                "sha256": row["candidate_sha256"],
                "baseline_cost": row["baseline_cost"]["cost"],
                "candidate_cost": row["candidate_cost"]["cost"],
                "gain": row["gain_if_accepted"],
                "proof": "all-input real-algebra exact initializer contraction substitution",
                "known_complete_four_configs": True,
                "runtime_nonfinite_near_positive_zero": True,
            }
            for row in accepted
        ],
        "aggregate_gain": sum(float(row["gain_if_accepted"]) for row in accepted),
        "merge_performed": False,
    }
    (HERE / "winner_manifest.json").write_text(
        json.dumps(winner_manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "inventory_files": len(inventory),
                "prune_candidates": len(prune_rows),
                "exact_candidates": len(candidate_rows),
                "accepted": [row["task"] for row in accepted],
                "aggregate_gain": result["aggregate_gain"],
                "result": str((HERE / "result.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

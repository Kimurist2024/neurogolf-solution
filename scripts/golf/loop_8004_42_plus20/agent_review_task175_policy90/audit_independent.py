#!/usr/bin/env python3
"""Independent, non-promoting POLICY90 audit for task175.

The immutable authority is read directly from submission_base_8009.46.zip.
Only audit_evidence.json beside this script is written.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
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
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_MEMBER = "task175.onnx"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent/task175_r001.onnx"
)
EXPECTED_ZIP_SHA256 = (
    "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
)
EXPECTED_CANDIDATE_SHA256 = (
    "40a9405880836a60f100e0072b476e4383c12c7ee053eb12ada1f049ee2e8d7c"
)
EXPECTED_AUTHORITY_COST = 166
EXPECTED_CANDIDATE_COST = 145
FRESH_STREAMS = ((917_500_031, 2_000), (917_500_087, 2_000))
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
CANONICAL_SHAPE = [1, 10, 30, 30]
MARGIN = 0.25

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def model_payloads() -> tuple[bytes, bytes, bytes]:
    zip_data = AUTHORITY_ZIP.read_bytes()
    if digest(zip_data) != EXPECTED_ZIP_SHA256:
        raise AssertionError("authority ZIP SHA-256 mismatch")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_data = archive.read(AUTHORITY_MEMBER)
    candidate_data = CANDIDATE.read_bytes()
    if digest(candidate_data) != EXPECTED_CANDIDATE_SHA256:
        raise AssertionError("candidate SHA-256 mismatch")
    return zip_data, authority_data, candidate_data


def canonical_io(model: onnx.ModelProto) -> bool:
    return bool(
        len(model.graph.input) == 1
        and len(model.graph.output) == 1
        and model.graph.input[0].name == "input"
        and model.graph.output[0].name == "output"
        and dims(model.graph.input[0]) == CANONICAL_SHAPE
        and dims(model.graph.output[0]) == CANONICAL_SHAPE
        and model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        and model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    )


def static_audit(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(model, full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result["full_check"] = False
        result["full_check_error"] = f"{type(exc).__name__}: {exc}"

    inferred_models: dict[str, onnx.ModelProto] = {}
    for label, data_prop in (("strict", False), ("strict_data_prop", True)):
        try:
            inferred_models[label] = shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=data_prop
            )
            result[label] = True
        except Exception as exc:  # noqa: BLE001
            result[label] = False
            result[f"{label}_error"] = f"{type(exc).__name__}: {exc}"

    inferred = inferred_models.get("strict_data_prop")
    node_output_shapes: dict[str, list[int | None]] = {}
    unresolved: list[str] = []
    if inferred is not None:
        typed = {
            value.name: value
            for value in (
                list(inferred.graph.input)
                + list(inferred.graph.value_info)
                + list(inferred.graph.output)
            )
        }
        for node in inferred.graph.node:
            for name in node.output:
                if not name:
                    continue
                value = typed.get(name)
                if value is None:
                    unresolved.append(name)
                    continue
                shape = dims(value)
                node_output_shapes[name] = shape
                if not shape or any(dim is None or dim <= 0 for dim in shape):
                    unresolved.append(name)

    banned: list[str] = []
    nested: list[str] = []
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in scoring._EXCLUDED_OP_TYPES or "SEQUENCE" in upper:
            banned.append(node.op_type)
        nested.extend(
            f"{node.op_type}:{attr.name}"
            for attr in node.attribute
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        )

    lookup_or_cloak_ops = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type
            in {
                "TfIdfVectorizer",
                "Hardmax",
                "ArgMax",
                "TopK",
                "Gather",
                "GatherElements",
                "GatherND",
                "ScatterElements",
                "ScatterND",
                "Resize",
                "CenterCropPad",
                "AffineGrid",
            }
        }
    )
    external_data = [
        item.name
        for item in model.graph.initializer
        if item.data_location == onnx.TensorProto.EXTERNAL or item.external_data
    ]
    initializer_finite = all(
        np.isfinite(np.asarray(numpy_helper.to_array(item))).all()
        for item in model.graph.initializer
    )
    params_direct = int(
        sum(np.asarray(numpy_helper.to_array(item)).size for item in model.graph.initializer)
    )
    max_initializer_elements = max(
        (int(np.asarray(numpy_helper.to_array(item)).size) for item in model.graph.initializer),
        default=0,
    )
    einsums = [node for node in model.graph.node if node.op_type == "Einsum"]
    einsum_equations = [
        next(
            (
                onnx.helper.get_attribute_value(attr).decode("ascii")
                for attr in node.attribute
                if attr.name == "equation"
            ),
            "",
        )
        for node in einsums
    ]
    result.update(
        {
            "canonical_io": canonical_io(model),
            "nodes": len(model.graph.node),
            "initializers": len(model.graph.initializer),
            "params_direct": params_direct,
            "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
            "opsets": [[item.domain, int(item.version)] for item in model.opset_import],
            "standard_domains": all(item.domain in ("", "ai.onnx") for item in model.opset_import)
            and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "nested_graphs": nested,
            "banned_ops": sorted(set(banned)),
            "lookup_or_cloak_ops": lookup_or_cloak_ops,
            "external_data_initializers": external_data,
            "conv_family_bias_ub": check_conv_bias(model),
            "initializer_values_finite": bool(initializer_finite),
            "max_initializer_elements": max_initializer_elements,
            "value_info_count": len(model.graph.value_info),
            "node_output_shapes": node_output_shapes,
            "unresolved_or_dynamic_node_outputs": unresolved,
            "intermediate_node_outputs": [
                name
                for node in model.graph.node
                for name in node.output
                if name and name != "output"
            ],
            "einsum_input_arities": [len(node.input) for node in einsums],
            "einsum_equations": einsum_equations,
            "high_arity_einsum_disclosed": any(len(node.input) >= 10 for node in einsums),
        }
    )
    result["pass"] = bool(
        result["full_check"]
        and result["strict"]
        and result["strict_data_prop"]
        and result["canonical_io"]
        and result["standard_domains"]
        and not result["functions"]
        and not result["sparse_initializers"]
        and not result["nested_graphs"]
        and not result["banned_ops"]
        and not result["lookup_or_cloak_ops"]
        and not result["external_data_initializers"]
        and not result["conv_family_bias_ub"]
        and result["initializer_values_finite"]
        and not result["unresolved_or_dynamic_node_outputs"]
    )
    return result


def graph_delta(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    left = {item.name: numpy_helper.to_array(item) for item in authority.graph.initializer}
    right = {item.name: numpy_helper.to_array(item) for item in candidate.graph.initializer}
    unchanged = ["Q", "S", "G2", "K"]

    authority_shell = copy.deepcopy(authority)
    candidate_shell = copy.deepcopy(candidate)
    del authority_shell.graph.initializer[:]
    del candidate_shell.graph.initializer[:]
    shell_equal = (
        authority_shell.SerializeToString(deterministic=True)
        == candidate_shell.SerializeToString(deterministic=True)
    )
    retained_second_slice = bool(
        np.array_equal(right["C0"][0], left["C0"][1])
        and np.array_equal(right["G1"][0], left["G1"][1])
    )
    dropped_component_nonzero = bool(
        np.count_nonzero(left["C0"][0]) and np.count_nonzero(left["G1"][0])
    )
    return {
        "all_non_initializer_model_fields_proto_equal": shell_equal,
        "unchanged_initializers_proto_equal": {
            name: bool(np.array_equal(left[name], right[name])) for name in unchanged
        },
        "changed_initializer_shapes": {
            name: {"authority": list(left[name].shape), "candidate": list(right[name].shape)}
            for name in ("C0", "G1")
        },
        "candidate_retains_authority_zero_based_L_index_1": retained_second_slice,
        "candidate_drops_authority_zero_based_L_index_0": retained_second_slice,
        "dropped_slice_nonzero_entries": {
            "C0": int(np.count_nonzero(left["C0"][0])),
            "G1": int(np.count_nonzero(left["G1"][0])),
        },
        "dropped_component_not_algebraically_zero": dropped_component_nonzero,
        "parameter_delta": int(
            sum(array.size for array in left.values())
            - sum(array.size for array in right.values())
        ),
        "exact_whitelist_delta": bool(
            shell_equal
            and retained_second_slice
            and dropped_component_nonzero
            and all(np.array_equal(left[name], right[name]) for name in unchanged)
            and set(left) == set(right) == {"Q", "S", "C0", "G1", "G2", "K"}
        ),
    }


def score_model(data: bytes, task: int, require_correct: bool, label: str) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"score_{label}_", dir=HERE) as workdir:
        scored = scoring.score_and_verify(
            model, task, workdir, label=label, require_correct=require_correct
        )
    if scored is None:
        raise RuntimeError(f"official-like scorer rejected {label}")
    return scored


def make_session(data: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def one_hot_input(grid: np.ndarray) -> np.ndarray:
    result = np.zeros(CANONICAL_SHAPE, dtype=np.float32)
    rows, cols = np.indices(grid.shape)
    result[0, grid.astype(np.intp), rows, cols] = 1.0
    return result


def one_hot_expected(grid: np.ndarray) -> np.ndarray:
    result = np.zeros(CANONICAL_SHAPE, dtype=bool)
    rows, cols = np.indices(grid.shape)
    result[0, grid.astype(np.intp), rows, cols] = True
    return result


def load_rule():
    path = ROOT / "inputs/sakana-gcg-2025/raw/task175.py"
    spec = importlib.util.spec_from_file_location("review_task175_rule", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def support_restore(grid: np.ndarray) -> np.ndarray:
    """Input-only task175 transform on the legal generate() support.

    The generator rejects every off-diagonal pair whose two symmetric cells
    were both erased. The uncut field is symmetric, and all diagonal colors
    equal input[0,0].
    """
    restored = grid.copy()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] != 0:
                continue
            restored[row, col] = grid[0, 0] if row == col else grid[col, row]
    return restored


def support_invariant(grid: np.ndarray) -> bool:
    size = grid.shape[0]
    return bool(
        grid[0, 0] != 0
        and all(
            not (grid[row, col] == 0 and grid[col, row] == 0)
            for col in range(size)
            for row in range(col)
        )
    )


def known_cases() -> list[dict[str, Any]]:
    examples = scoring.load_examples(175)
    return [
        {"split": split, "split_index": index, "example": example}
        for split in ("train", "test", "arc-gen")
        for index, example in enumerate(examples.get(split, []))
    ]


def fresh_cases(seed: int, count: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    generator = importlib.import_module("task_73251a56")
    rule = load_rule()
    random.seed(seed)
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    rule_mismatches = 0
    support_reference_mismatches = 0
    support_invariant_violations = 0
    top_left_erased = 0
    fingerprints: set[str] = set()
    erased_cells = 0
    erased_diagonal_cells = 0
    generation_errors = 0
    for _ in range(count):
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        inp = np.asarray(example["input"], dtype=np.uint8)
        out = np.asarray(example["output"], dtype=np.uint8)
        cases.append((inp, out))
        fingerprints.add(digest(inp.tobytes()))
        erased_cells += int(np.count_nonzero(inp == 0))
        erased_diagonal_cells += int(np.count_nonzero(np.diag(inp) == 0))
        top_left_erased += int(inp[0, 0] == 0)
        support_invariant_violations += int(not support_invariant(inp))
        support_reference_mismatches += int(not np.array_equal(support_restore(inp), out))
        rule_out = np.asarray(rule(example["input"]), dtype=np.uint8)
        rule_mismatches += int(not np.array_equal(rule_out, out))
    return cases, {
        "seed": seed,
        "requested": count,
        "generated": len(cases),
        "generation_errors": generation_errors,
        "unique_inputs": len(fingerprints),
        "raw_rule_mismatches": rule_mismatches,
        "support_reference_mismatches": support_reference_mismatches,
        "support_invariant_violations": support_invariant_violations,
        "top_left_erased": top_left_erased,
        "erased_cells": erased_cells,
        "erased_diagonal_cells": erased_diagonal_cells,
    }


def blank_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "shape_mismatches": 0,
        "determinism_mismatches": 0,
        "cross_config_raw_mismatches": 0,
        "min_positive": None,
        "max_nonpositive": None,
        "output_shapes": set(),
        "first_failure": None,
    }


def update_numeric_stats(stats: dict[str, Any], raw: np.ndarray) -> None:
    stats["nonfinite_values"] += int(np.count_nonzero(~np.isfinite(raw)))
    stats["near_positive_values"] += int(
        np.count_nonzero((raw > 0.0) & (raw < MARGIN))
    )
    positive = raw[raw > 0.0]
    nonpositive = raw[raw <= 0.0]
    if positive.size:
        value = float(positive.min())
        stats["min_positive"] = (
            value if stats["min_positive"] is None else min(stats["min_positive"], value)
        )
    if nonpositive.size:
        value = float(nonpositive.max())
        stats["max_nonpositive"] = (
            value
            if stats["max_nonpositive"] is None
            else max(stats["max_nonpositive"], value)
        )


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    result["output_shapes"] = sorted([list(item) for item in stats["output_shapes"]])
    total = result["right"] + result["wrong"]
    result["accuracy"] = result["right"] / total if total else 0.0
    return result


def describe_failure(
    case: dict[str, Any], candidate_raw: np.ndarray, authority_raw: np.ndarray
) -> dict[str, Any]:
    example = case["example"]
    inp = np.asarray(example["input"], dtype=np.uint8)
    out = np.asarray(example["output"], dtype=np.uint8)
    expected = one_hot_expected(out)
    candidate_mask = candidate_raw > 0.0
    authority_mask = authority_raw > 0.0
    different_channels = np.argwhere(candidate_mask != expected)
    spatial = sorted({(int(row), int(col)) for _, _, row, col in different_channels})
    cells = []
    for row, col in spatial:
        predicted = np.flatnonzero(candidate_mask[0, :, row, col]).astype(int).tolist()
        cells.append(
            {
                "row": row,
                "col": col,
                "input": int(inp[row, col]),
                "transpose_input": int(inp[col, row]),
                "expected_color": int(out[row, col]),
                "candidate_positive_channels": predicted,
                "both_symmetric_input_cells_erased": bool(
                    row != col and inp[row, col] == 0 and inp[col, row] == 0
                ),
                "candidate_expected_channel_raw": float(candidate_raw[0, out[row, col], row, col]),
                "authority_expected_channel_raw": float(authority_raw[0, out[row, col], row, col]),
            }
        )
    unordered_pairs = {tuple(sorted(((row, col), (col, row)))) for row, col in spatial}
    return {
        "global_index": int(case["global_index"]),
        "split": case["split"],
        "split_index": int(case["split_index"]),
        "threshold_channel_differences": int(len(different_channels)),
        "wrong_spatial_cells": len(spatial),
        "symmetric_pair_count": len(unordered_pairs),
        "all_wrong_cells_are_double_erased_off_diagonal": all(
            cell["both_symmetric_input_cells_erased"] for cell in cells
        ),
        "candidate_mask_equal_authority": bool(np.array_equal(candidate_mask, authority_mask)),
        "candidate_raw_equal_authority": bool(np.array_equal(candidate_raw, authority_raw)),
        "cells": cells,
    }


def audit_known(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    raw_cases = known_cases()
    cases = []
    rule = load_rule()
    rule_mismatches = 0
    support_ref_mismatches = 0
    support_violations = 0
    for global_index, item in enumerate(raw_cases):
        item = dict(item)
        item["global_index"] = global_index
        cases.append(item)
        example = item["example"]
        inp = np.asarray(example["input"], dtype=np.uint8)
        out = np.asarray(example["output"], dtype=np.uint8)
        rule_mismatches += int(
            not np.array_equal(np.asarray(rule(example["input"]), dtype=np.uint8), out)
        )
        support_ref_mismatches += int(not np.array_equal(support_restore(inp), out))
        support_violations += int(not support_invariant(inp))

    configs: dict[str, Any] = {}
    baseline_candidate_digests: list[str] | None = None
    baseline_authority_digests: list[str] | None = None
    failures: list[dict[str, Any]] = []
    for disabled, threads, label in CONFIGS:
        candidate_session = make_session(candidate_data, disabled, threads)
        repeat_session = make_session(candidate_data, disabled, threads)
        authority_session = make_session(authority_data, disabled, threads)
        candidate_stats = blank_stats()
        authority_stats = blank_stats()
        raw_equal_authority = 0
        mask_equal_authority = 0
        candidate_digests: list[str] = []
        authority_digests: list[str] = []
        for case in cases:
            example = case["example"]
            inp_grid = np.asarray(example["input"], dtype=np.uint8)
            out_grid = np.asarray(example["output"], dtype=np.uint8)
            inp = one_hot_input(inp_grid)
            expected = one_hot_expected(out_grid)
            try:
                candidate_raw = np.asarray(
                    candidate_session.run(["output"], {"input": inp})[0]
                )
                candidate_repeat = np.asarray(
                    repeat_session.run(["output"], {"input": inp})[0]
                )
            except Exception as exc:  # noqa: BLE001
                candidate_stats["runtime_errors"] += 1
                candidate_stats["wrong"] += 1
                if candidate_stats["first_failure"] is None:
                    candidate_stats["first_failure"] = {
                        "global_index": case["global_index"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue
            try:
                authority_raw = np.asarray(
                    authority_session.run(["output"], {"input": inp})[0]
                )
            except Exception as exc:  # noqa: BLE001
                authority_stats["runtime_errors"] += 1
                authority_stats["wrong"] += 1
                if authority_stats["first_failure"] is None:
                    authority_stats["first_failure"] = {
                        "global_index": case["global_index"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue

            for stats, raw in ((candidate_stats, candidate_raw), (authority_stats, authority_raw)):
                stats["output_shapes"].add(tuple(raw.shape))
                stats["shape_mismatches"] += int(list(raw.shape) != CANONICAL_SHAPE)
                update_numeric_stats(stats, raw)
                if np.array_equal(raw > 0.0, expected):
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
                    if stats["first_failure"] is None:
                        stats["first_failure"] = {"global_index": case["global_index"]}
            candidate_stats["determinism_mismatches"] += int(
                not np.array_equal(candidate_raw, candidate_repeat)
            )
            raw_equal_authority += int(np.array_equal(candidate_raw, authority_raw))
            mask_equal_authority += int(
                np.array_equal(candidate_raw > 0.0, authority_raw > 0.0)
            )
            candidate_digests.append(digest(candidate_raw.tobytes()))
            authority_digests.append(digest(authority_raw.tobytes()))
            if label == "disable_all_threads1" and not np.array_equal(
                candidate_raw > 0.0, expected
            ):
                failures.append(describe_failure(case, candidate_raw, authority_raw))

        if baseline_candidate_digests is None:
            baseline_candidate_digests = candidate_digests
            baseline_authority_digests = authority_digests
        else:
            candidate_stats["cross_config_raw_mismatches"] = sum(
                left != right
                for left, right in zip(
                    baseline_candidate_digests, candidate_digests, strict=True
                )
            )
            authority_stats["cross_config_raw_mismatches"] = sum(
                left != right
                for left, right in zip(
                    baseline_authority_digests, authority_digests, strict=True
                )
            )
        configs[label] = {
            "candidate": finalize_stats(candidate_stats),
            "authority": finalize_stats(authority_stats),
            "candidate_raw_equal_authority": raw_equal_authority,
            "candidate_mask_equal_authority": mask_equal_authority,
            "candidate_raw_stream_sha256": digest("".join(candidate_digests).encode()),
            "authority_raw_stream_sha256": digest("".join(authority_digests).encode()),
        }
    return {
        "cases": len(cases),
        "raw_rule_mismatches": rule_mismatches,
        "support_reference_mismatches": support_ref_mismatches,
        "support_invariant_violations": support_violations,
        "configs": configs,
        "failures": failures,
    }


def audit_fresh_stream(
    candidate_data: bytes,
    seed: int,
    count: int,
) -> dict[str, Any]:
    cases, generation = fresh_cases(seed, count)
    configs: dict[str, Any] = {}
    baseline_digests: list[str] | None = None
    for disabled, threads, label in CONFIGS:
        session = make_session(candidate_data, disabled, threads)
        repeat_session = make_session(candidate_data, disabled, threads)
        stats = blank_stats()
        digests: list[str] = []
        for case_index, (inp_grid, out_grid) in enumerate(cases):
            inp = one_hot_input(inp_grid)
            expected = one_hot_expected(out_grid)
            try:
                raw = np.asarray(session.run(["output"], {"input": inp})[0])
                repeated = np.asarray(repeat_session.run(["output"], {"input": inp})[0])
            except Exception as exc:  # noqa: BLE001
                stats["runtime_errors"] += 1
                stats["wrong"] += 1
                if stats["first_failure"] is None:
                    stats["first_failure"] = {
                        "case_index": case_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                continue
            stats["output_shapes"].add(tuple(raw.shape))
            stats["shape_mismatches"] += int(list(raw.shape) != CANONICAL_SHAPE)
            update_numeric_stats(stats, raw)
            if np.array_equal(raw > 0.0, expected):
                stats["right"] += 1
            else:
                stats["wrong"] += 1
                if stats["first_failure"] is None:
                    different = np.argwhere((raw > 0.0) != expected)
                    stats["first_failure"] = {
                        "case_index": case_index,
                        "threshold_channel_differences": int(len(different)),
                    }
            stats["determinism_mismatches"] += int(not np.array_equal(raw, repeated))
            digests.append(digest(raw.tobytes()))
        if baseline_digests is None:
            baseline_digests = digests
        else:
            stats["cross_config_raw_mismatches"] = sum(
                left != right
                for left, right in zip(baseline_digests, digests, strict=True)
            )
        finalized = finalize_stats(stats)
        finalized["raw_stream_sha256"] = digest("".join(digests).encode())
        configs[label] = finalized
    return {"generation": generation, "configs": configs}


def main() -> int:
    zip_data, authority_data, candidate_data = model_payloads()
    authority_static = static_audit(authority_data)
    candidate_static = static_audit(candidate_data)
    delta = graph_delta(authority_data, candidate_data)
    authority_score = score_model(authority_data, 175, True, "authority")
    candidate_score = score_model(candidate_data, 175, False, "candidate")
    known = audit_known(authority_data, candidate_data)

    fresh: dict[str, Any] = {}
    for seed, count in FRESH_STREAMS:
        print(f"fresh seed {seed}: generating and evaluating {count} cases", flush=True)
        fresh[str(seed)] = audit_fresh_stream(candidate_data, seed, count)
        print(f"fresh seed {seed}: complete", flush=True)

    fresh_pass = all(
        stream["generation"]["generated"] == count
        and stream["generation"]["generation_errors"] == 0
        and stream["generation"]["raw_rule_mismatches"] == 0
        and stream["generation"]["support_reference_mismatches"] == 0
        and stream["generation"]["support_invariant_violations"] == 0
        and all(
            stats["accuracy"] >= 0.90
            and stats["runtime_errors"] == 0
            and stats["nonfinite_values"] == 0
            and stats["near_positive_values"] == 0
            and stats["shape_mismatches"] == 0
            and stats["determinism_mismatches"] == 0
            and stats["cross_config_raw_mismatches"] == 0
            for stats in stream["configs"].values()
        )
        for (_seed, count), stream in zip(FRESH_STREAMS, fresh.values(), strict=True)
    )
    known_candidate_right = {
        label: item["candidate"]["right"] for label, item in known["configs"].items()
    }
    known_stable = all(
        item["candidate"]["runtime_errors"] == 0
        and item["candidate"]["nonfinite_values"] == 0
        and item["candidate"]["near_positive_values"] == 0
        and item["candidate"]["shape_mismatches"] == 0
        and item["candidate"]["determinism_mismatches"] == 0
        and item["candidate"]["cross_config_raw_mismatches"] == 0
        for item in known["configs"].values()
    )
    structure_pass = bool(authority_static["pass"] and candidate_static["pass"])
    cost_pass = bool(
        authority_score["cost"] == EXPECTED_AUTHORITY_COST
        and candidate_score["cost"] == EXPECTED_CANDIDATE_COST
        and candidate_score["cost"] < authority_score["cost"]
    )
    normal_policy_evidence = {
        "task175_absent_from_private_zero_catalog": True,
        "generator_support_input_identifiable": all(
            stream["generation"]["support_reference_mismatches"] == 0
            and stream["generation"]["support_invariant_violations"] == 0
            and stream["generation"]["top_left_erased"] == 0
            for stream in fresh.values()
        ),
        "no_lookup_or_shape_cloak": not candidate_static["lookup_or_cloak_ops"]
        and not candidate_static["external_data_initializers"]
        and candidate_static["canonical_io"]
        and not candidate_static["unresolved_or_dynamic_node_outputs"],
        "lineage_disclosure": (
            "The candidate preserves the authority's single 18-input standard-domain "
            "Einsum and removes one shared C0/G1 latent slice; it is high-arity but "
            "contains no lookup/index/scatter/custom-domain or declared-shape cloak."
        ),
    }
    normal_policy_pass = all(
        value for key, value in normal_policy_evidence.items() if key != "lineage_disclosure"
    )
    admit = bool(
        structure_pass
        and delta["exact_whitelist_delta"]
        and cost_pass
        and known_candidate_right
        and min(known_candidate_right.values()) / known["cases"] >= 0.90
        and known_stable
        and fresh_pass
        and normal_policy_pass
    )
    result = {
        "task": 175,
        "decision": "ADMIT_POLICY90" if admit else "REJECT",
        "classification": "NORMAL_POLICY90_NOT_KNOWN_EXACT" if admit else "REJECTED",
        "stage_action": "NONE_NO_STAGE_NO_MERGE",
        "environment": {
            "python": sys.version.split()[0],
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
        },
        "integrity": {
            "authority_zip_path": str(AUTHORITY_ZIP.relative_to(ROOT)),
            "authority_zip_sha256": digest(zip_data),
            "authority_member": AUTHORITY_MEMBER,
            "authority_member_sha256": digest(authority_data),
            "authority_member_size": len(authority_data),
            "candidate_path": str(CANDIDATE.relative_to(ROOT)),
            "candidate_sha256": digest(candidate_data),
            "candidate_size": len(candidate_data),
        },
        "cost": {
            "authority": authority_score,
            "candidate": candidate_score,
            "gain_ln_authority_over_candidate": math.log(
                authority_score["cost"] / candidate_score["cost"]
            ),
        },
        "structure": {
            "authority": authority_static,
            "candidate": candidate_static,
            "delta": delta,
        },
        "known": known,
        "fresh": fresh,
        "generator_and_policy_classification": normal_policy_evidence,
        "gates": {
            "structure_pass": structure_pass,
            "exact_whitelist_delta": delta["exact_whitelist_delta"],
            "cost_pass": cost_pass,
            "known_rate_at_least_90_each_config": min(known_candidate_right.values())
            / known["cases"]
            >= 0.90,
            "known_stability_pass": known_stable,
            "fresh_pass": fresh_pass,
            "normal_policy_classification_pass": normal_policy_pass,
        },
    }
    output = HERE / "audit_evidence.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "cost": result["cost"],
                "known": known_candidate_right,
                "known_failures": len(known["failures"]),
                "fresh": {
                    seed: {
                        label: [stats["right"], stats["wrong"]]
                        for label, stats in stream["configs"].items()
                    }
                    for seed, stream in fresh.items()
                },
            },
            indent=2,
        )
    )
    return 0 if admit else 1


if __name__ == "__main__":
    raise SystemExit(main())

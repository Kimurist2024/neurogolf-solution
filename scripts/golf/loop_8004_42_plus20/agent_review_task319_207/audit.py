#!/usr/bin/env python3
"""Independent fail-closed audit of the task319 lane-201 candidate.

This script is intentionally self-contained and reads the authority directly
from the immutable 8009.46 archive.  It never promotes or rewrites a model.
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
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ARCHIVE = ROOT / "submission_base_8009.46.zip"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task319_exact_201/candidates"
    / "task319_combined_runnable.onnx"
)
ARCHIVE_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_SHA256 = "29d5bfe25f86b18e0b5938d85e4f38cca72c34d8aad6390bff43579124d0e391"
CANDIDATE_SHA256 = "ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4"
TASK = 319
FRESH_SEEDS = (319_207_011, 319_207_029)
FRESH_PER_SEED = 1_500
CONFIGS = (
    ("disable_all_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_all_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("default_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("default_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dim_list(item: onnx.ValueInfoProto) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in item.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def model_structure(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    errors: list[str] = []
    checks: dict[str, bool] = {}
    for label, kwargs in (
        ("checker_full", {"full_check": True}),
        ("infer_strict", {"strict_mode": True}),
        ("infer_strict_data_prop", {"strict_mode": True, "data_prop": True}),
    ):
        try:
            if label == "checker_full":
                onnx.checker.check_model(model, **kwargs)
            else:
                shape_inference.infer_shapes(copy.deepcopy(model), **kwargs)
            checks[label] = True
        except Exception as exc:  # noqa: BLE001
            checks[label] = False
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    banned = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
    nonstandard_domains = [x.domain for x in model.opset_import if x.domain not in {"", "ai.onnx"}]
    nested = []
    banned_nodes = []
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in banned or "Sequence" in node.op_type:
            banned_nodes.append(node.op_type)
        for attr in node.attribute:
            if attr.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}:
                nested.append(f"{node.output[0]}:{attr.name}")

    all_vi = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    names = [item.name for item in all_vi]
    bad_dims = []
    for item in all_vi:
        if not item.type.HasField("tensor_type"):
            bad_dims.append([item.name, "non_tensor"])
            continue
        dims = dim_list(item)
        if any(not isinstance(size, int) or size <= 0 for size in dims):
            bad_dims.append([item.name, dims])

    init_nonfinite = []
    external = []
    for item in model.graph.initializer:
        if item.data_location == TensorProto.EXTERNAL or item.external_data:
            external.append(item.name)
        array = numpy_helper.to_array(item)
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            init_nonfinite.append(item.name)

    io = {
        "inputs": [[x.name, int(x.type.tensor_type.elem_type), dim_list(x)] for x in model.graph.input],
        "outputs": [[x.name, int(x.type.tensor_type.elem_type), dim_list(x)] for x in model.graph.output],
    }
    canonical_io = (
        io["inputs"] == [["input", TensorProto.FLOAT, [1, 10, 30, 30]]]
        and len(io["outputs"]) == 1
        and io["outputs"][0][0] == "output"
        and io["outputs"][0][2] == [1, 10, 30, 30]
    )

    # UB0: a Conv/QLinearConv bias, when present, must have exactly M entries.
    init = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
    bias_errors = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "QLinearConv"}:
            continue
        weight_slot = 1 if node.op_type == "Conv" else 3
        bias_slot = 2 if node.op_type == "Conv" else 8
        if len(node.input) <= bias_slot or not node.input[bias_slot]:
            continue
        if node.input[weight_slot] not in init or node.input[bias_slot] not in init:
            bias_errors.append(f"dynamic:{node.output[0]}")
            continue
        channels = int(init[node.input[weight_slot]].shape[0])
        bias_size = int(init[node.input[bias_slot]].size)
        if bias_size != channels:
            bias_errors.append(f"{node.output[0]}:{bias_size}!={channels}")

    checks.update(
        canonical_io=canonical_io,
        unique_value_info_names=len(names) == len(set(names)),
        positive_static_metadata=not bad_dims,
        standard_domain_only=not nonstandard_domains,
        no_banned_ops=not banned_nodes,
        no_nested_graphs=not nested,
        no_functions=not model.functions,
        no_sparse_initializers=not model.graph.sparse_initializer,
        no_external_data=not external,
        finite_initializers=not init_nonfinite,
        conv_bias_ub0=not bias_errors,
        size_limit=len(data) <= int(1.44 * 1024 * 1024),
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "bad_dims": bad_dims,
        "nonstandard_domains": nonstandard_domains,
        "banned_nodes": banned_nodes,
        "nested_graphs": nested,
        "external_initializers": external,
        "nonfinite_initializers": init_nonfinite,
        "conv_bias_errors": bias_errors,
        "io": io,
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "opsets": [[x.domain, int(x.version)] for x in model.opset_import],
    }


def make_session(data: bytes, level: ort.GraphOptimizationLevel, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def known_rows() -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    examples = scoring.load_examples(TASK)
    for subset in ("train", "test", "arc-gen"):
        for example in examples[subset]:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append({"input": converted["input"], "truth": converted["output"] > 0})
    return rows


def fresh_rows() -> tuple[list[dict[str, np.ndarray]], list[dict[str, int]]]:
    task_hash = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())[str(TASK)]
    generator = importlib.import_module(f"task_{task_hash}")
    rows: list[dict[str, np.ndarray]] = []
    generation: list[dict[str, int]] = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        accepted = attempts = errors = 0
        while accepted < FRESH_PER_SEED:
            attempts += 1
            try:
                converted = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            if converted is None:
                continue
            rows.append({"input": converted["input"], "truth": converted["output"] > 0})
            accepted += 1
        generation.append({"seed": seed, "accepted": accepted, "attempts": attempts, "errors": errors})
    return rows, generation


def compare_outputs(
    authority_data: bytes,
    candidate_data: bytes,
    known: list[dict[str, np.ndarray]],
    fresh: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config, level, threads in CONFIGS:
        sessions: dict[str, ort.InferenceSession] = {}
        session_errors: dict[str, str] = {}
        for label, data in (("authority", authority_data), ("candidate", candidate_data)):
            try:
                sessions[label] = make_session(data, level, threads)
            except Exception as exc:  # noqa: BLE001
                session_errors[label] = f"{type(exc).__name__}: {exc}"
        config_row: dict[str, Any] = {"session_errors": session_errors}
        for split, rows in (("known", known), ("fresh", fresh)):
            stats: dict[str, Any] = {
                "cases": len(rows),
                "raw_equal": 0,
                "raw_different": 0,
                "first_raw_difference": None,
                "authority_right": 0,
                "candidate_right": 0,
                "authority_errors": 0,
                "candidate_errors": 0,
                "authority_nonfinite": 0,
                "candidate_nonfinite": 0,
                "candidate_min_positive": None,
                "candidate_max": None,
            }
            min_positive: float | None = None
            max_value: float | None = None
            for case, row in enumerate(rows):
                raw: dict[str, np.ndarray] = {}
                for label in ("authority", "candidate"):
                    try:
                        session = sessions[label]
                        output = np.asarray(session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: row["input"]},
                        )[0])
                        raw[label] = output
                        if output.dtype.kind in "fc":
                            stats[f"{label}_nonfinite"] += int(np.count_nonzero(~np.isfinite(output)))
                        if np.array_equal(output > 0, row["truth"]):
                            stats[f"{label}_right"] += 1
                        if label == "candidate":
                            pos = output[output > 0]
                            if pos.size:
                                value = float(np.min(pos))
                                min_positive = value if min_positive is None else min(min_positive, value)
                            value = float(np.max(output))
                            max_value = value if max_value is None else max(max_value, value)
                    except Exception as exc:  # noqa: BLE001
                        stats[f"{label}_errors"] += 1
                        if stats["first_raw_difference"] is None:
                            stats["first_raw_difference"] = {
                                "case": case,
                                "model": label,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                if len(raw) != 2:
                    continue
                if np.array_equal(raw["authority"], raw["candidate"]):
                    stats["raw_equal"] += 1
                else:
                    stats["raw_different"] += 1
                    if stats["first_raw_difference"] is None:
                        stats["first_raw_difference"] = {
                            "case": case,
                            "different_values": int(np.count_nonzero(raw["authority"] != raw["candidate"])),
                        }
            stats["candidate_min_positive"] = min_positive
            stats["candidate_max"] = max_value
            config_row[split] = stats
        result[config] = config_row
    return result


def trace_session(
    data: bytes,
    level: ort.GraphOptimizationLevel,
) -> tuple[ort.InferenceSession, list[str], dict[str, list[int]]]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    declared = {
        name: [int(dim.dim_value) for dim in item.type.tensor_type.shape.dim]
        for name, item in typed.items()
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for output in node.output:
            if output and output in typed and output not in names:
                traced.graph.output.append(copy.deepcopy(typed[output]))
                names.append(output)
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session, names, declared


def trace_shapes(data: bytes, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        try:
            session, names, declared = trace_session(data, level)
        except Exception as exc:  # noqa: BLE001
            modes[mode] = {"session_error": f"{type(exc).__name__}: {exc}"}
            continue
        mismatches: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = {}
        runtime_errors = nonfinite = 0
        for row in rows:
            try:
                outputs = session.run(names, {session.get_inputs()[0].name: row["input"]})
            except Exception:  # noqa: BLE001
                runtime_errors += 1
                continue
            for name, output in zip(names, outputs, strict=True):
                array = np.asarray(output)
                if array.dtype.kind in "fc":
                    nonfinite += int(np.count_nonzero(~np.isfinite(array)))
                actual = tuple(int(x) for x in array.shape)
                wanted = tuple(declared[name])
                if actual != wanted:
                    key = (name, wanted, actual)
                    mismatches[key] = mismatches.get(key, 0) + 1
        rows_out = [
            {"tensor": name, "declared": list(wanted), "runtime": list(actual), "cases": count}
            for (name, wanted, actual), count in sorted(mismatches.items())
        ]
        modes[mode] = {
            "cases": len(rows),
            "runtime_tensors": len(names),
            "runtime_errors": runtime_errors,
            "nonfinite_values": nonfinite,
            "mismatch_count": len(rows_out),
            "mismatches": rows_out,
        }
    return modes


def selected_trace_session(
    data: bytes,
    level: ort.GraphOptimizationLevel,
    names: list[str],
) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    del model.graph.output[:]
    for name in names:
        model.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def rewrite_relation_trace(
    authority_data: bytes,
    candidate_data: bytes,
    rows: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    a_names = [
        "safe_name_83", "safe_name_96", "safe_name_97", "safe_name_98",
        "safe_name_100", "safe_name_101", "safe_name_108", "other_idx",
        "safe_name_113", "safe_name_117", "w_u8_2", "output",
    ]
    c_names = [
        "safe_name_83", "safe_name_96", "safe_name_97", "safe_name_98",
        "safe_name_100", "safe_name_108", "other_idx", "safe_name_113",
        "safe_name_117", "w_u8_2", "output",
    ]
    result: dict[str, Any] = {}
    relations = (
        "corr_input_transposed", "corr_kernel_transposed", "qconv_authority_twice_candidate",
        "qmax_authority_twice_candidate", "selected_count_equal", "left_shift_exact_double",
        "predicate_equal", "other_index_value_equal", "selected_bits_equal",
        "target_color_equal", "terminal_weights_equal", "output_equal",
    )
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        failures = {name: 0 for name in relations}
        first_failure: dict[str, int] = {}
        errors = 0
        max_values = {"authority_qconv": 0, "candidate_qconv": 0, "selected_count": 0, "authority_shift": 0}
        a_session = selected_trace_session(authority_data, level, a_names)
        c_session = selected_trace_session(candidate_data, level, c_names)
        for case, row in enumerate(rows):
            try:
                aa = dict(zip(a_names, a_session.run(a_names, {a_session.get_inputs()[0].name: row["input"]}), strict=True))
                cc = dict(zip(c_names, c_session.run(c_names, {c_session.get_inputs()[0].name: row["input"]}), strict=True))
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            predicates = {
                "corr_input_transposed": np.array_equal(cc["safe_name_83"], np.swapaxes(aa["safe_name_83"], -1, -2)),
                "corr_kernel_transposed": np.array_equal(cc["safe_name_96"], np.swapaxes(aa["safe_name_96"], -1, -2)),
                # Both correlation operands were transposed, so the complete
                # QLinearConv map is transposed as well as halved.  ReduceMax
                # immediately consumes both spatial axes.
                "qconv_authority_twice_candidate": np.array_equal(
                    aa["safe_name_97"].astype(np.int16),
                    2 * np.swapaxes(cc["safe_name_97"].astype(np.int16), -1, -2),
                ),
                "qmax_authority_twice_candidate": np.array_equal(aa["safe_name_98"].astype(np.int16), 2 * cc["safe_name_98"].astype(np.int16)),
                "selected_count_equal": np.array_equal(aa["safe_name_100"], cc["safe_name_100"]),
                "left_shift_exact_double": np.array_equal(aa["safe_name_101"].astype(np.int16), 2 * aa["safe_name_100"].astype(np.int16)),
                "predicate_equal": np.array_equal(aa["safe_name_108"], cc["safe_name_108"]),
                "other_index_value_equal": np.array_equal(np.ravel(aa["other_idx"]), np.ravel(cc["other_idx"])),
                "selected_bits_equal": np.array_equal(aa["safe_name_113"], cc["safe_name_113"]),
                "target_color_equal": np.array_equal(np.ravel(aa["safe_name_117"]), np.ravel(cc["safe_name_117"])),
                "terminal_weights_equal": np.array_equal(aa["w_u8_2"], cc["w_u8_2"]),
                "output_equal": np.array_equal(aa["output"], cc["output"]),
            }
            for relation, passed in predicates.items():
                if not passed:
                    failures[relation] += 1
                    first_failure.setdefault(relation, case)
            max_values["authority_qconv"] = max(max_values["authority_qconv"], int(np.max(aa["safe_name_97"])))
            max_values["candidate_qconv"] = max(max_values["candidate_qconv"], int(np.max(cc["safe_name_97"])))
            max_values["selected_count"] = max(max_values["selected_count"], int(np.max(aa["safe_name_100"])))
            max_values["authority_shift"] = max(max_values["authority_shift"], int(np.max(aa["safe_name_101"])))
        result[mode] = {
            "cases": len(rows),
            "runtime_errors": errors,
            "relation_failures": failures,
            "first_failure": first_failure,
            "max_observed": max_values,
            "passed": errors == 0 and not any(failures.values()),
        }
    return result


def cross_correlation(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    padded = np.pad(array, ((0, 2), (0, 2)))
    return np.asarray([
        [int(np.sum(padded[r:r + 5, c:c + 5] * kernel)) for c in range(3)]
        for r in range(3)
    ], dtype=np.int32)


def formal_rewrite_checks(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    a_nodes = {x.output[0]: x for x in authority.graph.node}
    c_nodes = {x.output[0]: x for x in candidate.graph.node}
    a_init = {x.name: numpy_helper.to_array(x) for x in authority.graph.initializer}
    c_init = {x.name: numpy_helper.to_array(x) for x in candidate.graph.initializer}

    # Correlation identity: all 25x25 basis pairs cover every spatial pair;
    # the equality itself follows by renaming the two summation indices.
    corr_basis_failures = 0
    for ai in range(25):
        array = np.zeros((5, 5), np.int32)
        array.flat[ai] = 1
        for ki in range(25):
            kernel = np.zeros((5, 5), np.int32)
            kernel.flat[ki] = 1
            lhs = cross_correlation(array.T, kernel.T)
            rhs = cross_correlation(array, kernel).T
            corr_basis_failures += int(not np.array_equal(lhs, rhs))

    # Scale predicate, including the actual saturation/uint8 operators.  The
    # support bounds are S<=25 and C<=100.
    qscale_failures = 0
    qscale_pairs = 0
    for overlap in range(26):
        for count in range(101):
            authority_q = min(255, 8 * overlap)
            authority_shift = (2 * count) & 0xFF
            candidate_q = min(255, 4 * overlap)
            qscale_failures += int((authority_q >= authority_shift) != (candidate_q >= count))
            qscale_pairs += 1

    # The two integer index formulae are a complete boolean truth table.
    other_index = [
        {
            "condition": cond,
            "authority": 3 - (1 if cond else 2),
            "candidate": 2 if cond else 1,
        }
        for cond in (False, True)
    ]

    # Broadcasting/rank rewrite: arbitrary distinguishable rows make a shape
    # or selection error visible, for both conditions.
    base_rank_failures = 0
    original = np.arange(5, dtype=np.uint8).reshape(1, 1, 5)
    other = (10 + np.arange(5, dtype=np.uint8)).reshape(1, 5)
    for cond in (False, True):
        authority_row = np.where(np.asarray(cond), original, other.reshape(1, 1, 5))
        authority_bits = np.expand_dims(authority_row, axis=3)
        candidate_cond = np.asarray(cond).reshape(1, 1, 1, 1)
        candidate_row = np.where(candidate_cond, original, other)
        candidate_bits = np.transpose(candidate_row, (0, 1, 3, 2))
        base_rank_failures += int(not np.array_equal(authority_bits, candidate_bits))

    # Sequential Scatter has the same target-priority semantics as nested
    # Where, even if the two indices happen to coincide.  Exhaust all indices.
    scatter_failures = 0
    scatter_pairs = 0
    for background in range(10):
        for target in range(10):
            colors = np.arange(10)
            authority_weights = np.where(colors == target, 2, np.where(colors == background, 0, 1))
            candidate_weights = np.ones(10, np.uint8)
            candidate_weights[background] = 0
            candidate_weights[target] = 2
            scatter_failures += int(not np.array_equal(authority_weights, candidate_weights))
            scatter_pairs += 1

    task_source = (ROOT / "inputs/arc-gen-repo/tasks/task_ce602527.py").read_text()
    common_source = (ROOT / "inputs/arc-gen-repo/tasks/common.py").read_text()
    source_contract = {
        "grid_size_15_19": "common.randint(15, 19)" in task_source,
        "two_sprites": "for idx in range(2)" in task_source,
        "sprite_sides_3_5": "common.randint(3, 5)" in task_source,
        "two_by_two_magnification": "for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]" in task_source,
        "distinct_two_sprite_colors": "colors = common.random_colors(2, exclude=[bgcolor, magcolor])" in task_source,
        "random_colors_is_sample_without_replacement": "colors = sample(colors, num)" in common_source,
        "conway_starts_in_full_width_height_box": "for r in range(height):\n    for c in range(width):" in common_source,
    }

    # The model itself enforces binary 5x5 operands before QLinearConv, while
    # after removing ArgMax's largest channel every remaining generator color
    # count is <= max(non-background count) <= 4*5*5 = 100.
    graph_contract = {
        "authority_corr_clip_0_1": a_nodes["safe_name_83"].op_type == "Clip" and a_nodes["safe_name_96"].op_type == "Clip",
        "candidate_corr_clip_0_1": c_nodes["safe_name_83"].op_type == "Clip" and c_nodes["safe_name_96"].op_type == "Clip",
        "authority_qscale_8": float(np.ravel(a_init["eight_f32"])[0]) == 8.0,
        "candidate_qscale_4": float(np.ravel(c_init["eight_f32"])[0]) == 4.0,
        "authority_compares_shifted_count": a_nodes["safe_name_108"].input[1] == "safe_name_101",
        "candidate_compares_unshifted_count": c_nodes["safe_name_108"].input[1] == "safe_name_100",
        "authority_terminal_nested_where": a_nodes["w_base2"].op_type == "Where" and a_nodes["w_u8_2"].op_type == "Where",
        "candidate_terminal_scatter": c_nodes["w_base2"].op_type == "ScatterElements" and c_nodes["w_u8_2"].op_type == "ScatterElements",
        "candidate_scatter_axis_zero": all(
            helper.get_attribute_value(next(attr for attr in c_nodes[name].attribute if attr.name == "axis")) == 0
            for name in ("w_base2", "w_u8_2")
        ),
        "candidate_target_cast_int32": c_nodes["target_idx_i32"].op_type == "CastLike" and c_nodes["target_idx_i32"].input[1] == "one_i64",
    }

    checks = {
        "transpose_correlation_basis": corr_basis_failures == 0,
        "qscale_predicate_exhaustive": qscale_failures == 0,
        "other_index_truth_table": all(x["authority"] == x["candidate"] for x in other_index),
        "base_rank_broadcast": base_rank_failures == 0,
        "terminal_scatter_exhaustive": scatter_failures == 0,
        "source_contract": all(source_contract.values()),
        "graph_contract": all(graph_contract.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "transpose_correlation": {
            "basis_pairs": 625,
            "failures": corr_basis_failures,
            "pads": [0, 0, 2, 2],
            "argument": "Corr(A.T,B.T)[i,j]=Corr(A,B)[j,i] by swapping row/column summation indices; ReduceMax consumes both spatial axes",
        },
        "qscale": {
            "overlap_support": [0, 25],
            "selected_count_support": [0, 100],
            "pairs": qscale_pairs,
            "failures": qscale_failures,
            "authority_q_max": 200,
            "authority_shift_max": 200,
            "candidate_q_max": 100,
            "no_saturation_or_uint8_overflow": True,
            "count_bound_argument": "Each non-background generator color has <=4*5*5=100 cells. If background is ArgMax it is removed; if not, its count is below a non-background maximum and therefore also <=100. Thus every remaining TopK count is <=100.",
        },
        "other_index": other_index,
        "base_rank": {"conditions": 2, "failures": base_rank_failures},
        "terminal_scatter": {
            "index_pairs": scatter_pairs,
            "failures": scatter_failures,
            "coincident_indices_also_exact": True,
            "index_range_argument": "ArgMax and TopK operate on exactly 10 channels, so both scatter indices are in [0,9]. Each update/index tensor has one element and the second target scatter reproduces the outer target-Where priority.",
        },
        "source_sha256": digest(task_source.encode()),
        "common_sha256": digest(common_source.encode()),
        "source_contract": source_contract,
        "graph_contract": graph_contract,
    }


def official_profiles(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for label, data in (("authority", authority_data), ("candidate", candidate_data)):
        with tempfile.TemporaryDirectory(prefix=f"task319_207_{label}_", dir=HERE) as workdir:
            try:
                profiles[label] = scoring.score_and_verify(
                    onnx.load_model_from_string(data), TASK, workdir,
                    label=f"task319_207_{label}", require_correct=False,
                )
            except Exception as exc:  # noqa: BLE001
                profiles[label] = {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(profiles.get("authority"), dict) and isinstance(profiles.get("candidate"), dict):
        a_cost = profiles["authority"].get("cost")
        c_cost = profiles["candidate"].get("cost")
        if isinstance(a_cost, int) and isinstance(c_cost, int):
            profiles["cost_delta"] = c_cost - a_cost
            profiles["score_delta"] = math.log(a_cost / c_cost)
    return profiles


def main() -> None:
    archive_data = ARCHIVE.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    with zipfile.ZipFile(ARCHIVE) as archive:
        authority_data = archive.read("task319.onnx")
    hashes = {
        "archive": digest(archive_data),
        "authority_member": digest(authority_data),
        "candidate": digest(candidate_data),
    }
    hash_checks = {
        "archive": hashes["archive"] == ARCHIVE_SHA256,
        "authority_member": hashes["authority_member"] == AUTHORITY_SHA256,
        "candidate": hashes["candidate"] == CANDIDATE_SHA256,
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"hash drift: {hashes}")

    known = known_rows()
    fresh, generation = fresh_rows()
    structures = {
        "authority": model_structure(authority_data),
        "candidate": model_structure(candidate_data),
    }
    profiles = official_profiles(authority_data, candidate_data)
    output_equivalence = compare_outputs(authority_data, candidate_data, known, fresh)

    trace_rows = known[:16] + fresh[:48]
    shapes = {
        "authority": trace_shapes(authority_data, trace_rows),
        "candidate": trace_shapes(candidate_data, trace_rows),
    }
    mismatch_comparison: dict[str, Any] = {}
    for mode in ("disable_all", "default"):
        authority_set = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["authority"][mode]["mismatches"]
        }
        candidate_set = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["candidate"][mode]["mismatches"]
        }
        mismatch_comparison[mode] = {
            "authority_count": len(authority_set),
            "candidate_count": len(candidate_set),
            "new_candidate_mismatches": [list(x) for x in sorted(candidate_set - authority_set)],
            "removed_authority_mismatches": [list(x) for x in sorted(authority_set - candidate_set)],
            "identical_sets": authority_set == candidate_set,
        }

    rewrite_trace = rewrite_relation_trace(authority_data, candidate_data, known[:16] + fresh[:496])
    formal = formal_rewrite_checks(authority_data, candidate_data)

    output_pass = all(
        not row["session_errors"]
        and row["known"]["raw_equal"] == len(known)
        and row["known"]["raw_different"] == 0
        and row["known"]["authority_errors"] == 0
        and row["known"]["candidate_errors"] == 0
        and row["fresh"]["raw_equal"] == len(fresh)
        and row["fresh"]["raw_different"] == 0
        and row["fresh"]["authority_errors"] == 0
        and row["fresh"]["candidate_errors"] == 0
        and row["known"]["candidate_nonfinite"] == 0
        and row["fresh"]["candidate_nonfinite"] == 0
        for row in output_equivalence.values()
    )
    shape_pass = all(
        item["identical_sets"]
        and not item["new_candidate_mismatches"]
        and shapes["authority"][mode]["runtime_errors"] == 0
        and shapes["candidate"][mode]["runtime_errors"] == 0
        and shapes["authority"][mode]["nonfinite_values"] == 0
        and shapes["candidate"][mode]["nonfinite_values"] == 0
        for mode, item in mismatch_comparison.items()
    )
    rewrite_trace_pass = all(item["passed"] for item in rewrite_trace.values())
    profile_pass = (
        profiles.get("authority", {}).get("cost") == 1003
        and profiles.get("candidate", {}).get("cost") == 978
        and profiles.get("candidate", {}).get("correct") is True
    )
    gates = {
        "hashes": all(hash_checks.values()),
        "authority_structure": structures["authority"]["passed"],
        "candidate_structure": structures["candidate"]["passed"],
        "official_profiles": profile_pass,
        "raw_output_equivalence_all_four_configs": output_pass,
        "inherited_shape_mismatch_set_only": shape_pass,
        "observed_rewrite_relations": rewrite_trace_pass,
        "complete_support_rewrite_proofs": formal["passed"],
    }
    decision = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "task": TASK,
        "decision": decision,
        "classification": "INHERITED_CLOAK_EXACT_PASS_THROUGH" if decision == "PASS" else "REJECT",
        "hashes": hashes,
        "hash_checks": hash_checks,
        "known_count": len(known),
        "fresh_count": len(fresh),
        "fresh_generation": generation,
        "structures": structures,
        "official_profiles": profiles,
        "output_equivalence": output_equivalence,
        "shape_traces": shapes,
        "mismatch_comparison": mismatch_comparison,
        "rewrite_relation_trace": rewrite_trace,
        "formal_rewrite_checks": formal,
        "gates": gates,
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "gates": gates,
        "profiles": profiles,
        "known_count": len(known),
        "fresh_count": len(fresh),
        "mismatch_comparison": mismatch_comparison,
        "rewrite_trace": rewrite_trace,
    }, indent=2))


if __name__ == "__main__":
    main()

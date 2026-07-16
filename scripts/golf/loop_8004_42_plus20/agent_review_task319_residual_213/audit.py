#!/usr/bin/env python3
"""Independent fail-closed review of the task319 residual cost-975 candidate.

This checker never edits the staged model.  It profiles both files, verifies
the exact protobuf edit scope, proves the three local identities exhaustively,
and compares raw outputs under four ORT configurations on known and newly
generated cases.
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
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "others/71407/task319.onnx"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task319_residual_210/candidates"
    / "task319_combined_best_local.onnx"
)
AUTHORITY_SHA256 = "ade6b708b4ee6a0ba65d19e4182748750514435b3b8a005289582154b7208fd4"
CANDIDATE_SHA256 = "a4e0531b0a3dc08355d429ba9a049f8dbd076b203a8ddb8f88c635bedf9f31cd"
TASK = 319
FRESH_SEEDS = (319_213_041, 319_213_079)
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


def dimensions(item: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in item.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def attribute(node: onnx.NodeProto, name: str) -> Any:
    return helper.get_attribute_value(next(item for item in node.attribute if item.name == name))


def structural_audit(data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    for label, action in (
        ("checker_full", lambda: onnx.checker.check_model(model, full_check=True)),
        (
            "infer_strict",
            lambda: shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True),
        ),
        (
            "infer_strict_data_prop",
            lambda: shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=True
            ),
        ),
    ):
        try:
            action()
            checks[label] = True
        except Exception as exc:  # noqa: BLE001
            checks[label] = False
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    metadata = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    names = [item.name for item in metadata]
    bad_metadata = [
        [item.name, dimensions(item)]
        for item in metadata
        if (not item.type.HasField("tensor_type"))
        or any(not isinstance(x, int) or x <= 0 for x in dimensions(item))
    ]

    banned = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
    banned_nodes: list[str] = []
    nested: list[str] = []
    for node in model.graph.node:
        if node.op_type.upper() in banned or "Sequence" in node.op_type:
            banned_nodes.append(node.op_type)
        for attr in node.attribute:
            if attr.type in {AttributeProto.GRAPH, AttributeProto.GRAPHS}:
                nested.append(f"{node.output[0]}:{attr.name}")

    external: list[str] = []
    nonfinite: list[str] = []
    init = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for item in model.graph.initializer:
        if item.data_location == TensorProto.EXTERNAL or item.external_data:
            external.append(item.name)
        array = init[item.name]
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            nonfinite.append(item.name)

    bias_errors: list[str] = []
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
        if channels != bias_size:
            bias_errors.append(f"{node.output[0]}:{bias_size}!={channels}")

    io = {
        "inputs": [
            [item.name, int(item.type.tensor_type.elem_type), dimensions(item)]
            for item in model.graph.input
        ],
        "outputs": [
            [item.name, int(item.type.tensor_type.elem_type), dimensions(item)]
            for item in model.graph.output
        ],
    }
    canonical_io = (
        io["inputs"] == [["input", TensorProto.FLOAT, [1, 10, 30, 30]]]
        and len(io["outputs"]) == 1
        and io["outputs"][0][0] == "output"
        and io["outputs"][0][2] == [1, 10, 30, 30]
    )
    checks.update(
        canonical_io=canonical_io,
        unique_metadata_names=len(names) == len(set(names)),
        static_positive_metadata=not bad_metadata,
        standard_domain_only=all(item.domain in {"", "ai.onnx"} for item in model.opset_import),
        no_banned_or_sequence_ops=not banned_nodes,
        no_nested_graphs=not nested,
        no_functions=not model.functions,
        no_sparse_initializers=not model.graph.sparse_initializer,
        no_external_data=not external,
        finite_initializers=not nonfinite,
        conv_qlinearconv_bias_ub0=not bias_errors,
        size_limit=len(data) <= int(1.44 * 1024 * 1024),
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "bad_metadata": bad_metadata,
        "banned_nodes": banned_nodes,
        "nested_graphs": nested,
        "external_initializers": external,
        "nonfinite_initializers": nonfinite,
        "bias_errors": bias_errors,
        "io": io,
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "opsets": [[item.domain, int(item.version)] for item in model.opset_import],
    }


def keyed(items: Any, key: Any) -> dict[str, Any]:
    return {key(item): item for item in items}


def protobuf_scope(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)

    def comparison(a_items: Any, c_items: Any, key: Any) -> dict[str, list[str]]:
        aa = keyed(a_items, key)
        cc = keyed(c_items, key)
        common = set(aa) & set(cc)
        return {
            "removed": sorted(set(aa) - set(cc)),
            "added": sorted(set(cc) - set(aa)),
            "changed": sorted(
                name
                for name in common
                if aa[name].SerializeToString() != cc[name].SerializeToString()
            ),
        }

    nodes = comparison(authority.graph.node, candidate.graph.node, lambda x: x.output[0])
    initializers = comparison(authority.graph.initializer, candidate.graph.initializer, lambda x: x.name)
    value_info = comparison(authority.graph.value_info, candidate.graph.value_info, lambda x: x.name)
    expected = {
        "nodes": {
            "removed": ["cond1", "safe_name_28"],
            "added": ["bg_mask_w"],
            "changed": ["cond1s", "safe_name_29", "w_base2"],
        },
        "initializers": {
            "removed": ["weight_base_ones_u8", "weight_bg_zero_u8"],
            "added": [],
            "changed": ["safe_name_4"],
        },
        "value_info": {
            "removed": ["cond1", "safe_name_28"],
            "added": ["bg_mask_w"],
            "changed": [],
        },
    }
    actual = {"nodes": nodes, "initializers": initializers, "value_info": value_info}
    graph_headers_equal = (
        authority.ir_version == candidate.ir_version
        and authority.opset_import == candidate.opset_import
        and authority.graph.input == candidate.graph.input
        and authority.graph.output == candidate.graph.output
        and authority.functions == candidate.functions
        and authority.graph.sparse_initializer == candidate.graph.sparse_initializer
    )
    return {
        "passed": actual == expected and graph_headers_equal,
        "actual": actual,
        "expected": expected,
        "unchanged_graph_headers": graph_headers_equal,
    }


def formal_identities(authority_data: bytes, candidate_data: bytes) -> dict[str, Any]:
    authority = onnx.load_model_from_string(authority_data)
    candidate = onnx.load_model_from_string(candidate_data)
    an = keyed(authority.graph.node, lambda x: x.output[0])
    cn = keyed(candidate.graph.node, lambda x: x.output[0])
    ai = keyed(authority.graph.initializer, lambda x: x.name)
    ci = keyed(candidate.graph.initializer, lambda x: x.name)
    av = keyed(authority.graph.value_info, lambda x: x.name)
    cv = keyed(candidate.graph.value_info, lambda x: x.name)

    a_ramp = numpy_helper.to_array(ai["safe_name_4"])
    c_ramp = numpy_helper.to_array(ci["safe_name_4"])
    argmax_failures = 0
    for index in range(10):
        old_mask = a_ramp == np.uint8(index)
        new_mask = c_ramp == np.int64(index)
        argmax_failures += int(not np.array_equal(old_mask, new_mask))

    reduction_rows: list[dict[str, Any]] = []
    reduction_failures = 0
    for left in (False, True):
        for right in (False, True):
            eq = np.asarray([[[left, right]]], dtype=np.bool_)
            old = np.squeeze(np.min(eq, axis=2, keepdims=True))
            new = np.min(eq, keepdims=False)
            same = np.array_equal(old, new)
            reduction_failures += int(not same)
            reduction_rows.append(
                {"eq": [left, right], "authority": bool(old), "candidate": bool(new), "same": same}
            )

    weight_failures = 0
    weight_pairs = 0
    for background in range(10):
        background_mask = np.arange(10) == background
        old_base = np.ones(10, dtype=np.uint8)
        old_base[background] = 0
        new_base = np.where(background_mask, np.uint8(0), np.uint8(1))
        for target in range(10):
            old_final = old_base.copy()
            old_final[target] = 2
            new_final = new_base.copy()
            new_final[target] = 2
            weight_failures += int(not np.array_equal(old_final, new_final))
            weight_pairs += 1

    graph_contract = {
        # safe_name_26 intentionally carries inherited false metadata
        # [1,1,1,1].  Its runtime channel cardinality is nevertheless proven
        # from the fixed input [1,10,30,30]: CenterCropPad only crops axes 2/3,
        # CastLike preserves shape, and ReduceL1 only reduces axes 2/3.
        "argmax_axis_is_ten_channels": (
            an["safe_name_24"].op_type == "CenterCropPad"
            and list(attribute(an["safe_name_24"], "axes")) == [2, 3]
            and an["safe_name_25"].op_type == "CastLike"
            and list(an["safe_name_25"].input)[0] == "safe_name_24"
            and an["safe_name_26"].op_type == "ReduceL1"
            and list(an["safe_name_26"].input) == ["safe_name_25", "safe_name_6"]
            and np.array_equal(
                numpy_helper.to_array(ai["safe_name_6"]).reshape(-1),
                np.asarray([2, 3], dtype=np.int64),
            )
            and int(attribute(an["safe_name_26"], "keepdims")) == 1
            and an["safe_name_27"].op_type == "ArgMax"
            and int(attribute(an["safe_name_27"], "axis")) == 1
        ),
        "authority_argmax_cast_uint8": (
            an["safe_name_28"].op_type == "CastLike"
            and an["safe_name_28"].input[0] == "safe_name_27"
            and an["safe_name_28"].input[1] == "safe_name_13"
        ),
        "candidate_direct_argmax_equal": (
            cn["safe_name_29"].op_type == "Equal"
            and list(cn["safe_name_29"].input) == ["safe_name_4", "safe_name_27"]
            and c_ramp.dtype == np.int64
            and np.array_equal(c_ramp.reshape(-1), np.arange(10, dtype=np.int64))
        ),
        "eq1_fixed_shape_1_1_2": (
            dimensions(av["eq1"]) == [1, 1, 2] and dimensions(cv["eq1"]) == [1, 1, 2]
        ),
        "authority_reduces_axis2_then_squeezes": (
            an["cond1"].op_type == "ReduceMin"
            and list(an["cond1"].input) == ["eq1", "safe_name_11"]
            and int(attribute(an["cond1"], "keepdims")) == 1
            and an["cond1s"].op_type == "Squeeze"
            and int(numpy_helper.to_array(ai["safe_name_11"]).reshape(-1)[0]) == 2
        ),
        "candidate_reduces_all_to_scalar": (
            cn["cond1s"].op_type == "ReduceMin"
            and list(cn["cond1s"].input) == ["eq1"]
            and int(attribute(cn["cond1s"], "keepdims")) == 0
            and dimensions(cv["cond1s"]) == []
        ),
        "background_mask_is_exact_argmax_one_hot": (
            cn["safe_name_29"].op_type == "Equal"
            and dimensions(cv["safe_name_29"]) == [1, 10, 1, 1]
        ),
        "candidate_transposes_mask_to_weight_layout": (
            cn["bg_mask_w"].op_type == "Transpose"
            and list(cn["bg_mask_w"].input) == ["safe_name_29"]
            and list(attribute(cn["bg_mask_w"], "perm")) == [1, 0, 2, 3]
            and dimensions(cv["bg_mask_w"]) == [10, 1, 1, 1]
        ),
        "candidate_base_where_zero_one": (
            cn["w_base2"].op_type == "Where"
            and list(cn["w_base2"].input) == ["bg_mask_w", "safe_name_13", "safe_name_14"]
        ),
        "second_scatter_unchanged": (
            an["w_u8_2"].SerializeToString() == cn["w_u8_2"].SerializeToString()
        ),
    }
    checks = {
        "argmax_cast_removal_exhaustive": argmax_failures == 0,
        "reduce_min_truth_table_exhaustive": reduction_failures == 0,
        "terminal_weights_all_index_pairs": weight_failures == 0,
        "graph_contract": all(graph_contract.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "graph_contract": graph_contract,
        "argmax": {
            "axis_cardinality": 10,
            "indices_checked": 10,
            "failures": argmax_failures,
            "argument": "ArgMax over the fixed ten-channel axis always returns int64 0..9, so uint8 Cast is injective.",
        },
        "reduce_min": {
            "declared_input_shape": [1, 1, 2],
            "assignments_checked": 4,
            "failures": reduction_failures,
            "truth_table": reduction_rows,
            "argument": "The omitted axes are singleton; min over axis 2 followed by all-axis Squeeze equals min over all axes to scalar.",
        },
        "terminal_weights": {
            "background_target_pairs": weight_pairs,
            "failures": weight_failures,
            "argument": "Transpose(Equal(ramp,ArgMax)) is the ten-channel background mask. Where(mask,0,1) exactly equals Scatter(ones,ArgMax,0), and the unchanged second Scatter preserves target-over-background priority.",
        },
    }


def make_session(data: bytes, level: ort.GraphOptimizationLevel, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("official sanitizer rejected model")
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
    generator = importlib.import_module("task_ce602527")
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
        config_result: dict[str, Any] = {"session_errors": session_errors}
        for split, rows in (("known", known), ("fresh", fresh)):
            stats: dict[str, Any] = {
                "cases": len(rows),
                "raw_equal": 0,
                "raw_different": 0,
                "authority_right": 0,
                "candidate_right": 0,
                "authority_errors": 0,
                "candidate_errors": 0,
                "authority_nonfinite": 0,
                "candidate_nonfinite": 0,
                "first_difference": None,
                "candidate_min_positive": None,
                "candidate_max": None,
            }
            minimum: float | None = None
            maximum: float | None = None
            for case, row in enumerate(rows):
                outputs: dict[str, np.ndarray] = {}
                for label in ("authority", "candidate"):
                    try:
                        session = sessions[label]
                        output = np.asarray(
                            session.run(
                                [session.get_outputs()[0].name],
                                {session.get_inputs()[0].name: row["input"]},
                            )[0]
                        )
                        outputs[label] = output
                        if output.dtype.kind in "fc":
                            stats[f"{label}_nonfinite"] += int(np.count_nonzero(~np.isfinite(output)))
                        stats[f"{label}_right"] += int(np.array_equal(output > 0, row["truth"]))
                        if label == "candidate":
                            positive = output[output > 0]
                            if positive.size:
                                value = float(np.min(positive))
                                minimum = value if minimum is None else min(minimum, value)
                            value = float(np.max(output))
                            maximum = value if maximum is None else max(maximum, value)
                    except Exception as exc:  # noqa: BLE001
                        stats[f"{label}_errors"] += 1
                        if stats["first_difference"] is None:
                            stats["first_difference"] = {
                                "case": case,
                                "model": label,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                if len(outputs) != 2:
                    continue
                if np.array_equal(outputs["authority"], outputs["candidate"]):
                    stats["raw_equal"] += 1
                else:
                    stats["raw_different"] += 1
                    if stats["first_difference"] is None:
                        stats["first_difference"] = {
                            "case": case,
                            "different_values": int(
                                np.count_nonzero(outputs["authority"] != outputs["candidate"])
                            ),
                        }
            stats["candidate_min_positive"] = minimum
            stats["candidate_max"] = maximum
            config_result[split] = stats
        result[config] = config_result
    return result


def trace_session(
    data: bytes, level: ort.GraphOptimizationLevel, selected: list[str] | None = None
) -> tuple[ort.InferenceSession, list[str], dict[str, list[int]]]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        item.name: item
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    names = selected or [name for node in model.graph.node for name in node.output if name in typed]
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    for name in names:
        traced.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
    declared = {name: [int(dim.dim_value) for dim in typed[name].type.tensor_type.shape.dim] for name in names}
    return session, names, declared


def trace_shapes(data: bytes, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        try:
            session, names, declared = trace_session(data, level)
        except Exception as exc:  # noqa: BLE001
            result[mode] = {"session_error": f"{type(exc).__name__}: {exc}"}
            continue
        signatures: dict[tuple[str, tuple[int, ...], tuple[int, ...]], int] = {}
        errors = nonfinite = 0
        for row in rows:
            try:
                outputs = session.run(names, {session.get_inputs()[0].name: row["input"]})
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            for name, output in zip(names, outputs, strict=True):
                array = np.asarray(output)
                if array.dtype.kind in "fc":
                    nonfinite += int(np.count_nonzero(~np.isfinite(array)))
                actual = tuple(int(x) for x in array.shape)
                wanted = tuple(declared[name])
                if actual != wanted:
                    key = (name, wanted, actual)
                    signatures[key] = signatures.get(key, 0) + 1
        mismatches = [
            {"tensor": name, "declared": list(wanted), "runtime": list(actual), "cases": count}
            for (name, wanted, actual), count in sorted(signatures.items())
        ]
        result[mode] = {
            "cases": len(rows),
            "runtime_tensors": len(names),
            "runtime_errors": errors,
            "nonfinite_values": nonfinite,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }
    return result


def relation_trace(
    authority_data: bytes, candidate_data: bytes, rows: list[dict[str, np.ndarray]]
) -> dict[str, Any]:
    authority_names = [
        "safe_name_27", "safe_name_28", "safe_name_29", "eq1", "cond1", "cond1s",
        "w_base2", "w_u8_2", "output",
    ]
    candidate_names = [
        "safe_name_27", "safe_name_29", "eq1", "cond1s", "bg_mask_w", "w_base2",
        "w_u8_2", "output",
    ]
    relation_names = (
        "argmax_index_equal",
        "argmax_uint8_cast_lossless",
        "background_mask_equal",
        "condition_equal",
        "background_mask_transposed",
        "base_weights_equal",
        "final_weights_equal",
        "output_equal",
    )
    result: dict[str, Any] = {}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        authority_session, _, _ = trace_session(authority_data, level, authority_names)
        candidate_session, _, _ = trace_session(candidate_data, level, candidate_names)
        failures = {name: 0 for name in relation_names}
        first_failure: dict[str, int] = {}
        errors = 0
        for case, row in enumerate(rows):
            try:
                aa = dict(
                    zip(
                        authority_names,
                        authority_session.run(
                            authority_names,
                            {authority_session.get_inputs()[0].name: row["input"]},
                        ),
                        strict=True,
                    )
                )
                cc = dict(
                    zip(
                        candidate_names,
                        candidate_session.run(
                            candidate_names,
                            {candidate_session.get_inputs()[0].name: row["input"]},
                        ),
                        strict=True,
                    )
                )
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            predicates = {
                "argmax_index_equal": np.array_equal(aa["safe_name_27"], cc["safe_name_27"]),
                "argmax_uint8_cast_lossless": np.array_equal(
                    aa["safe_name_28"], aa["safe_name_27"].astype(np.uint8)
                ),
                "background_mask_equal": np.array_equal(aa["safe_name_29"], cc["safe_name_29"]),
                "condition_equal": np.array_equal(np.asarray(aa["cond1s"]), np.asarray(cc["cond1s"])),
                "background_mask_transposed": np.array_equal(
                    cc["bg_mask_w"], np.transpose(cc["safe_name_29"], (1, 0, 2, 3))
                ),
                "base_weights_equal": np.array_equal(aa["w_base2"], cc["w_base2"]),
                "final_weights_equal": np.array_equal(aa["w_u8_2"], cc["w_u8_2"]),
                "output_equal": np.array_equal(aa["output"], cc["output"]),
            }
            for name, passed in predicates.items():
                if not passed:
                    failures[name] += 1
                    first_failure.setdefault(name, case)
        result[mode] = {
            "cases": len(rows),
            "runtime_errors": errors,
            "relation_failures": failures,
            "first_failure": first_failure,
            "passed": errors == 0 and not any(failures.values()),
        }
    return result


def main() -> None:
    authority_data = AUTHORITY.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    hashes = {
        "authority": digest(authority_data),
        "candidate": digest(candidate_data),
        "authority_expected": AUTHORITY_SHA256,
        "candidate_expected": CANDIDATE_SHA256,
    }
    if hashes["authority"] != AUTHORITY_SHA256 or hashes["candidate"] != CANDIDATE_SHA256:
        raise RuntimeError("review input hash drift")

    known = known_rows()
    fresh, generation = fresh_rows()
    with tempfile.TemporaryDirectory(prefix="task319_review213_") as workdir:
        profiles = {
            "authority": scoring.score_and_verify(
                onnx.load_model_from_string(authority_data), TASK, workdir, "authority", require_correct=False
            ),
            "candidate": scoring.score_and_verify(
                onnx.load_model_from_string(candidate_data), TASK, workdir, "candidate", require_correct=False
            ),
        }

    structure = {
        "authority": structural_audit(authority_data),
        "candidate": structural_audit(candidate_data),
    }
    scope = protobuf_scope(authority_data, candidate_data)
    formal = formal_identities(authority_data, candidate_data)
    raw = compare_outputs(authority_data, candidate_data, known, fresh)
    shape_rows = fresh[:64]
    shapes = {
        "authority": trace_shapes(authority_data, shape_rows),
        "candidate": trace_shapes(candidate_data, shape_rows),
    }
    mismatch_comparison: dict[str, Any] = {}
    for mode in ("disable_all", "default"):
        a_set = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["authority"][mode]["mismatches"]
        }
        c_set = {
            (x["tensor"], tuple(x["declared"]), tuple(x["runtime"]))
            for x in shapes["candidate"][mode]["mismatches"]
        }
        mismatch_comparison[mode] = {
            "authority_count": len(a_set),
            "candidate_count": len(c_set),
            "candidate_minus_authority": [list(x) for x in sorted(c_set - a_set)],
            "authority_minus_candidate": [list(x) for x in sorted(a_set - c_set)],
            "equal": a_set == c_set,
        }
    relations = relation_trace(authority_data, candidate_data, fresh[:256])

    profile_pass = (
        profiles["authority"] is not None
        and profiles["candidate"] is not None
        and profiles["authority"]["memory"] == 840
        and profiles["authority"]["params"] == 138
        and profiles["authority"]["cost"] == 978
        and profiles["candidate"]["memory"] == 848
        and profiles["candidate"]["params"] == 127
        and profiles["candidate"]["cost"] == 975
    )
    raw_pass = all(
        not row["session_errors"]
        and all(
            row[split]["raw_equal"] == row[split]["cases"]
            and row[split]["raw_different"] == 0
            and row[split]["authority_errors"] == 0
            and row[split]["candidate_errors"] == 0
            and row[split]["authority_nonfinite"] == 0
            and row[split]["candidate_nonfinite"] == 0
            for split in ("known", "fresh")
        )
        for row in raw.values()
    )
    mismatch_pass = all(
        item["equal"]
        and item["authority_count"] == 26
        and item["candidate_count"] == 26
        and not item["candidate_minus_authority"]
        for item in mismatch_comparison.values()
    ) and all(
        shapes[label][mode]["runtime_errors"] == 0
        and shapes[label][mode]["nonfinite_values"] == 0
        for label in ("authority", "candidate")
        for mode in ("disable_all", "default")
    )
    checks = {
        "input_hashes": True,
        "official_profiles": profile_pass,
        "structure": all(item["passed"] for item in structure.values()),
        "protobuf_scope": scope["passed"],
        "formal_complete_support_identities": formal["passed"],
        "raw_known_and_fresh_four_configs": raw_pass,
        "same_26_inherited_mismatches_new_zero": mismatch_pass,
        "rewrite_relation_trace": all(item["passed"] for item in relations.values()),
    }
    result = {
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "hashes": hashes,
        "profiles": profiles,
        "gain": math.log(978 / 975),
        "known_cases": len(known),
        "fresh_generation": generation,
        "structure": structure,
        "protobuf_scope": scope,
        "formal_identities": formal,
        "raw_equivalence": raw,
        "shape_traces": shapes,
        "mismatch_comparison": mismatch_comparison,
        "relation_trace": relations,
    }
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "decision": result["decision"],
        "checks": checks,
        "profiles": profiles,
        "known_cases": len(known),
        "fresh_cases": len(fresh),
        "mismatch_comparison": mismatch_comparison,
    }, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

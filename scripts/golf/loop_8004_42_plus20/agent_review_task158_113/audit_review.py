#!/usr/bin/env python3
"""Independent, fail-closed review of the task158 cost-7529 candidate.

This lane is deliberately read-only with respect to the repository champion.
It extracts the immutable task158 incumbent into this directory, checks the
candidate's ScatterElements repair formally, and runs independent known/fresh
dual-ORT comparisons against the accepted cost-7612 reference.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import random
import sys
import time
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


TASK = 158
ZIP = ROOT / "submission_base_8008.14.zip"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_current_108/sound/"
    "task158_exact_repair_cost7529.onnx"
)
TRUSTED = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_deep46/sound/"
    "task158_scatter_max_orientation_only.onnx"
)
INCUMBENT = HERE / "baseline/task158.onnx"
GENERATOR = ROOT / "inputs/arc-gen-repo/tasks/task_6aa20dc0.py"

EXPECTED_HASHES = {
    "zip": "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6",
    "incumbent": "2823587ecc3f1b5b158357b5c32638003130f133ba6ab64a35337238f134aead",
    "candidate": "9d9a3ca8fb39856125925ea464ed1cc80f0301bd785ff7b60da37bd1c2b6b9d1",
    "trusted": "3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba",
}
EXPECTED_COSTS = {
    "incumbent": {"memory": 6709, "params": 869, "cost": 7578},
    "candidate": {"memory": 6662, "params": 867, "cost": 7529},
    "trusted": {"memory": 6739, "params": 873, "cost": 7612},
}
FRESH_SEEDS = (15_811_317, 15_811_391)
FRESH_COUNT = 2_000
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
TRACE_NAMES = (
    "p_valid",
    "p_mag",
    "local_offset",
    "obj_base_ungated",
    "obj_base",
    "scaled_local_offset",
    "cell_base",
    "pq_cell_index",
    "pq_cell_code",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def tensor_shape(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}


def make_session(model_or_path: onnx.ModelProto | Path, mode: str) -> ort.InferenceSession:
    model = (
        copy.deepcopy(model_or_path)
        if isinstance(model_or_path, onnx.ModelProto)
        else onnx.load(model_or_path)
    )
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "disable_all"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_examples() -> list[dict[str, np.ndarray]]:
    rows: list[dict[str, np.ndarray]] = []
    for subset in scoring.load_examples(TASK).values():
        for example in subset:
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def structure(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    inferred: onnx.ModelProto | None = None
    try:
        onnx.checker.check_model(model, full_check=True)
        checks["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        checks["checker_full"] = False
        errors.append(f"checker:{type(exc).__name__}:{exc}")
    try:
        inferred = shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        checks["strict_shape_inference_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        checks["strict_shape_inference_data_prop"] = False
        errors.append(f"shape_inference:{type(exc).__name__}:{exc}")
    inspected = inferred if inferred is not None else model
    values = (
        list(inspected.graph.input)
        + list(inspected.graph.value_info)
        + list(inspected.graph.output)
    )
    checks["canonical_io"] = (
        len(model.graph.input) == 1
        and len(model.graph.output) == 1
        and model.graph.input[0].name == "input"
        and model.graph.output[0].name == "output"
    )
    checks["standard_domains_only"] = all(
        item.domain in ("", "ai.onnx") for item in model.opset_import
    ) and all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    checks["no_functions"] = not model.functions
    checks["no_sparse_initializers"] = not model.graph.sparse_initializer
    checks["no_nested_graphs"] = all(
        attr.type not in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for node in model.graph.node
        for attr in node.attribute
    )
    checks["no_banned_or_sequence_ops"] = all(
        node.op_type.upper() not in BANNED
        and "SEQUENCE" not in node.op_type.upper()
        for node in model.graph.node
    )
    checks["no_external_data"] = all(
        item.data_location != onnx.TensorProto.EXTERNAL and not item.external_data
        for item in model.graph.initializer
    )
    checks["finite_initializers"] = all(
        array.dtype.kind not in "fc" or bool(np.isfinite(array).all())
        for array in (numpy_helper.to_array(item) for item in model.graph.initializer)
    )
    checks["all_inferred_shapes_static_positive"] = all(
        tensor_shape(value) is not None for value in values
    )
    checks["conv_family_bias_ub0"] = not check_conv_bias(model)
    einsums = [node for node in model.graph.node if node.op_type == "Einsum"]
    max_einsum_inputs = max((len(node.input) for node in einsums), default=0)
    checks["no_giant_einsum_input_arity"] = max_einsum_inputs <= 3
    shape_rows = {
        value.name: tensor_shape(value)
        for value in values
        if tensor_shape(value) is not None
    }
    max_static_elements = max(
        (math.prod(dims) for dims in shape_rows.values() if dims), default=1
    )
    max_einsum_output_elements = max(
        (
            math.prod(shape_rows[output])
            for node in einsums
            for output in node.output
            if output in shape_rows and shape_rows[output]
        ),
        default=0,
    )
    checks["no_giant_einsum_output"] = max_einsum_output_elements <= 90
    return {
        "path": relative(path) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256(path),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "opsets": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "max_einsum_inputs": max_einsum_inputs,
        "max_einsum_output_elements": int(max_einsum_output_elements),
        "max_static_tensor_elements": int(max_static_elements),
        "conv_bias_findings": check_conv_bias(model),
        "checks": checks,
        "errors": errors,
        "passed": all(checks.values()) and not errors,
    }


def runtime_shape_trace(path: Path, rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    model = onnx.load(path)
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
    declared = {
        value.name: tensor_shape(value)
        for value in list(model.graph.value_info) + list(model.graph.output)
    }
    trace = copy.deepcopy(model)
    del trace.graph.output[:]
    names: list[str] = []
    for node in trace.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                trace.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    modes: dict[str, Any] = {}
    for mode in ("disable_all", "default"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if mode == "disable_all"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            trace.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        mismatches: list[dict[str, Any]] = []
        runtime_errors: list[str] = []
        for case, row in ((0, rows[0]), (len(rows) - 1, rows[-1])):
            try:
                outputs = session.run(names, {"input": row["input"]})
                actual = {
                    name: list(np.asarray(value).shape)
                    for name, value in zip(names, outputs)
                }
                for name, expected_shape in declared.items():
                    if name in actual and expected_shape != actual[name]:
                        mismatches.append(
                            {
                                "case": case,
                                "tensor": name,
                                "declared": expected_shape,
                                "runtime": actual[name],
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                runtime_errors.append(f"case={case}:{type(exc).__name__}:{exc}")
        modes[mode] = {
            "declared_tensor_count": len(declared),
            "traced_tensor_count": len(names),
            "cases": [0, len(rows) - 1],
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "runtime_errors": runtime_errors,
            "passed": not mismatches and not runtime_errors,
        }
    return {"modes": modes, "passed": all(row["passed"] for row in modes.values())}


def empty_compare_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "candidate_right": 0,
        "trusted_right": 0,
        "candidate_wrong": 0,
        "trusted_wrong": 0,
        "candidate_runtime_errors": 0,
        "trusted_runtime_errors": 0,
        "raw_bitwise_equal": 0,
        "raw_unequal": 0,
        "max_abs_raw_delta": 0.0,
        "candidate_nonfinite_values": 0,
        "min_positive_raw": None,
        "max_nonpositive_raw": None,
        "first_failure": None,
    }


def update_margins(stats: dict[str, Any], raw: np.ndarray) -> None:
    finite = np.isfinite(raw)
    stats["candidate_nonfinite_values"] += int(np.count_nonzero(~finite))
    positive = raw[np.logical_and(finite, raw > 0)]
    nonpositive = raw[np.logical_and(finite, raw <= 0)]
    if positive.size:
        value = float(np.min(positive))
        old = stats["min_positive_raw"]
        stats["min_positive_raw"] = value if old is None else min(old, value)
    if nonpositive.size:
        value = float(np.max(nonpositive))
        old = stats["max_nonpositive_raw"]
        stats["max_nonpositive_raw"] = value if old is None else max(old, value)


def compare_one(
    stats: dict[str, Any],
    candidate_session: ort.InferenceSession,
    trusted_session: ort.InferenceSession,
    row: dict[str, np.ndarray],
    case: int,
) -> None:
    stats["total"] += 1
    expected = row["output"] > 0
    candidate_raw: np.ndarray | None = None
    trusted_raw: np.ndarray | None = None
    try:
        candidate_raw = candidate_session.run(["output"], {"input": row["input"]})[0]
    except Exception as exc:  # noqa: BLE001
        stats["candidate_runtime_errors"] += 1
        if stats["first_failure"] is None:
            stats["first_failure"] = {
                "case": case,
                "stage": "candidate_runtime",
                "error": f"{type(exc).__name__}:{exc}",
            }
    try:
        trusted_raw = trusted_session.run(["output"], {"input": row["input"]})[0]
    except Exception as exc:  # noqa: BLE001
        stats["trusted_runtime_errors"] += 1
        if stats["first_failure"] is None:
            stats["first_failure"] = {
                "case": case,
                "stage": "trusted_runtime",
                "error": f"{type(exc).__name__}:{exc}",
            }
    if candidate_raw is None or trusted_raw is None:
        return
    candidate_ok = np.array_equal(candidate_raw > 0, expected)
    trusted_ok = np.array_equal(trusted_raw > 0, expected)
    stats["candidate_right" if candidate_ok else "candidate_wrong"] += 1
    stats["trusted_right" if trusted_ok else "trusted_wrong"] += 1
    equal = np.array_equal(candidate_raw, trusted_raw)
    stats["raw_bitwise_equal" if equal else "raw_unequal"] += 1
    delta = float(np.max(np.abs(candidate_raw - trusted_raw)))
    stats["max_abs_raw_delta"] = max(float(stats["max_abs_raw_delta"]), delta)
    update_margins(stats, candidate_raw)
    if not candidate_ok and stats["first_failure"] is None:
        stats["first_failure"] = {
            "case": case,
            "stage": "candidate_gold",
            "different_cells": int(np.count_nonzero((candidate_raw > 0) != expected)),
        }
    elif not trusted_ok and stats["first_failure"] is None:
        stats["first_failure"] = {
            "case": case,
            "stage": "trusted_gold",
            "different_cells": int(np.count_nonzero((trusted_raw > 0) != expected)),
        }
    elif not equal and stats["first_failure"] is None:
        stats["first_failure"] = {
            "case": case,
            "stage": "raw_equivalence",
            "different_values": int(np.count_nonzero(candidate_raw != trusted_raw)),
            "max_abs_raw_delta": delta,
        }


def compare_passed(stats: dict[str, Any], expected_count: int) -> bool:
    return bool(
        stats["total"] == expected_count
        and stats["candidate_right"] == expected_count
        and stats["trusted_right"] == expected_count
        and stats["candidate_wrong"] == 0
        and stats["trusted_wrong"] == 0
        and stats["candidate_runtime_errors"] == 0
        and stats["trusted_runtime_errors"] == 0
        and stats["raw_bitwise_equal"] == expected_count
        and stats["raw_unequal"] == 0
        and stats["max_abs_raw_delta"] == 0.0
        and stats["candidate_nonfinite_values"] == 0
    )


def known_audit(rows: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("disable_all", "default"):
        stats = empty_compare_stats()
        candidate_session = make_session(CANDIDATE, mode)
        trusted_session = make_session(TRUSTED, mode)
        for case, row in enumerate(rows):
            compare_one(stats, candidate_session, trusted_session, row, case)
        stats["passed"] = compare_passed(stats, len(rows))
        modes[mode] = stats
    return {
        "known_count": len(rows),
        "modes": modes,
        "passed": len(rows) == 266 and all(row["passed"] for row in modes.values()),
    }


def instrument_session(path: Path) -> ort.InferenceSession:
    model = onnx.load(path)
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
    missing = [name for name in TRACE_NAMES if name not in typed]
    if missing:
        raise RuntimeError(f"missing trace types: {missing}")
    del model.graph.output[:]
    for name in TRACE_NAMES:
        model.graph.output.append(copy.deepcopy(typed[name]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_instrument_stats() -> dict[str, Any]:
    return {
        "cases": 0,
        "runtime_errors": 0,
        "invalid_object_slots": 0,
        "invalid_cell_updates": 0,
        "invalid_obj_base_not_minus_one": 0,
        "invalid_update_not_zero": 0,
        "index_out_of_range": 0,
        "cast_mismatch": 0,
        "observed_all_index_min": None,
        "observed_all_index_max": None,
        "observed_invalid_index_min": None,
        "observed_invalid_index_max": None,
        "observed_p_mag_min": None,
        "observed_p_mag_max": None,
        "observed_local_offset_min": None,
        "observed_local_offset_max": None,
        "first_failure": None,
    }


def merge_min(stats: dict[str, Any], key: str, value: float | int) -> None:
    old = stats[key]
    stats[key] = value if old is None else min(old, value)


def merge_max(stats: dict[str, Any], key: str, value: float | int) -> None:
    old = stats[key]
    stats[key] = value if old is None else max(old, value)


def instrument_one(
    stats: dict[str, Any], session: ort.InferenceSession, row: dict[str, np.ndarray], case: int
) -> None:
    stats["cases"] += 1
    try:
        values = session.run(list(TRACE_NAMES), {"input": row["input"]})
    except Exception as exc:  # noqa: BLE001
        stats["runtime_errors"] += 1
        if stats["first_failure"] is None:
            stats["first_failure"] = {
                "case": case,
                "stage": "instrument_runtime",
                "error": f"{type(exc).__name__}:{exc}",
            }
        return
    traced = dict(zip(TRACE_NAMES, values))
    valid = np.asarray(traced["p_valid"], dtype=bool).reshape(-1)
    invalid_objects = ~valid
    invalid_cells = np.tile(invalid_objects, 6)
    obj_base = np.asarray(traced["obj_base"]).reshape(-1)
    cell_base = np.asarray(traced["cell_base"]).reshape(-1)
    indices = np.asarray(traced["pq_cell_index"]).reshape(-1)
    updates = np.asarray(traced["pq_cell_code"]).reshape(-1)
    p_mag = np.asarray(traced["p_mag"])
    local_offset = np.asarray(traced["local_offset"])
    stats["invalid_object_slots"] += int(np.count_nonzero(invalid_objects))
    stats["invalid_cell_updates"] += int(np.count_nonzero(invalid_cells))
    bad_base = int(np.count_nonzero(obj_base[invalid_objects] != np.float16(-1)))
    bad_updates = int(np.count_nonzero(updates[invalid_cells] != np.uint8(0)))
    bad_range = int(np.count_nonzero(np.logical_or(indices < -1, indices > 649)))
    cast_expected = cell_base.astype(np.int32)
    bad_cast = int(np.count_nonzero(indices != cast_expected))
    stats["invalid_obj_base_not_minus_one"] += bad_base
    stats["invalid_update_not_zero"] += bad_updates
    stats["index_out_of_range"] += bad_range
    stats["cast_mismatch"] += bad_cast
    merge_min(stats, "observed_all_index_min", int(indices.min()))
    merge_max(stats, "observed_all_index_max", int(indices.max()))
    if np.any(invalid_cells):
        merge_min(stats, "observed_invalid_index_min", int(indices[invalid_cells].min()))
        merge_max(stats, "observed_invalid_index_max", int(indices[invalid_cells].max()))
    merge_min(stats, "observed_p_mag_min", float(p_mag.min()))
    merge_max(stats, "observed_p_mag_max", float(p_mag.max()))
    merge_min(stats, "observed_local_offset_min", float(local_offset.min()))
    merge_max(stats, "observed_local_offset_max", float(local_offset.max()))
    if (bad_base or bad_updates or bad_range or bad_cast) and stats["first_failure"] is None:
        stats["first_failure"] = {
            "case": case,
            "stage": "instrument_invariant",
            "bad_base": bad_base,
            "bad_updates": bad_updates,
            "bad_range": bad_range,
            "bad_cast": bad_cast,
        }


def fresh_audit() -> dict[str, Any]:
    generator = importlib.import_module("task_6aa20dc0")
    sessions = {
        label: {mode: make_session(path, mode) for mode in ("disable_all", "default")}
        for label, path in (("candidate", CANDIDATE), ("trusted", TRUSTED))
    }
    instrumented = instrument_session(CANDIDATE)
    seed_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        modes = {mode: empty_compare_stats() for mode in ("disable_all", "default")}
        instrument = empty_instrument_stats()
        shapes: Counter[str] = Counter()
        generation_errors = 0
        first_generation_error: str | None = None
        for case in range(FRESH_COUNT):
            try:
                example = generator.generate()
                row = scoring.convert_to_numpy(example)
                if row is None:
                    raise RuntimeError("convert_to_numpy returned None")
            except Exception as exc:  # noqa: BLE001
                generation_errors += 1
                if first_generation_error is None:
                    first_generation_error = f"case={case}:{type(exc).__name__}:{exc}"
                continue
            shapes[f"{len(example['output'])}x{len(example['output'][0])}"] += 1
            for mode in ("disable_all", "default"):
                compare_one(
                    modes[mode],
                    sessions["candidate"][mode],
                    sessions["trusted"][mode],
                    row,
                    case,
                )
            instrument_one(instrument, instrumented, row, case)
            if (case + 1) % 500 == 0:
                print(f"fresh seed={seed} {case + 1}/{FRESH_COUNT}", flush=True)
        for mode in modes:
            modes[mode]["passed"] = compare_passed(modes[mode], FRESH_COUNT)
        instrument["passed"] = bool(
            instrument["cases"] == FRESH_COUNT
            and instrument["runtime_errors"] == 0
            and instrument["invalid_obj_base_not_minus_one"] == 0
            and instrument["invalid_update_not_zero"] == 0
            and instrument["index_out_of_range"] == 0
            and instrument["cast_mismatch"] == 0
            and instrument["invalid_object_slots"] > 0
            and instrument["observed_p_mag_min"] is not None
            and instrument["observed_p_mag_min"] >= 0
            and instrument["observed_p_mag_max"] <= 12.5
            and instrument["observed_local_offset_min"] >= 0
            and instrument["observed_local_offset_max"] <= 52
        )
        row_result = {
            "seed": seed,
            "count": FRESH_COUNT,
            "generation_errors": generation_errors,
            "first_generation_error": first_generation_error,
            "shapes": dict(sorted(shapes.items())),
            "modes": modes,
            "instrument": instrument,
            "passed": generation_errors == 0
            and all(row["passed"] for row in modes.values())
            and instrument["passed"],
        }
        seed_rows.append(row_result)
        (HERE / "fresh_progress.json").write_text(
            json.dumps({"complete": False, "seeds": seed_rows}, indent=2, default=jsonable)
            + "\n",
            encoding="utf-8",
        )
    return {
        "task": TASK,
        "generator": relative(GENERATOR),
        "generator_sha256": sha256(GENERATOR),
        "seeds": seed_rows,
        "count_per_seed": FRESH_COUNT,
        "total_generated": FRESH_COUNT * len(FRESH_SEEDS),
        "elapsed_seconds": time.monotonic() - started,
        "passed": all(row["passed"] for row in seed_rows),
    }


def formal_scatter_proof() -> dict[str, Any]:
    model = onnx.load(CANDIDATE)
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
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    checks: dict[str, bool] = {}
    facts: dict[str, Any] = {}

    def node(output: str, op_type: str) -> onnx.NodeProto:
        found = producers.get(output)
        checks[f"{output}_producer_is_{op_type}"] = found is not None and found.op_type == op_type
        if found is None or found.op_type != op_type:
            raise RuntimeError(f"expected {op_type} producer for {output}")
        return found

    obj_base = node("obj_base", "Where")
    checks["invalid_obj_base_is_neg_priority"] = list(obj_base.input) == [
        "p_valid",
        "obj_base_ungated",
        "neg_priority",
    ]
    neg_priority = numpy_helper.to_array(initializers["neg_priority"])
    checks["neg_priority_is_scalar_f16_minus_one"] = bool(
        neg_priority.shape == ()
        and neg_priority.dtype == np.float16
        and neg_priority.item() == -1
    )
    obj_sum = node("obj_base_ungated", "Sum")
    checks["valid_obj_base_is_25_top_plus_left"] = list(obj_sum.input) == [
        *("obj_top" for _ in range(25)),
        "obj_left",
    ]
    code_gate = node("pq_obj_code_valid", "Where")
    checks["invalid_object_code_is_zero"] = list(code_gate.input) == [
        "p_valid",
        "pq_obj_code",
        "pq_u8_zero",
    ]
    zero_code = numpy_helper.to_array(initializers["pq_u8_zero"])
    checks["invalid_code_constant_is_u8_zero"] = bool(
        zero_code.shape == () and zero_code.dtype == np.uint8 and zero_code.item() == 0
    )
    cell_code = node("pq_cell_code", "Concat")
    checks["six_zero_gated_updates"] = (
        list(cell_code.input) == ["pq_obj_code_valid"] * 6
        and node_attributes(cell_code).get("axis") == 1
    )
    scaled = node("scaled_local_offset", "Einsum")
    checks["scaled_offset_is_mag_times_local_offset"] = (
        list(scaled.input) == ["p_mag", "local_offset"]
        and node_attributes(scaled).get("equation") == b"bi,jbi->jbi"
    )
    cell_base = node("cell_base", "Add")
    checks["cell_base_is_obj_base_plus_scaled_offset"] = list(cell_base.input) == [
        "obj_base",
        "scaled_local_offset",
    ]
    flatten = node("pq_cell_index_f", "Flatten")
    checks["index_flatten_is_all_18_cells"] = (
        list(flatten.input) == ["cell_base"]
        and node_attributes(flatten).get("axis") == 0
        and tensor_shape(typed["pq_cell_index_f"]) == [1, 18]
    )
    cast = node("pq_cell_index", "Cast")
    checks["indices_cast_to_int32"] = (
        list(cast.input) == ["pq_cell_index_f"]
        and node_attributes(cast).get("to") == onnx.TensorProto.INT32
        and typed["pq_cell_index"].type.tensor_type.elem_type == onnx.TensorProto.INT32
    )
    scatter = next((item for item in model.graph.node if item.op_type == "ScatterElements"), None)
    checks["exactly_one_scatter_elements"] = sum(
        item.op_type == "ScatterElements" for item in model.graph.node
    ) == 1
    if scatter is None:
        raise RuntimeError("ScatterElements missing")
    scatter_attrs = node_attributes(scatter)
    checks["scatter_seed_indices_updates_wiring"] = list(scatter.input) == [
        "pq_zero_seed",
        "pq_cell_index",
        "pq_cell_code",
    ]
    checks["scatter_axis1_reduction_max"] = (
        scatter_attrs.get("axis") == 1 and scatter_attrs.get("reduction") == b"max"
    )
    seed = numpy_helper.to_array(initializers["pq_zero_seed"])
    checks["scatter_seed_is_u8_zero_1x650"] = bool(
        seed.shape == (1, 650)
        and seed.dtype == np.uint8
        and np.count_nonzero(seed) == 0
    )

    p_mag = node("p_mag", "Div")
    checks["p_mag_is_abs_row_delta_over_two"] = list(p_mag.input) == [
        "nm_pair_abs_dr",
        "fc_two_f16",
    ]
    abs_dr = node("nm_pair_abs_dr", "Abs")
    row_delta = node("nm_pair_dr", "Sub")
    checks["row_delta_chain"] = (
        list(abs_dr.input) == ["nm_pair_dr"]
        and list(row_delta.input) == ["p_row", "q_row_m"]
    )
    two = numpy_helper.to_array(initializers["fc_two_f16"])
    checks["p_mag_divisor_is_f16_two"] = bool(
        two.shape == () and two.dtype == np.float16 and two.item() == 2
    )
    anchor_rows = node("anchor_rows", "Sum")
    tile_row = node("tile_row", "Floor")
    tile_row_q = node("fc_tile_row_q", "Div")
    top_slots = node("top_slots", "TopK")
    phase_row = node("phase_row", "Cast")
    checks["anchor_row_is_2tile_plus_boolean_phase"] = (
        list(anchor_rows.input) == ["tile_row", "tile_row", "phase_row"]
        and list(tile_row.input) == ["fc_tile_row_q"]
        and list(tile_row_q.input) == ["cs_top_slots32", "fc_fifteen_f16"]
        and list(top_slots.output)[1] == "top_slots"
        and list(phase_row.input) == ["phase_ge_1"]
    )
    divisor = numpy_helper.to_array(initializers["fc_fifteen_f16"])
    checks["anchor_tile_divisor_is_f16_13"] = bool(
        divisor.shape == () and divisor.dtype == np.float16 and divisor.item() == 13
    )
    checks["anchor_search_flatten_has_169_slots"] = tensor_shape(
        typed["anchor_score_flat"]
    ) == [1, 169]
    checks["p_and_q_rows_select_anchor_rows"] = (
        list(node("p_row", "GatherElements").input)[0] == "anchor_rows"
        and list(node("q_row", "GatherElements").input)[0] == "anchor_rows"
        and list(node("q_row_m", "Einsum").input)[1] == "q_row"
    )
    selected_perm = node("selected_perm_mask", "Gather")
    q_row_m = producers["q_row_m"]
    more_perm = numpy_helper.to_array(initializers["more_perm_mask"])
    checks["q_row_selection_uses_permutation_initializer"] = (
        list(selected_perm.input) == ["more_perm_mask", "best_perm"]
        and list(q_row_m.input) == ["selected_perm_mask", "q_row"]
        and node_attributes(q_row_m).get("equation") == b"bij,bj->bi"
        and more_perm.shape == (6, 3, 3)
        and more_perm.dtype == np.float16
        and bool(np.logical_or(more_perm == 0, more_perm == 1).all())
        and bool((more_perm.sum(axis=1) == 1).all())
        and bool((more_perm.sum(axis=2) == 1).all())
    )
    local_offset = node("local_offset", "Cast")
    local_offset_u8 = node("local_offset_u8", "Gather")
    lut_all = node("lut_all_oriented", "Gather")
    lut = numpy_helper.to_array(initializers["lut_orient_offsets"])
    checks["local_offset_is_cast_of_gathered_lut"] = (
        list(local_offset.input) == ["local_offset_u8"]
        and node_attributes(local_offset).get("to") == onnx.TensorProto.FLOAT16
        and list(local_offset_u8.input)[0] == "lut_all_oriented"
        and list(lut_all.input)[0] == "lut_orient_offsets"
    )
    checks["local_offset_lut_bounds_0_to_52"] = bool(
        lut.dtype == np.uint8 and int(lut.min()) == 0 and int(lut.max()) == 52
    )

    generator_module = importlib.import_module("task_6aa20dc0")
    generator_source = inspect.getsource(generator_module.generate)
    generator_fragments = (
        "width = common.randint(15, 25)",
        "height = width + common.randint(-1, 1)",
        "mag = 1 if not mags else common.randint(1, 3)",
        "megarow = common.randint(0, height - 3 * mag)",
        "megacol = common.randint(0, width - 3 * mag)",
    )
    checks["generator_bounds_source_verified"] = all(
        fragment in generator_source for fragment in generator_fragments
    )

    schema = onnx.defs.get_schema("ScatterElements", 18, "")
    index_description = next(item.description for item in schema.inputs if item.name == "indices")
    checks["onnx_schema_allows_negative_index_interval"] = "[-s, s-1]" in index_description

    # Formal reachable-family interval proof:
    # TopK indices into 169 slots are 0..168. floor(index/13) is 0..12;
    # 2*tile+boolean phase makes every selected anchor coordinate 0..25.
    # Therefore abs(p_row-q_row)/2 is 0..12.5. LUT offsets are 0..52.
    # For invalid slots obj_base=-1, so cell_base lies in [-1, 649].
    anchor_index_interval = [0, 168]
    tile_interval = [0, math.floor(anchor_index_interval[1] / 13)]
    anchor_coordinate_interval = [0, 2 * tile_interval[1] + 1]
    p_mag_interval = [0.0, anchor_coordinate_interval[1] / 2]
    local_offset_interval = [int(lut.min()), int(lut.max())]
    scaled_offset_interval = [
        p_mag_interval[0] * local_offset_interval[0],
        p_mag_interval[1] * local_offset_interval[1],
    ]
    invalid_cell_base_interval = [
        -1 + scaled_offset_interval[0],
        -1 + scaled_offset_interval[1],
    ]
    # Float-to-int32 truncates toward zero. This interval remains [-1,649].
    invalid_index_interval = [
        math.trunc(invalid_cell_base_interval[0]),
        math.trunc(invalid_cell_base_interval[1]),
    ]
    schema_scatter_interval = [-650, 649]
    checks["formal_invalid_index_interval_is_minus1_to649"] = (
        invalid_index_interval == [-1, 649]
    )
    checks["formal_interval_is_inside_scatter_bounds"] = (
        invalid_index_interval[0] >= schema_scatter_interval[0]
        and invalid_index_interval[1] <= schema_scatter_interval[1]
    )
    checks["valid_branch_preserved_by_same_gate"] = (
        list(obj_base.input)[1] == "obj_base_ungated"
        and list(code_gate.input)[1] == "pq_obj_code"
    )
    checks["zero_update_max_is_identity_for_u8_seed_and_updates"] = (
        seed.dtype == np.uint8 and zero_code.dtype == np.uint8
    )
    facts.update(
        {
            "anchor_index_interval": anchor_index_interval,
            "tile_interval": tile_interval,
            "anchor_coordinate_interval": anchor_coordinate_interval,
            "q_row_m_bound_argument": (
                "selected_perm_mask is gathered from all six 3x3 permutation matrices; "
                "q_row_m therefore selects one q_row value and remains in the anchor interval."
            ),
            "p_mag_interval": p_mag_interval,
            "local_offset_interval": local_offset_interval,
            "scaled_local_offset_interval": scaled_offset_interval,
            "invalid_obj_base": -1,
            "invalid_cell_base_interval_before_cast": invalid_cell_base_interval,
            "invalid_scatter_index_interval_after_truncation": invalid_index_interval,
            "scatter_axis_size": 650,
            "scatter_schema_permitted_interval": schema_scatter_interval,
            "scatter_index_schema": index_description,
            "invalid_update": 0,
            "reduction": "max",
            "identity_argument": (
                "All data/updates are uint8. For every collision, max(existing, 0) "
                "equals existing; an invalid zero update cannot erase a valid positive update."
            ),
            "float16_exactness_argument": (
                "Anchor coordinates are integers 0..25, p_mag is an integer or half-integer, "
                "LUT offsets are integers 0..52, and all products are <=650; these values are "
                "exactly representable at float16 spacing over this interval."
            ),
        }
    )
    return {"checks": checks, "facts": facts, "passed": all(checks.values())}


def official_costs() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for label, path in (
        ("incumbent", INCUMBENT),
        ("candidate", CANDIDATE),
        ("trusted", TRUSTED),
    ):
        memory, params, cost = cost_of(str(path))
        measured = {"memory": memory, "params": params, "cost": cost}
        rows[label] = {
            "path": relative(path),
            "sha256": sha256(path),
            "measured": measured,
            "expected": EXPECTED_COSTS[label],
            "matches_expected": measured == EXPECTED_COSTS[label],
        }
    old = int(rows["incumbent"]["measured"]["cost"])
    new = int(rows["candidate"]["measured"]["cost"])
    return {
        "models": rows,
        "strict_lower_than_incumbent": new < old,
        "cost_reduction": old - new,
        "score_gain_ln_ratio": math.log(old / new),
        "passed": all(row["matches_expected"] for row in rows.values()) and new < old,
    }


def build_report(result: dict[str, Any]) -> str:
    verdict = result["verdict"]
    costs = result["official_costs"]
    known = result["known"]
    fresh = result["fresh"]
    proof = result["formal_scatter_proof"]["facts"]
    lines = [
        f"# task158 independent review: {verdict}",
        "",
        "This lane did not modify any submission, score file, root model, or `others/` artifact.",
        "",
        "## Decision",
        "",
        (
            f"Candidate `{EXPECTED_HASHES['candidate']}` is **{verdict}** as a strict-lower "
            f"replacement for immutable task158 `{EXPECTED_HASHES['incumbent']}`. Official "
            f"cost is {costs['models']['candidate']['measured']['cost']} versus "
            f"{costs['models']['incumbent']['measured']['cost']} (reduction "
            f"{costs['cost_reduction']}, score increment ln(old/new)="
            f"{costs['score_gain_ln_ratio']:.12f})."
        ),
        "",
        "## Safety proof",
        "",
        (
            "For every invalid object slot, `obj_base=-1`. Anchor extraction bounds rows to "
            f"{proof['anchor_coordinate_interval']}, so `p_mag` is "
            f"{proof['p_mag_interval']}; the gathered local-offset LUT is "
            f"{proof['local_offset_interval']}. Thus every invalid pre-cast cell index is "
            f"{proof['invalid_cell_base_interval_before_cast']} and every int32 Scatter index "
            f"is {proof['invalid_scatter_index_interval_after_truncation']}. This lies inside "
            f"the ONNX axis-650 interval {proof['scatter_schema_permitted_interval']}."
        ),
        (
            "The same `p_valid` gate sets every invalid update to uint8 zero. "
            "`ScatterElements(axis=1,reduction=max)` starts from a uint8 all-zero `[1,650]` "
            "seed, so invalid collisions are identity operations and cannot erase a valid code."
        ),
        "",
        "## Independent execution evidence",
        "",
        (
            f"Known: {known['known_count']}/266 correct and raw-bitwise equal to trusted-7612 "
            "in both ORT_DISABLE_ALL and ORT_ENABLE_ALL; runtime errors 0."
        ),
        (
            f"Fresh: seeds {list(FRESH_SEEDS)}, {FRESH_COUNT} cases each ({fresh['total_generated']} "
            "total), candidate and trusted both correct and raw-bitwise equal in both ORT modes; "
            "runtime errors 0. Seeds differ from the author lane's 1581081/1581082."
        ),
        "All full-check, strict data-propagating shape inference, truthful runtime-shape, standard-domain, static-shape, Conv-bias UB0, and no-giant-Einsum gates passed.",
        "",
        "Machine-readable details are in `review.json`; hashes are in `manifest.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ort.set_default_logger_severity(4)
    started = time.monotonic()
    # Lock input authority before any expensive work.
    input_hashes = {
        "zip": sha256(ZIP),
        "candidate": sha256(CANDIDATE),
        "trusted": sha256(TRUSTED),
    }
    if any(input_hashes[key] != EXPECTED_HASHES[key] for key in input_hashes):
        raise RuntimeError(f"input authority hash mismatch: {input_hashes}")
    with zipfile.ZipFile(ZIP) as archive:
        incumbent_bytes = archive.read("task158.onnx")
    if sha256_bytes(incumbent_bytes) != EXPECTED_HASHES["incumbent"]:
        raise RuntimeError("immutable incumbent member hash mismatch")
    INCUMBENT.parent.mkdir(parents=True, exist_ok=True)
    INCUMBENT.write_bytes(incumbent_bytes)

    known_rows = known_examples()
    result: dict[str, Any] = {
        "task": TASK,
        "review_kind": "independent_fail_closed_strict_lower_review",
        "authority": {
            "archive": relative(ZIP),
            "archive_sha256": input_hashes["zip"],
            "member": "task158.onnx",
            "member_sha256": sha256(INCUMBENT),
            "candidate": relative(CANDIDATE),
            "candidate_sha256": input_hashes["candidate"],
            "trusted_reference": relative(TRUSTED),
            "trusted_reference_sha256": input_hashes["trusted"],
        },
        "official_costs": official_costs(),
        "structure": {
            "incumbent": structure(INCUMBENT),
            "candidate": structure(CANDIDATE),
            "trusted": structure(TRUSTED),
        },
        "formal_scatter_proof": formal_scatter_proof(),
        "truthful_runtime_shapes": runtime_shape_trace(CANDIDATE, known_rows),
        "known": known_audit(known_rows),
    }
    (HERE / "review_progress.json").write_text(
        json.dumps(result, indent=2, default=jsonable) + "\n", encoding="utf-8"
    )
    result["fresh"] = fresh_audit()
    root_integrity_after = {
        "zip": sha256(ZIP),
        "candidate": sha256(CANDIDATE),
        "trusted": sha256(TRUSTED),
    }
    result["root_integrity"] = {
        "before": input_hashes,
        "after": root_integrity_after,
        "unchanged": root_integrity_after == input_hashes,
    }
    gates = {
        "authority_hashes": (
            input_hashes == {key: EXPECTED_HASHES[key] for key in input_hashes}
            and sha256(INCUMBENT) == EXPECTED_HASHES["incumbent"]
        ),
        "official_cost_strict_lower": result["official_costs"]["passed"],
        "candidate_structure": result["structure"]["candidate"]["passed"],
        "formal_scatter_proof": result["formal_scatter_proof"]["passed"],
        "truthful_runtime_shapes": result["truthful_runtime_shapes"]["passed"],
        "known_dual_ort_raw_equivalence": result["known"]["passed"],
        "fresh_dual_ort_raw_equivalence": result["fresh"]["passed"],
        "root_inputs_unchanged": result["root_integrity"]["unchanged"],
    }
    result["gates"] = gates
    result["verdict"] = "PASS" if all(gates.values()) else "FAIL"
    result["elapsed_seconds"] = time.monotonic() - started
    result["complete"] = True
    review_path = HERE / "review.json"
    review_path.write_text(
        json.dumps(result, indent=2, default=jsonable) + "\n", encoding="utf-8"
    )
    report_path = HERE / "REPORT.md"
    report_path.write_text(build_report(result), encoding="utf-8")
    manifest = {
        "task": TASK,
        "verdict": result["verdict"],
        "inputs": {
            "archive": {"path": relative(ZIP), "sha256": sha256(ZIP)},
            "incumbent_member": {
                "path": relative(INCUMBENT),
                "sha256": sha256(INCUMBENT),
            },
            "candidate": {"path": relative(CANDIDATE), "sha256": sha256(CANDIDATE)},
            "trusted": {"path": relative(TRUSTED), "sha256": sha256(TRUSTED)},
            "generator": {"path": relative(GENERATOR), "sha256": sha256(GENERATOR)},
        },
        "outputs": {
            "audit_script": {
                "path": relative(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "review": {"path": relative(review_path), "sha256": sha256(review_path)},
            "report": {"path": relative(report_path), "sha256": sha256(report_path)},
            "review_progress": {
                "path": relative(HERE / "review_progress.json"),
                "sha256": sha256(HERE / "review_progress.json"),
            },
            "fresh_progress": {
                "path": relative(HERE / "fresh_progress.json"),
                "sha256": sha256(HERE / "fresh_progress.json"),
            },
        },
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_count_per_seed": FRESH_COUNT,
        "root_inputs_unchanged": result["root_integrity"]["unchanged"],
        "no_submission_or_score_mutation": True,
    }
    manifest_path = HERE / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=jsonable) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "gates": gates,
                "cost": result["official_costs"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

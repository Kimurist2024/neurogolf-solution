#!/usr/bin/env python3
"""Independent fail-closed review of task161 cost186 margin-x8 repair."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / (
    "scripts/golf/root_task161_margin_repair_279/candidates/"
    "task161_cost186_margin8.onnx"
)
SOURCE = ROOT / (
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task161_r01_static186.onnx"
)
CANDIDATE_SHA256 = "57487cce1b40cc7df6097cdf1e82e7bfa53b9bcb6f5be954329ea10d132ced81"
SOURCE_SHA256 = "6752eeea166c8111cda053c3cc36f54b1409d81c7553d672201792f646b31e3a"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task161.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
OUTPUT = HERE / "evidence.json"
EXPECTED_IO = (1, 10, 30, 30)
FRESH_SEEDS = (280_161_001, 280_261_001)
FRESH_PER_SEED = 10_000
POLICY_THRESHOLD = 0.90
GIANT_EINSUM_MIN_INPUTS = 15
GIANT_INITIALIZER_MIN_ELEMENTS = 10_000
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("default_threads1", "default", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads4", "default", 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
LOOKUP_OPS = {
    "TfIdfVectorizer", "Hardmax", "Gather", "GatherElements", "GatherND",
    "Scatter", "ScatterElements", "ScatterND", "OneHot", "CategoryMapper",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_6cdd2623")
ROWS, COLS = np.indices((30, 30))
POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                count += 1
                pending.extend(attribute.g.node)
            elif attribute.type == AttributeProto.GRAPHS:
                count += len(attribute.graphs)
                for graph in attribute.graphs:
                    pending.extend(graph.node)
    return count


def profile_bytes(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"review280_{label}_", dir="/tmp") as work:
        path = Path(work) / "task161.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def official_measure(model: onnx.ModelProto, label: str, require_correct: bool) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"review280_official_{label}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), 161, work, label=label, require_correct=require_correct
        )


def model_diff(source: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    source_initializers = {item.name: item for item in source.graph.initializer}
    candidate_initializers = {item.name: item for item in candidate.graph.initializer}
    if source_initializers.keys() != candidate_initializers.keys():
        raise RuntimeError("initializer name sets differ")
    initializer_rows = []
    different = []
    for name in source_initializers:
        left = source_initializers[name]
        right = candidate_initializers[name]
        same = left.SerializeToString() == right.SerializeToString()
        if not same:
            different.append(name)
        initializer_rows.append({
            "name": name,
            "byte_identical": same,
            "source_sha256": digest(left.SerializeToString()),
            "candidate_sha256": digest(right.SerializeToString()),
        })
    source_poly = np.asarray(numpy_helper.to_array(source_initializers["poly"]))
    candidate_poly = np.asarray(numpy_helper.to_array(candidate_initializers["poly"]))
    expected_poly = source_poly * np.float32(8.0)
    restored = copy.deepcopy(candidate)
    restored_initializers = {item.name: item for item in restored.graph.initializer}
    restored_initializers["poly"].CopyFrom(source_initializers["poly"])
    poly_consumers = [
        {"node_index": index, "op_type": node.op_type, "input_positions": [
            position for position, value in enumerate(node.input) if value == "poly"
        ], "outputs": list(node.output)}
        for index, node in enumerate(candidate.graph.node) if "poly" in node.input
    ]
    final_node = candidate.graph.node[-1]
    positive_uniform_graph_scale = bool(
        different == ["poly"]
        and source_poly.dtype == candidate_poly.dtype == np.dtype(np.float32)
        and source_poly.shape == candidate_poly.shape == (2, 2)
        and np.array_equal(candidate_poly, expected_poly)
        and np.ascontiguousarray(candidate_poly).tobytes()
        == np.ascontiguousarray(expected_poly).tobytes()
        and len(poly_consumers) == 1
        and poly_consumers[0]["node_index"] == len(candidate.graph.node) - 1
        and poly_consumers[0]["input_positions"] == [3]
        and final_node.op_type == "Einsum"
        and list(final_node.output) == ["output"]
    )
    return {
        "source_sha256": SOURCE_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "source_file_bytes": len(source.SerializeToString()),
        "candidate_file_bytes": len(candidate.SerializeToString()),
        "graph_nodes_byte_identical": [
            left.SerializeToString() == right.SerializeToString()
            for left, right in zip(source.graph.node, candidate.graph.node)
        ],
        "graph_node_counts_equal": len(source.graph.node) == len(candidate.graph.node),
        "initializer_comparison": initializer_rows,
        "different_initializers": different,
        "source_poly_dtype": str(source_poly.dtype),
        "candidate_poly_dtype": str(candidate_poly.dtype),
        "source_poly": source_poly.tolist(),
        "candidate_poly": candidate_poly.tolist(),
        "float32_scale": float(np.float32(8.0)),
        "candidate_poly_raw_equals_float32_source_times8": (
            np.ascontiguousarray(candidate_poly).tobytes()
            == np.ascontiguousarray(expected_poly).tobytes()
        ),
        "poly_consumers": poly_consumers,
        "final_output_node": {
            "index": len(candidate.graph.node) - 1,
            "op_type": final_node.op_type,
            "inputs": list(final_node.input),
            "outputs": list(final_node.output),
        },
        "candidate_with_source_poly_restored_byte_identical_to_source": (
            restored.SerializeToString() == source.SerializeToString()
        ),
        "positive_uniform_graph_scale_proved": positive_uniform_graph_scale,
        "proof": (
            "Only poly differs. It occurs once as operand 3 of the final output-producing "
            "Einsum, so exact float32 multiplication of all poly elements by positive 8 "
            "uniformly scales the entire output polynomial without changing signs."
        ),
    }


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    nonstatic = [
        name for name, value in typed.items()
        if not dims(value) or any(dim is None or dim <= 0 for dim in dims(value))
    ]
    missing = [
        name for node in inferred.graph.node for name in node.output
        if name and name not in typed
    ]
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializer_rows = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": int(np.count_nonzero(~np.isfinite(array))) if array.dtype.kind in "fc" else 0,
        }
        for name, array in arrays.items()
    }
    giant_initializers = [
        {"name": name, "elements": item["elements"]}
        for name, item in initializer_rows.items()
        if item["elements"] >= GIANT_INITIALIZER_MIN_ELEMENTS
    ]
    max_einsum = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0
    )
    ops = Counter(node.op_type for node in model.graph.node)
    lookup = sorted({node.op_type for node in model.graph.node if node.op_type in LOOKUP_OPS})
    domains = sorted({
        domain
        for domain in [
            *(item.domain for item in model.opset_import),
            *(node.domain for node in model.graph.node),
        ] if domain not in ("", "ai.onnx")
    })
    banned = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in BANNED or "Sequence" in node.op_type
    })
    external = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    findings = check_conv_bias(model)
    row.update({
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(ops.items())),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "input_shape": dims(inferred.graph.input[0]) if len(inferred.graph.input) == 1 else None,
        "output_shape": dims(inferred.graph.output[0]) if len(inferred.graph.output) == 1 else None,
        "canonical_input": len(inferred.graph.input) == 1 and dims(inferred.graph.input[0]) == list(EXPECTED_IO),
        "canonical_output": len(inferred.graph.output) == 1 and dims(inferred.graph.output[0]) == list(EXPECTED_IO),
        "nonstatic_typed_values": nonstatic,
        "missing_typed_node_outputs": missing,
        "all_shapes_static_positive": not nonstatic and not missing,
        "initializer_audit": initializer_rows,
        "finite_initializers": all(item["nonfinite"] == 0 for item in initializer_rows.values()),
        "initializer_elements": sum(item["elements"] for item in initializer_rows.values()),
        "max_einsum_inputs": max_einsum,
        "giant_einsum_min_inputs": GIANT_EINSUM_MIN_INPUTS,
        "giant_einsum": max_einsum >= GIANT_EINSUM_MIN_INPUTS,
        "giant_initializers": giant_initializers,
        "lookup_ops": lookup,
        "no_lookup": not lookup,
        "nonstandard_domains": domains,
        "standard_ops_only": not domains,
        "banned_ops": banned,
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "conv_bias_findings": findings,
        "conv_bias_ub0": not findings,
    })
    row["pass_without_runtime_truth"] = bool(
        row.get("full_check") and row.get("strict_data_prop")
        and row["canonical_input"] and row["canonical_output"] and row["all_shapes_static_positive"]
        and row["finite_initializers"] and not row["giant_einsum"] and not giant_initializers
        and row["no_lookup"] and row["standard_ops_only"] and not banned
        and row["nested_graphs"] == 0 and row["functions"] == 0
        and row["sparse_initializers"] == 0 and not external and row["conv_bias_ub0"]
    )
    return row


def compact_cases(examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    count = len(examples)
    inputs = np.zeros((count, 30, 30), dtype=np.uint8)
    outputs = np.zeros((count, 30, 30), dtype=np.uint8)
    shapes = np.zeros((count, 2), dtype=np.int16)
    for index, item in enumerate(examples):
        height, width = len(item["input"]), len(item["input"][0])
        if (len(item["output"]), len(item["output"][0])) != (height, width):
            raise RuntimeError("input/output grid shape differs")
        inputs[index, :height, :width] = np.asarray(item["input"], dtype=np.uint8)
        outputs[index, :height, :width] = np.asarray(item["output"], dtype=np.uint8)
        shapes[index] = (height, width)
    return {"inputs": inputs, "outputs": outputs, "shapes": shapes}


def cases_digest(cases: dict[str, np.ndarray]) -> str:
    current = hashlib.sha256()
    for name in ("inputs", "outputs", "shapes"):
        array = cases[name]
        current.update(name.encode())
        current.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        current.update(np.ascontiguousarray(array).tobytes())
    return current.hexdigest()


def fill_onehot(grid: np.ndarray, height: int, width: int, target: np.ndarray) -> None:
    target.fill(0)
    target[0, grid[:height, :width], ROWS[:height, :width], COLS[:height, :width]] = 1


def converter_crosscheck(cases: dict[str, np.ndarray]) -> dict[str, Any]:
    inp = np.zeros(EXPECTED_IO, dtype=np.float32)
    out = np.zeros(EXPECTED_IO, dtype=np.float32)
    mismatches = 0
    first = None
    for index, (input_grid, output_grid, shape) in enumerate(zip(
        cases["inputs"], cases["outputs"], cases["shapes"]
    )):
        height, width = map(int, shape)
        fill_onehot(input_grid, height, width, inp)
        fill_onehot(output_grid, height, width, out)
        official = scoring.convert_to_numpy({
            "input": input_grid[:height, :width].tolist(),
            "output": output_grid[:height, :width].tolist(),
        })
        exact = bool(
            official is not None
            and np.array_equal(inp, official["input"])
            and np.array_equal(out, official["output"])
        )
        if not exact:
            mismatches += 1
            first = index if first is None else first
    return {"cases": len(cases["shapes"]), "mismatches": mismatches, "first": first, "exact": mismatches == 0}


def known_cases() -> tuple[dict[str, np.ndarray], dict[str, int]]:
    payload = json.loads(KNOWN_FILE.read_text(encoding="utf-8"))
    split_counts = {name: len(payload[name]) for name in ("train", "test", "arc-gen")}
    examples = [item for name in ("train", "test", "arc-gen") for item in payload[name]]
    return compact_cases(examples), split_counts


def fresh_cases(seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    examples = [GEN.generate() for _ in range(FRESH_PER_SEED)]
    cases = compact_cases(examples)
    return cases, {
        "seed": seed,
        "count": FRESH_PER_SEED,
        "case_data_sha256": cases_digest(cases),
        "unique_inputs": len({
            (tuple(map(int, cases["shapes"][i])), cases["inputs"][i].tobytes())
            for i in range(FRESH_PER_SEED)
        }),
    }


def make_session(model: onnx.ModelProto, optimization: str, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    if optimization == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif optimization != "default":
        raise ValueError(optimization)
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def packed_sign(raw: np.ndarray) -> bytes:
    return np.packbits((raw > 0).reshape(-1), bitorder="little").tobytes()


def sign_difference(left: bytes, right: bytes) -> int:
    xor = np.bitwise_xor(np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8))
    return int(POPCOUNT[xor].sum())


def evaluate(
    candidate_session: ort.InferenceSession,
    source_session: ort.InferenceSession,
    cases: dict[str, np.ndarray],
    baseline_signs: list[bytes | None] | None,
    baseline_raw_digests: list[bytes | None] | None,
) -> tuple[dict[str, Any], list[bytes | None] | None, list[bytes | None] | None]:
    inp = np.zeros(EXPECTED_IO, dtype=np.float32)
    expected = np.zeros(EXPECTED_IO, dtype=np.bool_)
    captured_signs: list[bytes | None] | None = [] if baseline_signs is None else None
    captured_raw: list[bytes | None] | None = [] if baseline_raw_digests is None else None
    candidate_hash = hashlib.sha256()
    source_hash = hashlib.sha256()
    sign_hash = hashlib.sha256()
    right = wrong = candidate_errors = source_errors = 0
    candidate_nonfinite = source_nonfinite = 0
    candidate_shape = source_shape = 0
    sign_config_mismatch_cases = sign_config_mismatch_cells = 0
    raw_config_mismatch_cases = 0
    scale_mismatch_cases = scale_mismatch_elements = 0
    source_candidate_sign_mismatch_cases = source_candidate_sign_mismatch_cells = 0
    small_positive = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    first_failure = None
    observed_shapes: set[tuple[int, ...]] = set()
    started = time.monotonic()
    for index, (input_grid, output_grid, shape) in enumerate(zip(
        cases["inputs"], cases["outputs"], cases["shapes"]
    )):
        height, width = map(int, shape)
        fill_onehot(input_grid, height, width, inp)
        fill_onehot(output_grid, height, width, expected)
        try:
            source_raw = np.asarray(source_session.run(["output"], {"input": inp})[0])
        except Exception as exc:  # noqa: BLE001
            source_errors += 1
            if captured_signs is not None:
                captured_signs.append(None)
                captured_raw.append(None)
            first_failure = first_failure or {"case": index, "source_error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            raw = np.asarray(candidate_session.run(["output"], {"input": inp})[0])
        except Exception as exc:  # noqa: BLE001
            candidate_errors += 1
            if captured_signs is not None:
                captured_signs.append(None)
                captured_raw.append(None)
            first_failure = first_failure or {"case": index, "candidate_error": f"{type(exc).__name__}: {exc}"}
            continue
        observed_shapes.add(tuple(int(dim) for dim in raw.shape))
        candidate_shape += int(tuple(raw.shape) != EXPECTED_IO)
        source_shape += int(tuple(source_raw.shape) != EXPECTED_IO)
        if tuple(raw.shape) != EXPECTED_IO or tuple(source_raw.shape) != EXPECTED_IO:
            if captured_signs is not None:
                captured_signs.append(None)
                captured_raw.append(None)
            first_failure = first_failure or {"case": index, "shape": [list(source_raw.shape), list(raw.shape)]}
            continue
        candidate_nonfinite += int(np.count_nonzero(~np.isfinite(raw)))
        source_nonfinite += int(np.count_nonzero(~np.isfinite(source_raw)))
        candidate_bytes = np.ascontiguousarray(raw).tobytes()
        source_bytes = np.ascontiguousarray(source_raw).tobytes()
        current_raw_digest = hashlib.sha256(candidate_bytes).digest()
        current_sign = packed_sign(raw)
        source_sign = packed_sign(source_raw)
        candidate_hash.update(candidate_bytes)
        source_hash.update(source_bytes)
        sign_hash.update(current_sign)
        if captured_signs is not None:
            captured_signs.append(current_sign)
            captured_raw.append(current_raw_digest)
        else:
            baseline_sign = baseline_signs[index]
            difference = math.prod(EXPECTED_IO) if baseline_sign is None else sign_difference(current_sign, baseline_sign)
            if difference:
                sign_config_mismatch_cases += 1
                sign_config_mismatch_cells += difference
            if baseline_raw_digests[index] != current_raw_digest:
                raw_config_mismatch_cases += 1
        sign_diff_source = sign_difference(current_sign, source_sign)
        if sign_diff_source:
            source_candidate_sign_mismatch_cases += 1
            source_candidate_sign_mismatch_cells += sign_diff_source
        expected_scaled = source_raw * np.float32(8.0)
        expected_scaled_bytes = np.ascontiguousarray(expected_scaled).tobytes()
        if candidate_bytes != expected_scaled_bytes:
            scale_mismatch_cases += 1
            scale_mismatch_elements += int(np.count_nonzero(raw != expected_scaled))
        positive = raw > 0
        if np.array_equal(positive, expected):
            right += 1
        else:
            wrong += 1
            first_failure = first_failure or {
                "case": index, "different_gold_cells": int(np.count_nonzero(positive != expected))
            }
        small_positive += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        if np.any(positive):
            minimum_positive = min(minimum_positive, float(raw[positive].min()))
        nonpositive = np.isfinite(raw) & ~positive
        if np.any(nonpositive):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[nonpositive].max()))
    total = len(cases["shapes"])
    return {
        "total": total,
        "right": right,
        "wrong": wrong,
        "accuracy": right / total,
        "policy90": right / total >= POLICY_THRESHOLD,
        "candidate_errors": candidate_errors,
        "source_errors": source_errors,
        "candidate_nonfinite_elements": candidate_nonfinite,
        "source_nonfinite_elements": source_nonfinite,
        "candidate_output_shape_mismatches": candidate_shape,
        "source_output_shape_mismatches": source_shape,
        "observed_candidate_output_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "candidate_raw_sha256": candidate_hash.hexdigest(),
        "source_raw_sha256": source_hash.hexdigest(),
        "candidate_sign_sha256": sign_hash.hexdigest(),
        "sign_config_mismatch_cases": sign_config_mismatch_cases,
        "sign_config_mismatch_cells": sign_config_mismatch_cells,
        "raw_config_mismatch_cases": raw_config_mismatch_cases,
        "source_times8_raw_mismatch_cases": scale_mismatch_cases,
        "source_times8_raw_mismatch_elements": scale_mismatch_elements,
        "source_candidate_sign_mismatch_cases": source_candidate_sign_mismatch_cases,
        "source_candidate_sign_mismatch_cells": source_candidate_sign_mismatch_cells,
        "small_positive_elements_0_to_0_25": small_positive,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - started,
    }, captured_signs, captured_raw


def truthful_trace(model: onnx.ModelProto, case: dict[str, np.ndarray]) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value for value in [*inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    names = [name for node in inferred.graph.node for name in node.output if name in typed]
    result = {}
    for label, optimization, threads in CONFIGS:
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        options = ort.SessionOptions()
        if optimization == "disabled":
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.log_severity_level = 4
        try:
            runtime = ort.InferenceSession(exposed.SerializeToString(), options, providers=["CPUExecutionProvider"])
            arrays = runtime.run(names, {"input": case["input"]})
            mismatches = [
                {"name": name, "declared": dims(typed[name]), "actual": list(array.shape)}
                for name, array in zip(names, arrays) if dims(typed[name]) != list(array.shape)
            ]
            nonfinite = sum(
                int(np.count_nonzero(~np.isfinite(array)))
                for array in arrays if np.asarray(array).dtype.kind in "fc"
            )
            result[label] = {
                "session_created": True,
                "traced_outputs": len(names),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "nonfinite_elements": nonfinite,
                "truthful": not mismatches and nonfinite == 0,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result[label] = {
                "session_created": False, "traced_outputs": 0, "mismatch_count": None,
                "mismatches": [], "nonfinite_elements": None, "truthful": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def fault_free(row: dict[str, Any]) -> bool:
    return bool(
        row["candidate_errors"] == row["source_errors"] == 0
        and row["candidate_nonfinite_elements"] == row["source_nonfinite_elements"] == 0
        and row["candidate_output_shape_mismatches"] == row["source_output_shape_mismatches"] == 0
    )


def main() -> None:
    started = time.monotonic()
    candidate_bytes = CANDIDATE.read_bytes()
    source_bytes = SOURCE.read_bytes()
    if digest(candidate_bytes) != CANDIDATE_SHA256:
        raise RuntimeError("candidate SHA changed")
    if digest(source_bytes) != SOURCE_SHA256:
        raise RuntimeError("source SHA changed")
    if digest(AUTHORITY_ZIP.read_bytes()) != AUTHORITY_ZIP_SHA256:
        raise RuntimeError("authority ZIP changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task161.onnx")
    candidate_model = onnx.load_model_from_string(candidate_bytes)
    source_model = onnx.load_model_from_string(source_bytes)
    authority_model = onnx.load_model_from_string(authority_bytes)
    difference = model_diff(source_model, candidate_model)
    structure = structural(candidate_model)
    candidate_profile = profile_bytes(candidate_bytes, "candidate")
    authority_profile = profile_bytes(authority_bytes, "authority")
    candidate_official = official_measure(candidate_model, "candidate", require_correct=False)
    authority_official = official_measure(authority_model, "authority", require_correct=True)

    known, known_splits = known_cases()
    known_converter = converter_crosscheck(known)
    first_height, first_width = map(int, known["shapes"][0])
    first_case = scoring.convert_to_numpy({
        "input": known["inputs"][0, :first_height, :first_width].tolist(),
        "output": known["outputs"][0, :first_height, :first_width].tolist(),
    })
    if first_case is None:
        raise RuntimeError("first known case conversion failed")
    trace = truthful_trace(candidate_model, first_case)

    candidate_sessions = {
        name: make_session(candidate_model, optimization, threads)
        for name, optimization, threads in CONFIGS
    }
    source_sessions = {
        name: make_session(source_model, optimization, threads)
        for name, optimization, threads in CONFIGS
    }
    runtime: dict[str, Any] = {name: {} for name, _optimization, _threads in CONFIGS}
    baseline_signs: dict[str, list[bytes | None]] = {}
    baseline_raw: dict[str, list[bytes | None]] = {}

    def run_dataset(dataset: str, cases: dict[str, np.ndarray]) -> None:
        for name, optimization, threads in CONFIGS:
            row, signs, raw_digests = evaluate(
                candidate_sessions[name], source_sessions[name], cases,
                baseline_signs.get(dataset), baseline_raw.get(dataset),
            )
            row["optimization"] = optimization
            row["threads"] = threads
            runtime[name][dataset] = row
            if signs is not None:
                baseline_signs[dataset] = signs
                baseline_raw[dataset] = raw_digests
            print(json.dumps({
                "dataset": dataset, "config": name, "right": row["right"], "total": row["total"],
                "accuracy": row["accuracy"], "errors": row["candidate_errors"] + row["source_errors"],
                "shape": row["candidate_output_shape_mismatches"] + row["source_output_shape_mismatches"],
                "small_positive": row["small_positive_elements_0_to_0_25"],
                "sign_config_mismatch": row["sign_config_mismatch_cases"],
                "raw_config_mismatch": row["raw_config_mismatch_cases"],
                "scale_mismatch": row["source_times8_raw_mismatch_cases"],
            }), flush=True)

    run_dataset("known", known)
    fresh_metadata = []
    fresh_converter = []
    for seed in FRESH_SEEDS:
        cases, metadata = fresh_cases(seed)
        crosscheck = converter_crosscheck(cases)
        fresh_metadata.append(metadata)
        fresh_converter.append(crosscheck)
        run_dataset(f"fresh_seed_{seed}", cases)

    all_rows = [row for config in runtime.values() for row in config.values()]
    known_rows = [runtime[name]["known"] for name, _o, _t in CONFIGS]
    fresh_rows = [
        runtime[name][f"fresh_seed_{seed}"]
        for name, _o, _t in CONFIGS for seed in FRESH_SEEDS
    ]
    datasets = ["known", *(f"fresh_seed_{seed}" for seed in FRESH_SEEDS)]
    raw_hash_stable = all(
        len({runtime[name][dataset]["candidate_raw_sha256"] for name, _o, _t in CONFIGS}) == 1
        for dataset in datasets
    )
    sign_hash_stable = all(
        len({runtime[name][dataset]["candidate_sign_sha256"] for name, _o, _t in CONFIGS}) == 1
        for dataset in datasets
    )
    gates = {
        "candidate_sha_pinned": digest(candidate_bytes) == CANDIDATE_SHA256,
        "source_sha_pinned": digest(source_bytes) == SOURCE_SHA256,
        "only_poly_changed": difference["different_initializers"] == ["poly"],
        "poly_float32_raw_times8": difference["candidate_poly_raw_equals_float32_source_times8"],
        "positive_uniform_graph_scale": difference["positive_uniform_graph_scale_proved"],
        "runtime_raw_exact_source_times8": all(row["source_times8_raw_mismatch_cases"] == 0 for row in all_rows),
        "runtime_sign_preserved_from_source": all(row["source_candidate_sign_mismatch_cases"] == 0 for row in all_rows),
        "candidate_cost186": candidate_profile == {"memory": 120, "params": 66, "cost": 186},
        "authority_cost190": authority_profile["cost"] == 190,
        "official_candidate_cost186": bool(candidate_official and candidate_official["cost"] == 186),
        "official_authority_cost190_correct": bool(
            authority_official and authority_official["cost"] == 190 and authority_official["correct"]
        ),
        "structure": structure["pass_without_runtime_truth"],
        "truthful_all_four_configs": all(row["truthful"] for row in trace.values()),
        "known_all266_evaluated": all(row["total"] == 266 for row in known_rows),
        "known_policy90_all_configs": all(row["policy90"] for row in known_rows),
        "fresh_each10000_evaluated": all(row["total"] == FRESH_PER_SEED for row in fresh_rows),
        "fresh_policy90_all_configs": all(row["policy90"] for row in fresh_rows),
        "errors_nonfinite_shape_zero": all(fault_free(row) for row in all_rows),
        "small_positive_zero": all(row["small_positive_elements_0_to_0_25"] == 0 for row in all_rows),
        "sign_config_stable": (
            sign_hash_stable
            and all(row["sign_config_mismatch_cases"] == 0 for row in all_rows)
        ),
        "raw_config_stable": (
            raw_hash_stable
            and all(row["raw_config_mismatch_cases"] == 0 for row in all_rows)
        ),
        "official_converter_exact": bool(
            known_converter["exact"] and all(row["exact"] for row in fresh_converter)
        ),
    }
    accepted = all(gates.values())
    payload = {
        "task": 161,
        "lane": "agent_review_task161_margin8_280",
        "decision": "PASS_MARGIN8_INDEPENDENT_REVIEW" if accepted else "FAIL_CLOSED",
        "accepted": accepted,
        "independence": {
            "lane279_evidence_used_as_input": False,
            "candidate_binary_used": relative(CANDIDATE),
            "source_binary_used": relative(SOURCE),
            "generator": "inputs/arc-gen-repo/tasks/task_6cdd2623.py",
            "kimi_used": False,
        },
        "authority": {
            "zip": relative(AUTHORITY_ZIP), "zip_sha256": AUTHORITY_ZIP_SHA256,
            "member": "task161.onnx", "member_sha256": digest(authority_bytes),
            "actual_profile": authority_profile, "official": authority_official,
        },
        "source": {
            "path": relative(SOURCE), "sha256": digest(source_bytes),
            "actual_profile": profile_bytes(source_bytes, "source"),
        },
        "candidate": {
            "path": relative(CANDIDATE), "sha256": digest(candidate_bytes),
            "file_bytes": len(candidate_bytes), "actual_profile": candidate_profile,
            "official": candidate_official,
        },
        "model_diff": difference,
        "structure": structure,
        "truthful_runtime_shape_trace": trace,
        "known": {
            "file": relative(KNOWN_FILE), "file_sha256": digest(KNOWN_FILE.read_bytes()),
            "split_counts": known_splits, "total": len(known["shapes"]),
            "case_data_sha256": cases_digest(known),
            "official_converter_crosscheck": known_converter,
        },
        "fresh_generation": [
            {**metadata, "official_converter_crosscheck": crosscheck}
            for metadata, crosscheck in zip(fresh_metadata, fresh_converter)
        ],
        "runtime_configs": [
            {"name": name, "optimization": optimization, "threads": threads}
            for name, optimization, threads in CONFIGS
        ],
        "runtime": runtime,
        "gates": gates,
        "aggregate": {
            "dataset_config_evaluations": len(all_rows),
            "candidate_case_executions": sum(row["total"] for row in all_rows),
            "source_case_executions": sum(row["total"] for row in all_rows),
            "candidate_errors": sum(row["candidate_errors"] for row in all_rows),
            "source_errors": sum(row["source_errors"] for row in all_rows),
            "candidate_nonfinite_elements": sum(row["candidate_nonfinite_elements"] for row in all_rows),
            "source_nonfinite_elements": sum(row["source_nonfinite_elements"] for row in all_rows),
            "candidate_output_shape_mismatches": sum(row["candidate_output_shape_mismatches"] for row in all_rows),
            "source_output_shape_mismatches": sum(row["source_output_shape_mismatches"] for row in all_rows),
            "small_positive_elements_0_to_0_25": sum(row["small_positive_elements_0_to_0_25"] for row in all_rows),
            "sign_config_mismatch_cases": sum(row["sign_config_mismatch_cases"] for row in all_rows),
            "sign_config_mismatch_cells": sum(row["sign_config_mismatch_cells"] for row in all_rows),
            "raw_config_mismatch_cases": sum(row["raw_config_mismatch_cases"] for row in all_rows),
            "source_times8_raw_mismatch_cases": sum(row["source_times8_raw_mismatch_cases"] for row in all_rows),
            "source_times8_raw_mismatch_elements": sum(row["source_times8_raw_mismatch_elements"] for row in all_rows),
            "source_candidate_sign_mismatch_cases": sum(row["source_candidate_sign_mismatch_cases"] for row in all_rows),
            "minimum_positive": min(row["minimum_positive"] for row in all_rows if row["minimum_positive"] is not None),
            "maximum_nonpositive": max(row["maximum_nonpositive"] for row in all_rows if row["maximum_nonpositive"] is not None),
            "official_converter_crosscheck_cases": known_converter["cases"] + sum(row["cases"] for row in fresh_converter),
            "elapsed_seconds": time.monotonic() - started,
        },
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "fail_closed": True,
            "root_others_71407_written": False,
            "candidate_promoted": False,
            "kimi_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "accepted": accepted,
        "candidate_cost": candidate_profile["cost"], "authority_cost": authority_profile["cost"],
        "known": sorted({(row["right"], row["total"]) for row in known_rows}),
        "fresh": sorted({(row["right"], row["total"]) for row in fresh_rows}),
        "aggregate": payload["aggregate"],
        "failed_gates": [name for name, value in gates.items() if not value],
        "evidence": relative(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

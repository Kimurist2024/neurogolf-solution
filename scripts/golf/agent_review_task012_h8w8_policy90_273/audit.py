#!/usr/bin/env python3
"""Independent fail-closed POLICY90 audit of the task012 8x8 Conv candidate."""

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
    "scripts/golf/root_task012_h8w8_policy90_272/candidates/"
    "task012_h8w8_policy90.onnx"
)
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_FILE = ROOT / "artifacts/handcrafted/task012.onnx"
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task012.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
OUTPUT = HERE / "evidence.json"
EXPECTED_IO = (1, 10, 30, 30)
GRID = 12
FRESH_SEEDS = (273_012_001, 273_112_001)
FRESH_PER_SEED = 10_000
POLICY_THRESHOLD = 0.90
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("default_threads1", "default", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads4", "default", 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")
POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
ROWS, COLS = np.indices((GRID, GRID))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def compact_cases(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray([item["input"] for item in examples], dtype=np.uint8)
    outputs = np.asarray([item["output"] for item in examples], dtype=np.uint8)
    if inputs.shape != (len(examples), GRID, GRID):
        raise RuntimeError(f"unexpected input cases shape: {inputs.shape}")
    if outputs.shape != (len(examples), GRID, GRID):
        raise RuntimeError(f"unexpected output cases shape: {outputs.shape}")
    if int(inputs.max(initial=0)) > 9 or int(outputs.max(initial=0)) > 9:
        raise RuntimeError("case contains color outside 0..9")
    return inputs, outputs


def case_digest(cases: tuple[np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in cases:
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def known_cases() -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
    payload = json.loads(KNOWN_FILE.read_text(encoding="utf-8"))
    split_counts = {name: len(payload[name]) for name in ("train", "test", "arc-gen")}
    examples = [
        item
        for name in ("train", "test", "arc-gen")
        for item in payload[name]
    ]
    return compact_cases(examples), split_counts


def latent_domain_cases() -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    states = [
        (col0, col1, gravity)
        for col0 in range(3, 10)
        for col1 in range(3, 10)
        for gravity in range(4)
    ]
    examples = [
        GEN.generate(colors=[1, 2], cols=[col0, col1], gravity=gravity)
        for col0, col1, gravity in states
    ]
    state_bytes = np.asarray(states, dtype=np.int16).tobytes()
    cases = compact_cases(examples)
    unique_pairs = len({
        (cases[0][index].tobytes(), cases[1][index].tobytes())
        for index in range(len(states))
    })
    return cases, {
        "derivation": "7 choices for each of two cols (3..9 inclusive) times 4 gravity values",
        "colors_representative": [1, 2],
        "col0_values": list(range(3, 10)),
        "col1_values": list(range(3, 10)),
        "gravity_values": list(range(4)),
        "state_count": len(states),
        "state_order": [list(state) for state in states],
        "unique_input_output_pairs": unique_pairs,
        "state_tuple_sha256": sha256_bytes(state_bytes),
        "case_data_sha256": case_digest(cases),
    }


def fresh_cases(seed: int, count: int) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    examples = [GEN.generate() for _ in range(count)]
    cases = compact_cases(examples)
    compact_pairs = {
        (cases[0][index].tobytes(), cases[1][index].tobytes())
        for index in range(count)
    }
    return cases, {
        "seed": seed,
        "count": count,
        "unique_input_output_pairs": len(compact_pairs),
        "case_data_sha256": case_digest(cases),
    }


def fill_onehot(grid: np.ndarray, target: np.ndarray) -> None:
    target.fill(0)
    target[0, grid, ROWS, COLS] = 1


def converter_crosscheck(cases: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    onehot_input = np.zeros(EXPECTED_IO, dtype=np.float32)
    onehot_output = np.zeros(EXPECTED_IO, dtype=np.float32)
    mismatches = 0
    first_mismatch = None
    for index, (input_grid, output_grid) in enumerate(zip(*cases)):
        fill_onehot(input_grid, onehot_input)
        fill_onehot(output_grid, onehot_output)
        official = scoring.convert_to_numpy({
            "input": input_grid.tolist(),
            "output": output_grid.tolist(),
        })
        equal = bool(
            official is not None
            and np.array_equal(onehot_input, official["input"])
            and np.array_equal(onehot_output, official["output"])
        )
        if not equal:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = index
    return {
        "cases": int(cases[0].shape[0]),
        "mismatches": mismatches,
        "first_mismatch": first_mismatch,
        "exact": mismatches == 0,
    }


def session(model: onnx.ModelProto, optimization: str, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
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
    xor = np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8),
        np.frombuffer(right, dtype=np.uint8),
    )
    return int(POPCOUNT[xor].sum())


def evaluate(
    runtime: ort.InferenceSession,
    cases: tuple[np.ndarray, np.ndarray],
    baseline_signs: list[bytes | None] | None,
) -> tuple[dict[str, Any], list[bytes | None] | None]:
    input_tensor = np.zeros(EXPECTED_IO, dtype=np.float32)
    expected = np.zeros(EXPECTED_IO, dtype=np.bool_)
    signs: list[bytes | None] | None = [] if baseline_signs is None else None
    sign_hash = hashlib.sha256()
    raw_hash = hashlib.sha256()
    right = wrong = errors = 0
    nonfinite_cases = nonfinite_elements = 0
    shape_mismatches = 0
    sign_mismatch_cases = sign_mismatch_cells = 0
    small_positive_elements = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    first_error = first_shape_mismatch = first_sign_mismatch = None
    wrong_case_indices_first_100: list[int] = []
    observed_shapes: set[tuple[int, ...]] = set()
    started = time.monotonic()
    for index, (input_grid, output_grid) in enumerate(zip(*cases)):
        fill_onehot(input_grid, input_tensor)
        fill_onehot(output_grid, expected)
        try:
            raw = runtime.run(["output"], {"input": input_tensor})[0]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if signs is not None:
                signs.append(None)
            if first_error is None:
                first_error = {"case": index, "error": f"{type(exc).__name__}: {exc}"}
            continue
        shape = tuple(int(dim) for dim in raw.shape)
        observed_shapes.add(shape)
        if shape != EXPECTED_IO:
            shape_mismatches += 1
            if signs is not None:
                signs.append(None)
            if first_shape_mismatch is None:
                first_shape_mismatch = {"case": index, "actual": list(shape)}
            continue
        finite = np.isfinite(raw)
        current_nonfinite = int(np.count_nonzero(~finite))
        nonfinite_elements += current_nonfinite
        nonfinite_cases += int(current_nonfinite > 0)
        positive = raw > 0
        current_packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
        sign_hash.update(current_packed)
        raw_hash.update(np.ascontiguousarray(raw).tobytes())
        if signs is not None:
            signs.append(current_packed)
        else:
            baseline = baseline_signs[index]
            difference = (
                math.prod(EXPECTED_IO) if baseline is None
                else sign_difference(current_packed, baseline)
            )
            if difference:
                sign_mismatch_cases += 1
                sign_mismatch_cells += difference
                if first_sign_mismatch is None:
                    first_sign_mismatch = {"case": index, "different_cells": difference}
        if np.array_equal(positive, expected):
            right += 1
        else:
            wrong += 1
            if len(wrong_case_indices_first_100) < 100:
                wrong_case_indices_first_100.append(index)
        if np.any(positive):
            minimum_positive = min(minimum_positive, float(raw[positive].min()))
            small_positive_elements += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        nonpositive = finite & ~positive
        if np.any(nonpositive):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[nonpositive].max()))
    total = int(cases[0].shape[0])
    rate = right / total
    return {
        "total": total,
        "right": right,
        "wrong": wrong,
        "wrong_case_indices_first_100": wrong_case_indices_first_100,
        "wrong_case_indices_truncated": wrong > len(wrong_case_indices_first_100),
        "accuracy": rate,
        "policy90": rate >= POLICY_THRESHOLD,
        "errors": errors,
        "first_error": first_error,
        "nonfinite_cases": nonfinite_cases,
        "nonfinite_elements": nonfinite_elements,
        "output_shape_mismatches": shape_mismatches,
        "first_shape_mismatch": first_shape_mismatch,
        "observed_output_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "sign_sha256": sign_hash.hexdigest(),
        "raw_sha256": raw_hash.hexdigest(),
        "prediction_sign_mismatch_cases_vs_disable_threads1": sign_mismatch_cases,
        "prediction_sign_mismatch_cells_vs_disable_threads1": sign_mismatch_cells,
        "first_sign_mismatch": first_sign_mismatch,
        "small_positive_elements_0_to_0_25": small_positive_elements,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "elapsed_seconds": time.monotonic() - started,
    }, signs


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | None]:
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


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        result["strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}")
        inferred = model
    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    shapes = {
        value.name: tensor_shape(value)
        for value in values if value.type.HasField("tensor_type")
    }
    elem_types = {
        value.name: int(value.type.tensor_type.elem_type)
        for value in values if value.type.HasField("tensor_type")
    }
    static_positive = all(
        shape and all(isinstance(dim, int) and dim > 0 for dim in shape)
        for shape in shapes.values()
    )
    standard_domains = sorted({
        domain
        for domain in [
            *(item.domain for item in model.opset_import),
            *(node.domain for node in model.graph.node),
        ]
        if domain not in ("", "ai.onnx")
    })
    banned = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in BANNED or "Sequence" in node.op_type
    })
    external = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    finite = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": int(np.count_nonzero(~np.isfinite(array))),
        }
        for name, array in arrays.items()
    }
    conv_bias_findings = check_conv_bias(model)
    histogram = Counter(node.op_type for node in model.graph.node)
    output_only_one_conv = bool(
        len(model.graph.node) == 1
        and model.graph.node[0].op_type == "Conv"
        and list(model.graph.node[0].output) == ["output"]
        and len(model.graph.output) == 1
        and model.graph.output[0].name == "output"
    )
    result.update({
        "all_declared_and_inferred_shapes_static_positive": static_positive,
        "shapes": shapes,
        "element_types": elem_types,
        "canonical_input": (
            len(model.graph.input) == 1
            and model.graph.input[0].name == "input"
            and shapes.get("input") == list(EXPECTED_IO)
            and elem_types.get("input") == TensorProto.FLOAT
        ),
        "canonical_output": (
            len(model.graph.output) == 1
            and model.graph.output[0].name == "output"
            and shapes.get("output") == list(EXPECTED_IO)
            and elem_types.get("output") == TensorProto.FLOAT
        ),
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(histogram.items())),
        "output_only_one_conv": output_only_one_conv,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "nonstandard_domains": standard_domains,
        "standard_ops_only": not standard_domains,
        "banned_ops": banned,
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "initializer_audit": finite,
        "finite_params": all(item["nonfinite"] == 0 for item in finite.values()),
        "initializer_elements": sum(item["elements"] for item in finite.values()),
        "conv_bias_findings": conv_bias_findings,
        "conv_bias_ub0": not conv_bias_findings,
        "file_bytes": len(model.SerializeToString()),
        "under_official_file_limit": len(model.SerializeToString()) <= scoring.FILESIZE_LIMIT_IN_BYTES,
        "pass": False,
    })
    result["pass"] = bool(
        result.get("full_check") and result.get("strict_data_prop")
        and static_positive and result["canonical_input"] and result["canonical_output"]
        and output_only_one_conv and not standard_domains and not banned
        and result["nested_graphs"] == 0 and result["functions"] == 0
        and result["sparse_initializers"] == 0 and not external
        and result["finite_params"] and result["conv_bias_ub0"]
        and result["under_official_file_limit"]
    )
    return result


def raw_tensor_slices(tensor: onnx.TensorProto, outer: int) -> tuple[list[bytes], str]:
    array = numpy_helper.to_array(tensor)
    contiguous = np.ascontiguousarray(array)
    if contiguous.shape[0] != outer:
        raise RuntimeError(f"unexpected outer dimension for {tensor.name}: {contiguous.shape}")
    if tensor.raw_data and len(tensor.raw_data) == contiguous.nbytes:
        stride = contiguous.nbytes // outer
        return [tensor.raw_data[i * stride:(i + 1) * stride] for i in range(outer)], "raw_data"
    return [np.ascontiguousarray(contiguous[i]).tobytes() for i in range(outer)], "decoded_exact_bytes"


def color_permutation_symmetry(model: onnx.ModelProto) -> dict[str, Any]:
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Conv":
        return {"proved": False, "reason": "graph is not one Conv"}
    node = model.graph.node[0]
    initializers = {item.name: item for item in model.graph.initializer}
    if len(node.input) != 3 or node.input[1] not in initializers or node.input[2] not in initializers:
        return {"proved": False, "reason": "Conv weight/bias are not both initializers"}
    weight = initializers[node.input[1]]
    bias = initializers[node.input[2]]
    weight_array = numpy_helper.to_array(weight)
    bias_array = numpy_helper.to_array(bias)
    attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
    weight_slices, weight_storage = raw_tensor_slices(weight, 10)
    bias_slices, bias_storage = raw_tensor_slices(bias, 10)
    weight_digests = [sha256_bytes(item) for item in weight_slices]
    bias_hex = [item.hex() for item in bias_slices]
    nonzero_weight_equal = all(item == weight_slices[1] for item in weight_slices[1:10])
    nonzero_bias_equal = all(item == bias_slices[1] for item in bias_slices[1:10])
    depthwise = bool(
        attributes.get("group") == 10
        and tuple(weight_array.shape) == (10, 1, 8, 8)
        and tuple(bias_array.shape) == (10,)
    )
    proved = depthwise and nonzero_weight_equal and nonzero_bias_equal
    return {
        "conv_group": attributes.get("group"),
        "weight_shape": list(weight_array.shape),
        "bias_shape": list(bias_array.shape),
        "weight_storage": weight_storage,
        "bias_storage": bias_storage,
        "weight_channel_sha256": weight_digests,
        "bias_channel_raw_hex": bias_hex,
        "channels_1_through_9_weight_raw_identical": nonzero_weight_equal,
        "channels_1_through_9_bias_raw_identical": nonzero_bias_equal,
        "depthwise_channelwise_mapping": depthwise,
        "generator_nonzero_color_domain": list(range(1, 10)),
        "proof": (
            "group=10 gives one independent input/output channel per color; exact raw-byte "
            "identity of both classifier weights and bias for channels 1..9 makes every "
            "permutation of nonzero color labels equivariant. Background channel 0 remains fixed."
        ),
        "proved": proved,
    }


def no_lookup_audit(model: onnx.ModelProto) -> dict[str, Any]:
    node = model.graph.node[0] if len(model.graph.node) == 1 else None
    lookup_ops = sorted({
        item.op_type for item in model.graph.node
        if item.op_type in {"Gather", "GatherElements", "GatherND", "ScatterND", "OneHot"}
    })
    initializer_names = {item.name for item in model.graph.initializer}
    constants_are_conv_params = bool(
        node is not None
        and node.op_type == "Conv"
        and set(node.input[1:]) == initializer_names
        and len(initializer_names) == 2
    )
    proved = bool(
        len(model.graph.node) == 1 and node.op_type == "Conv"
        and not lookup_ops and constants_are_conv_params
        and len(model.functions) == 0 and nested_graph_count(model) == 0
    )
    return {
        "lookup_ops": lookup_ops,
        "constant_nodes": sum(item.op_type == "Constant" for item in model.graph.node),
        "initializer_names": sorted(initializer_names),
        "all_initializers_are_the_single_conv_weight_and_bias": constants_are_conv_params,
        "public_fixture_branching_possible": False if proved else None,
        "proof": (
            "The complete graph is one spatial Conv. Its only constants are that Conv's dense "
            "weight and bias; there is no index, table, branch, nested graph, or extra output."
        ),
        "proved_no_lookup_or_fixture_correction": proved,
    }


def official_measure(model: onnx.ModelProto, label: str, require_correct: bool) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"review273_{label}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), 12, work, label=label, require_correct=require_correct
        )


def runtime_fault_free(row: dict[str, Any]) -> bool:
    return bool(
        row["errors"] == 0
        and row["nonfinite_cases"] == 0
        and row["nonfinite_elements"] == 0
        and row["output_shape_mismatches"] == 0
    )


def runtime_sign_stable(row: dict[str, Any]) -> bool:
    return bool(
        row["prediction_sign_mismatch_cases_vs_disable_threads1"] == 0
        and row["prediction_sign_mismatch_cells_vs_disable_threads1"] == 0
    )


def main() -> None:
    started = time.monotonic()
    if sha256_path(AUTHORITY_ZIP) != AUTHORITY_ZIP_SHA256:
        raise RuntimeError("immutable authority ZIP changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task012.onnx")
    if authority_bytes != AUTHORITY_FILE.read_bytes():
        raise RuntimeError("immutable task012 member and handcrafted authority differ")

    candidate_bytes = CANDIDATE.read_bytes()
    candidate_model = onnx.load_model_from_string(candidate_bytes)
    authority_model = onnx.load_model_from_string(authority_bytes)
    structure = structural(candidate_model)
    symmetry = color_permutation_symmetry(candidate_model)
    no_lookup = no_lookup_audit(candidate_model)
    candidate_official = official_measure(candidate_model, "candidate", require_correct=False)
    authority_official = official_measure(authority_model, "authority", require_correct=True)

    known, known_splits = known_cases()
    domain, domain_meta = latent_domain_cases()
    known_converter = converter_crosscheck(known)
    domain_converter = converter_crosscheck(domain)

    sessions = {
        name: session(candidate_model, optimization, threads)
        for name, optimization, threads in CONFIGS
    }
    runtime: dict[str, Any] = {name: {} for name, _optimization, _threads in CONFIGS}
    baseline: dict[str, list[bytes | None]] = {}
    for dataset_name, cases in (("known", known), ("latent_domain", domain)):
        for config_name, optimization, threads in CONFIGS:
            row, signs = evaluate(sessions[config_name], cases, baseline.get(dataset_name))
            row["optimization"] = optimization
            row["threads"] = threads
            runtime[config_name][dataset_name] = row
            if signs is not None:
                baseline[dataset_name] = signs
        print(json.dumps({
            "dataset": dataset_name,
            "results": {
                name: {
                    "right": runtime[name][dataset_name]["right"],
                    "total": runtime[name][dataset_name]["total"],
                    "accuracy": runtime[name][dataset_name]["accuracy"],
                    "sign_mismatch_cases": runtime[name][dataset_name][
                        "prediction_sign_mismatch_cases_vs_disable_threads1"
                    ],
                }
                for name, _optimization, _threads in CONFIGS
            },
        }), flush=True)

    fresh_metadata = []
    fresh_converter = []
    for seed in FRESH_SEEDS:
        cases, metadata = fresh_cases(seed, FRESH_PER_SEED)
        fresh_metadata.append(metadata)
        fresh_converter.append(converter_crosscheck(cases))
        dataset_name = f"fresh_seed_{seed}"
        for config_name, optimization, threads in CONFIGS:
            row, signs = evaluate(sessions[config_name], cases, baseline.get(dataset_name))
            row["optimization"] = optimization
            row["threads"] = threads
            runtime[config_name][dataset_name] = row
            if signs is not None:
                baseline[dataset_name] = signs
            print(json.dumps({
                "dataset": dataset_name,
                "config": config_name,
                "right": row["right"],
                "total": row["total"],
                "accuracy": row["accuracy"],
                "errors": row["errors"],
                "nonfinite": row["nonfinite_elements"],
                "shape": row["output_shape_mismatches"],
                "sign_mismatch_cases": row[
                    "prediction_sign_mismatch_cases_vs_disable_threads1"
                ],
            }), flush=True)

    all_runtime_rows = [
        row
        for config in runtime.values()
        for row in config.values()
    ]
    known_rows = [runtime[name]["known"] for name, _optimization, _threads in CONFIGS]
    domain_rows = [runtime[name]["latent_domain"] for name, _optimization, _threads in CONFIGS]
    fresh_rows = [
        runtime[name][f"fresh_seed_{seed}"]
        for name, _optimization, _threads in CONFIGS
        for seed in FRESH_SEEDS
    ]
    all_sign_digest_stable = all(
        len({runtime[name][dataset]["sign_sha256"] for name, _o, _t in CONFIGS}) == 1
        for dataset in [
            "known", "latent_domain", *(f"fresh_seed_{seed}" for seed in FRESH_SEEDS)
        ]
    )
    all_raw_digest_stable = all(
        len({runtime[name][dataset]["raw_sha256"] for name, _o, _t in CONFIGS}) == 1
        for dataset in [
            "known", "latent_domain", *(f"fresh_seed_{seed}" for seed in FRESH_SEEDS)
        ]
    )
    domain_meta["wrong_states_disable_threads1"] = [
        domain_meta["state_order"][index]
        for index in runtime["disable_threads1"]["latent_domain"][
            "wrong_case_indices_first_100"
        ]
    ]
    gates = {
        "immutable_authority_member": sha256_bytes(authority_bytes)
        == "478a310e10fcf0a3e82df943fd6ab43671c47059f8e6eb675bf0004bef576500",
        "candidate_structure": structure["pass"],
        "nonzero_color_permutation_symmetry": symmetry["proved"],
        "no_lookup_or_fixture_correction": no_lookup["proved_no_lookup_or_fixture_correction"],
        "official_authority_cost710_correct": bool(
            authority_official and authority_official.get("correct")
            and authority_official.get("cost") == 710
        ),
        "official_candidate_cost650": bool(
            candidate_official and candidate_official.get("cost") == 650
        ),
        "all_196_latent_states_regenerated": domain_meta["state_count"] == 196,
        "known_all265_evaluated": all(row["total"] == 265 for row in known_rows),
        "fresh_each10000_evaluated": all(row["total"] == FRESH_PER_SEED for row in fresh_rows),
        "known_accuracy_at_least_90_all_configs": all(row["policy90"] for row in known_rows),
        "latent_accuracy_at_least_90_all_configs": all(row["policy90"] for row in domain_rows),
        "fresh_accuracy_at_least_90_each_seed_all_configs": all(row["policy90"] for row in fresh_rows),
        "runtime_errors_nonfinite_shape_zero": all(runtime_fault_free(row) for row in all_runtime_rows),
        "no_shape_cloak": bool(
            structure["strict_data_prop"]
            and structure["all_declared_and_inferred_shapes_static_positive"]
            and structure["output_only_one_conv"]
            and all(row["output_shape_mismatches"] == 0 for row in all_runtime_rows)
        ),
        "prediction_sign_stable_across_configs": (
            all(runtime_sign_stable(row) for row in all_runtime_rows)
            and all_sign_digest_stable
        ),
        "raw_output_bytes_stable_across_configs": all_raw_digest_stable,
        "official_converter_crosscheck_exact": bool(
            known_converter["exact"] and domain_converter["exact"]
            and all(row["exact"] for row in fresh_converter)
        ),
        "margin_no_small_positive": all(
            row["small_positive_elements_0_to_0_25"] == 0 for row in all_runtime_rows
        ),
    }
    accepted = all(gates.values())
    payload = {
        "task": 12,
        "lane": "agent_review_task012_h8w8_policy90_273",
        "decision": "PASS_POLICY90_INDEPENDENT_REVIEW" if accepted else "FAIL_CLOSED",
        "accepted": accepted,
        "policy_threshold": POLICY_THRESHOLD,
        "independence": {
            "lane272_milp_or_evidence_used_as_audit_input": False,
            "candidate_binary_only_from_lane272": True,
            "generator_source": "inputs/arc-gen-repo/tasks/task_0962bcdd.py",
            "known_source": relative(KNOWN_FILE),
        },
        "authority": {
            "zip": relative(AUTHORITY_ZIP),
            "zip_sha256": AUTHORITY_ZIP_SHA256,
            "member": "task012.onnx",
            "member_sha256": sha256_bytes(authority_bytes),
            "matches_handcrafted_byte_for_byte": authority_bytes == AUTHORITY_FILE.read_bytes(),
            "handcrafted_path": relative(AUTHORITY_FILE),
            "official": authority_official,
        },
        "candidate": {
            "path": relative(CANDIDATE),
            "sha256": sha256_bytes(candidate_bytes),
            "file_bytes": len(candidate_bytes),
            "official": candidate_official,
            "cost_reduction_vs_authority": (
                authority_official["cost"] - candidate_official["cost"]
                if authority_official and candidate_official else None
            ),
        },
        "structure": structure,
        "color_permutation_symmetry": symmetry,
        "no_lookup": no_lookup,
        "known": {
            "file_sha256": sha256_path(KNOWN_FILE),
            "split_counts": known_splits,
            "total": int(known[0].shape[0]),
            "case_data_sha256": case_digest(known),
            "official_converter_crosscheck": known_converter,
        },
        "latent_domain": {
            **domain_meta,
            "official_converter_crosscheck": domain_converter,
            "color_permutation_generalization": (
                "The 196 [1,2] representatives cover every cols/gravity latent state. "
                "The proved channels1..9 raw identity extends each result to every ordered "
                "pair of distinct nonzero generator colors."
            ),
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
            "runtime_dataset_config_evaluations": len(all_runtime_rows),
            "runtime_case_executions": sum(row["total"] for row in all_runtime_rows),
            "errors": sum(row["errors"] for row in all_runtime_rows),
            "nonfinite_cases": sum(row["nonfinite_cases"] for row in all_runtime_rows),
            "nonfinite_elements": sum(row["nonfinite_elements"] for row in all_runtime_rows),
            "output_shape_mismatches": sum(row["output_shape_mismatches"] for row in all_runtime_rows),
            "prediction_sign_mismatch_cases": sum(
                row["prediction_sign_mismatch_cases_vs_disable_threads1"]
                for row in all_runtime_rows
            ),
            "prediction_sign_mismatch_cells": sum(
                row["prediction_sign_mismatch_cells_vs_disable_threads1"]
                for row in all_runtime_rows
            ),
            "small_positive_elements_0_to_0_25": sum(
                row["small_positive_elements_0_to_0_25"] for row in all_runtime_rows
            ),
            "minimum_positive": min(
                row["minimum_positive"] for row in all_runtime_rows
                if row["minimum_positive"] is not None
            ),
            "maximum_nonpositive": max(
                row["maximum_nonpositive"] for row in all_runtime_rows
                if row["maximum_nonpositive"] is not None
            ),
            "official_converter_crosscheck_cases": (
                known_converter["cases"] + domain_converter["cases"]
                + sum(row["cases"] for row in fresh_converter)
            ),
            "elapsed_seconds": time.monotonic() - started,
        },
        "policy": {
            "fail_closed": True,
            "root_or_71407_written": False,
            "candidate_promoted": False,
            "kimi_used": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "accepted": accepted,
        "authority_cost": authority_official["cost"] if authority_official else None,
        "candidate_cost": candidate_official["cost"] if candidate_official else None,
        "known_results": sorted({(row["right"], row["total"]) for row in known_rows}),
        "latent_results": sorted({(row["right"], row["total"]) for row in domain_rows}),
        "fresh_results": sorted({(row["right"], row["total"]) for row in fresh_rows}),
        "aggregate": payload["aggregate"],
        "failed_gates": [name for name, value in gates.items() if not value],
        "evidence": relative(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

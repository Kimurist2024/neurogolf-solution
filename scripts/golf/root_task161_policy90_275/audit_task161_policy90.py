#!/usr/bin/env python3
"""Independent fail-closed normal-POLICY90 audit for task161 cost 186."""

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
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task161_r01_static186.onnx"
)
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
AUTHORITY_MEMBER_SHA256 = "5dc274d8515f1ac2a5c58583197984cd60fa2ede69fbe8206992f98940a38fbe"
CANDIDATE_SHA256 = "6752eeea166c8111cda053c3cc36f54b1409d81c7553d672201792f646b31e3a"
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task161.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
PRIVATE_ZERO = ROOT / "docs/golf/private_zero_tasks.md"
OUTPUT = HERE / "evidence.json"
EXPECTED_IO = (1, 10, 30, 30)
POLICY_THRESHOLD = 0.90
FRESH_SEEDS = (275_161_001, 275_261_001)
FRESH_PER_SEED = 10_000
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads1", "default", 1),
    ("default_threads4", "default", 4),
)
BASE_CONFIG = "disable_threads1"
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
LOOKUP_OPS = {
    "TfIdfVectorizer", "Hardmax", "Gather", "GatherElements", "GatherND",
    "Scatter", "ScatterElements", "ScatterND", "OneHot", "CategoryMapper",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from lib import scoring  # noqa: E402

TASK_HASH = json.loads(TASK_MAP.read_text(encoding="utf-8"))["161"]
GENERATOR_MODULE = f"task_{TASK_HASH}"
GEN = importlib.import_module(GENERATOR_MODULE)
COMMON = importlib.import_module("common")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


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


def graph_reachability(model: onnx.ModelProto) -> dict[str, Any]:
    producers = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output if output
    }
    live_nodes: set[int] = set()
    pending = [item.name for item in model.graph.output]
    while pending:
        name = pending.pop()
        index = producers.get(name)
        if index is None or index in live_nodes:
            continue
        live_nodes.add(index)
        pending.extend(item for item in model.graph.node[index].input if item)
    used_inputs = {
        name for index in live_nodes for name in model.graph.node[index].input if name
    }
    initializers = {item.name for item in model.graph.initializer}
    return {
        "live_nodes": len(live_nodes),
        "dead_node_indices": sorted(set(range(len(model.graph.node))) - live_nodes),
        "unused_initializers": sorted(initializers - used_inputs),
        "all_nodes_and_initializers_output_reachable": (
            len(live_nodes) == len(model.graph.node) and initializers <= used_inputs
        ),
    }


def conv_bias_audit(model: onnx.ModelProto) -> dict[str, Any]:
    inits = {item.name: item for item in model.graph.initializer}
    findings = []
    for node in model.graph.node:
        expected = None
        bias_index = None
        if node.op_type == "Conv":
            bias_index = 2
            weight = inits.get(node.input[1]) if len(node.input) > 1 else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        elif node.op_type == "ConvTranspose":
            bias_index = 2
            weight = inits.get(node.input[1]) if len(node.input) > 1 else None
            attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
            group = int(attrs.get("group", 1))
            expected = int(weight.dims[1] * group) if weight is not None and len(weight.dims) > 1 else None
        elif node.op_type == "QLinearConv":
            bias_index = 8
            weight = inits.get(node.input[3]) if len(node.input) > 3 else None
            expected = int(weight.dims[0]) if weight is not None and weight.dims else None
        if bias_index is None or len(node.input) <= bias_index or not node.input[bias_index]:
            continue
        bias = inits.get(node.input[bias_index])
        count = int(math.prod(bias.dims)) if bias is not None else None
        safe = bool(bias is not None and len(bias.dims) == 1 and count == expected)
        findings.append({
            "node": node.name or (node.output[0] if node.output else ""),
            "op": node.op_type,
            "bias": node.input[bias_index],
            "bias_shape": list(bias.dims) if bias is not None else None,
            "expected_channels": expected,
            "safe": safe,
        })
    return {
        "findings": findings,
        "applicable_nodes": len(findings),
        "ub0": all(item["safe"] for item in findings),
        "interpretation": "No Conv-family node; UB0 is vacuous." if not findings else "All Conv-family biases have exact channel length.",
    }


def structural(model: onnx.ModelProto, label: str) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        result["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(full_check=False, full_check_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        result["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        result.update(strict_shape_data_prop=False, strict_shape_error=f"{type(exc).__name__}: {exc}")
        inferred = model

    values = [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    shapes = {
        value.name: tensor_shape(value)
        for value in values if value.type.HasField("tensor_type")
    }
    element_types = {
        value.name: int(value.type.tensor_type.elem_type)
        for value in values if value.type.HasField("tensor_type")
    }
    static_positive = all(
        shape and all(isinstance(dim, int) and dim > 0 for dim in shape)
        for shape in shapes.values()
    )
    nonstandard_domains = sorted({
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
    lookup = sorted({node.op_type for node in model.graph.node if node.op_type in LOOKUP_OPS})
    external = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    initializer_audit = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": (
                int(np.count_nonzero(~np.isfinite(array)))
                if np.issubdtype(array.dtype, np.number) else None
            ),
        }
        for name, array in arrays.items()
    }
    finite = all(item["nonfinite"] in (0, None) for item in initializer_audit.values())
    histogram = Counter(node.op_type for node in model.graph.node)
    equations = [
        helper.get_attribute_value(attr).decode("utf-8")
        for node in model.graph.node if node.op_type == "Einsum"
        for attr in node.attribute if attr.name == "equation"
    ]
    reachability = graph_reachability(model)
    conv_bias = conv_bias_audit(model)
    canonical_input = bool(
        len(model.graph.input) == 1 and model.graph.input[0].name == "input"
        and shapes.get("input") == list(EXPECTED_IO)
        and element_types.get("input") == TensorProto.FLOAT
    )
    canonical_output = bool(
        len(model.graph.output) == 1 and model.graph.output[0].name == "output"
        and shapes.get("output") == list(EXPECTED_IO)
        and element_types.get("output") == TensorProto.FLOAT
    )
    no_lookup = bool(
        not lookup
        and max((item["elements"] for item in initializer_audit.values()), default=0) < 10_000
        and len(model.functions) == 0
        and nested_graph_count(model) == 0
        and reachability["all_nodes_and_initializers_output_reachable"]
    )
    result.update({
        "shapes": shapes,
        "element_types": element_types,
        "all_declared_and_inferred_shapes_static_positive": static_positive,
        "canonical_input": canonical_input,
        "canonical_output": canonical_output,
        "node_count": len(model.graph.node),
        "op_histogram": dict(sorted(histogram.items())),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "standard_domains_only": not nonstandard_domains,
        "nonstandard_domains": nonstandard_domains,
        "banned_ops": banned,
        "lookup_ops": lookup,
        "einsum_equations": equations,
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "giant_einsum_ge15": any(node.op_type == "Einsum" and len(node.input) >= 15 for node in model.graph.node),
        "nested_graphs": nested_graph_count(model),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "external_initializers": external,
        "initializer_audit": initializer_audit,
        "initializer_elements": sum(item["elements"] for item in initializer_audit.values()),
        "largest_initializer_elements": max((item["elements"] for item in initializer_audit.values()), default=0),
        "finite_initializers": finite,
        "reachability": reachability,
        "conv_bias": conv_bias,
        "no_lookup_or_fixture_table": no_lookup,
        "no_lookup_reason": (
            "Complete graph has only standard arithmetic contractions/addition; all 66/70 constants are small, live coefficients, with no index/table/branch operator."
        ),
        "file_bytes": len(model.SerializeToString()),
        "under_file_limit": len(model.SerializeToString()) <= scoring.FILESIZE_LIMIT_IN_BYTES,
    })
    result["pass"] = bool(
        result.get("full_check") and result.get("strict_shape_data_prop")
        and static_positive and canonical_input and canonical_output
        and not nonstandard_domains and not banned and not lookup
        and result["nested_graphs"] == 0 and result["functions"] == 0
        and result["sparse_initializers"] == 0 and not external
        and finite and reachability["all_nodes_and_initializers_output_reachable"]
        and conv_bias["ub0"] and no_lookup and result["under_file_limit"]
    )
    return result


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


def runtime_shape_trace(model: onnx.ModelProto) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    }
    declared = {
        value.name: tensor_shape(value)
        for value in [*model.graph.output, *model.graph.value_info]
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    output_names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in output_names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                output_names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    example = scoring.load_examples(161)["train"][0]
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        raise RuntimeError("known trace example conversion failed")
    outputs = session.run(output_names, {"input": benchmark["input"]})
    shapes = {name: list(np.asarray(value).shape) for name, value in zip(output_names, outputs)}
    mismatches = [
        {"tensor": name, "declared": shape, "actual": shapes.get(name)}
        for name, shape in declared.items()
        if name in shapes and shape != shapes[name]
    ]
    return {
        "traced_tensors": len(shapes),
        "runtime_shapes": shapes,
        "declared_shapes": declared,
        "declared_actual_mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "truthful": len(mismatches) == 0,
    }


def official_measure(model: onnx.ModelProto, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"task161_policy90_{label}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), 161, work, label=label, require_correct=False
        )


def case_id(example: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("input", "output"):
        array = np.asarray(example[key], dtype=np.uint8)
        digest.update(np.asarray(array.shape, dtype=np.int16).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def case_stream_digest_update(digest: Any, identifier: str) -> None:
    digest.update(bytes.fromhex(identifier))


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite_cases": 0,
        "nonfinite_elements": 0,
        "output_shape_mismatches": 0,
        "small_positive_elements_0_to_0_25": 0,
        "minimum_positive": math.inf,
        "maximum_nonpositive": -math.inf,
        "first_error": None,
        "first_wrong": None,
        "first_shape_mismatch": None,
        "raw_sha256_state": hashlib.sha256(),
        "sign_sha256_state": hashlib.sha256(),
        "config_sign_mismatch_cases": 0,
        "config_sign_mismatch_cells": 0,
        "config_raw_mismatch_cases": 0,
    }


def empty_comparison() -> dict[str, Any]:
    return {
        "total": 0,
        "both_correct": 0,
        "authority_correct_candidate_wrong": 0,
        "candidate_correct_authority_wrong": 0,
        "both_wrong": 0,
        "sign_mismatch_cases": 0,
        "sign_mismatch_cells": 0,
        "raw_equal_cases": 0,
        "raw_mismatch_cases": 0,
        "first_candidate_added_regression": None,
        "first_authority_only_failure": None,
    }


def finalize_stats(row: dict[str, Any]) -> dict[str, Any]:
    total = row["total"]
    out = {
        key: value for key, value in row.items()
        if key not in {"raw_sha256_state", "sign_sha256_state"}
    }
    out["accuracy"] = row["right"] / total if total else 0.0
    out["policy90"] = bool(total and out["accuracy"] >= POLICY_THRESHOLD)
    out["minimum_positive"] = None if row["minimum_positive"] == math.inf else row["minimum_positive"]
    out["maximum_nonpositive"] = None if row["maximum_nonpositive"] == -math.inf else row["maximum_nonpositive"]
    out["raw_sha256"] = row["raw_sha256_state"].hexdigest()
    out["sign_sha256"] = row["sign_sha256_state"].hexdigest()
    return out


def finalize_comparison(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    total = row["total"]
    out["candidate_added_regression_rate"] = (
        row["authority_correct_candidate_wrong"] / total if total else 0.0
    )
    out["authority_only_failure_rate"] = (
        row["candidate_correct_authority_wrong"] / total if total else 0.0
    )
    out["sign_equal_cases"] = total - row["sign_mismatch_cases"]
    return out


def update_stats(
    row: dict[str, Any], raw: np.ndarray | None, expected: np.ndarray,
    location: dict[str, Any], error: Exception | None,
    baseline_raw: np.ndarray | None,
) -> None:
    row["total"] += 1
    if error is not None or raw is None:
        row["errors"] += 1
        if row["first_error"] is None:
            row["first_error"] = {**location, "error": f"{type(error).__name__}: {error}"}
        return
    shape = tuple(int(dim) for dim in raw.shape)
    if shape != EXPECTED_IO:
        row["output_shape_mismatches"] += 1
        if row["first_shape_mismatch"] is None:
            row["first_shape_mismatch"] = {**location, "actual": list(shape)}
        return
    finite = np.isfinite(raw)
    nonfinite = int(np.count_nonzero(~finite))
    row["nonfinite_elements"] += nonfinite
    row["nonfinite_cases"] += int(nonfinite > 0)
    positive = raw > 0
    packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
    row["raw_sha256_state"].update(np.ascontiguousarray(raw).tobytes())
    row["sign_sha256_state"].update(packed)
    correct = bool(np.array_equal(positive, expected))
    row["right" if correct else "wrong"] += 1
    if not correct and row["first_wrong"] is None:
        row["first_wrong"] = {
            **location,
            "different_cells": int(np.count_nonzero(positive != expected)),
        }
    if np.any(positive):
        row["minimum_positive"] = min(row["minimum_positive"], float(raw[positive].min()))
        row["small_positive_elements_0_to_0_25"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
    nonpositive = finite & ~positive
    if np.any(nonpositive):
        row["maximum_nonpositive"] = max(row["maximum_nonpositive"], float(raw[nonpositive].max()))
    if baseline_raw is not None:
        base_positive = baseline_raw > 0
        different = int(np.count_nonzero(positive != base_positive))
        row["config_sign_mismatch_cases"] += int(different > 0)
        row["config_sign_mismatch_cells"] += different
        row["config_raw_mismatch_cases"] += int(not np.array_equal(raw, baseline_raw))


def compare_pair(
    row: dict[str, Any], authority_raw: np.ndarray | None, candidate_raw: np.ndarray | None,
    expected: np.ndarray, location: dict[str, Any],
) -> None:
    if authority_raw is None or candidate_raw is None:
        return
    if tuple(authority_raw.shape) != EXPECTED_IO or tuple(candidate_raw.shape) != EXPECTED_IO:
        return
    row["total"] += 1
    authority_sign = authority_raw > 0
    candidate_sign = candidate_raw > 0
    authority_correct = bool(np.array_equal(authority_sign, expected))
    candidate_correct = bool(np.array_equal(candidate_sign, expected))
    if authority_correct and candidate_correct:
        row["both_correct"] += 1
    elif authority_correct:
        row["authority_correct_candidate_wrong"] += 1
        if row["first_candidate_added_regression"] is None:
            row["first_candidate_added_regression"] = {
                **location,
                "candidate_different_cells_vs_gold": int(np.count_nonzero(candidate_sign != expected)),
                "candidate_different_cells_vs_authority": int(np.count_nonzero(candidate_sign != authority_sign)),
            }
    elif candidate_correct:
        row["candidate_correct_authority_wrong"] += 1
        if row["first_authority_only_failure"] is None:
            row["first_authority_only_failure"] = {
                **location,
                "authority_different_cells_vs_gold": int(np.count_nonzero(authority_sign != expected)),
                "candidate_different_cells_vs_authority": int(np.count_nonzero(candidate_sign != authority_sign)),
            }
    else:
        row["both_wrong"] += 1
    different = int(np.count_nonzero(authority_sign != candidate_sign))
    row["sign_mismatch_cases"] += int(different > 0)
    row["sign_mismatch_cells"] += different
    raw_equal = bool(np.array_equal(authority_raw, candidate_raw))
    row["raw_equal_cases" if raw_equal else "raw_mismatch_cases"] += 1


def evaluate_dataset(
    label: str,
    cases: Any,
    sessions: dict[tuple[str, str], ort.InferenceSession],
    count: int,
    progress_every: int = 0,
) -> dict[str, Any]:
    started = time.monotonic()
    stats = {
        model_name: {config_name: empty_stats() for config_name, _opt, _threads in CONFIGS}
        for model_name in ("authority", "candidate")
    }
    comparisons = {config_name: empty_comparison() for config_name, _opt, _threads in CONFIGS}
    case_digest = hashlib.sha256()
    seen_ids: set[str] = set()
    for ordinal, (location, example) in enumerate(cases, start=1):
        identifier = case_id(example)
        case_stream_digest_update(case_digest, identifier)
        seen_ids.add(identifier)
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"official converter rejected {label} case {ordinal}")
        expected = benchmark["output"] > 0
        raws: dict[tuple[str, str], np.ndarray | None] = {}
        errors: dict[tuple[str, str], Exception | None] = {}
        for model_name in ("authority", "candidate"):
            for config_name, _optimization, _threads in CONFIGS:
                key = (model_name, config_name)
                try:
                    raws[key] = np.asarray(sessions[key].run(["output"], {"input": benchmark["input"]})[0])
                    errors[key] = None
                except Exception as exc:  # noqa: BLE001
                    raws[key] = None
                    errors[key] = exc
        for model_name in ("authority", "candidate"):
            baseline_raw = raws[(model_name, BASE_CONFIG)]
            for config_name, _optimization, _threads in CONFIGS:
                update_stats(
                    stats[model_name][config_name], raws[(model_name, config_name)],
                    expected, location, errors[(model_name, config_name)],
                    None if config_name == BASE_CONFIG else baseline_raw,
                )
        for config_name, _optimization, _threads in CONFIGS:
            compare_pair(
                comparisons[config_name], raws[("authority", config_name)],
                raws[("candidate", config_name)], expected, location,
            )
        if progress_every and ordinal % progress_every == 0:
            print(json.dumps({"dataset": label, "processed": ordinal}), flush=True)
    if ordinal != count:
        raise RuntimeError(f"{label}: expected {count} cases, evaluated {ordinal}")
    return {
        "label": label,
        "total": count,
        "unique_case_ids": len(seen_ids),
        "case_stream_sha256": case_digest.hexdigest(),
        "models": {
            model_name: {
                config_name: finalize_stats(row)
                for config_name, row in model_rows.items()
            }
            for model_name, model_rows in stats.items()
        },
        "candidate_vs_authority": {
            config_name: finalize_comparison(row)
            for config_name, row in comparisons.items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "case_ids": seen_ids,
    }


def known_cases() -> tuple[Any, int, dict[str, int], set[str]]:
    examples = scoring.load_examples(161)
    split_counts = {name: len(examples[name]) for name in ("train", "test", "arc-gen")}
    rows = [
        ({"subset": subset, "index": index, "case_id": case_id(example)}, example)
        for subset in ("train", "test", "arc-gen")
        for index, example in enumerate(examples[subset])
    ]
    return iter(rows), len(rows), split_counts, {item[0]["case_id"] for item in rows}


def fresh_cases(seed: int, known_ids: set[str], previously_accepted: set[str]) -> tuple[Any, dict[str, int]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    COMMON.random.seed(seed)
    counters = {"attempts": 0, "known_collision_skips": 0, "cross_stream_or_duplicate_skips": 0}

    def iterator() -> Any:
        accepted = 0
        while accepted < FRESH_PER_SEED:
            counters["attempts"] += 1
            example = GEN.generate()
            identifier = case_id(example)
            if identifier in known_ids:
                counters["known_collision_skips"] += 1
                continue
            if identifier in previously_accepted:
                counters["cross_stream_or_duplicate_skips"] += 1
                continue
            previously_accepted.add(identifier)
            location = {
                "seed": seed,
                "accepted_index": accepted,
                "generator_attempt": counters["attempts"] - 1,
                "case_id": identifier,
            }
            accepted += 1
            yield location, example

    return iterator(), counters


def runtime_fault_free(row: dict[str, Any]) -> bool:
    return bool(
        row["errors"] == 0 and row["nonfinite_cases"] == 0
        and row["nonfinite_elements"] == 0 and row["output_shape_mismatches"] == 0
    )


def config_stable(row: dict[str, Any]) -> bool:
    return bool(
        row["config_sign_mismatch_cases"] == 0
        and row["config_sign_mismatch_cells"] == 0
        and row["config_raw_mismatch_cases"] == 0
    )


def public_private_zero_catalog_contains_task161() -> bool:
    text = PRIVATE_ZERO.read_text(encoding="utf-8")
    tokens = {int(token) for token in __import__("re").findall(r"(?<!\d)(\d{1,3})(?!\d)", text)}
    return 161 in tokens


def compact_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dataset.items() if key != "case_ids"}


def main() -> None:
    started = time.monotonic()
    if sha256_path(AUTHORITY_ZIP) != AUTHORITY_ZIP_SHA256:
        raise RuntimeError("immutable authority ZIP changed")
    if sha256_path(CANDIDATE) != CANDIDATE_SHA256:
        raise RuntimeError("candidate source changed")
    if TASK_HASH != "6cdd2623":
        raise RuntimeError(f"unexpected task161 generator hash: {TASK_HASH}")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task161.onnx")
    if sha256_bytes(authority_bytes) != AUTHORITY_MEMBER_SHA256:
        raise RuntimeError("immutable authority task161 member changed")
    candidate_bytes = CANDIDATE.read_bytes()
    authority_model = onnx.load_model_from_string(authority_bytes)
    candidate_model = onnx.load_model_from_string(candidate_bytes)

    structures = {
        "authority": structural(authority_model, "authority"),
        "candidate": structural(candidate_model, "candidate"),
    }
    shape_traces = {
        "authority": runtime_shape_trace(authority_model),
        "candidate": runtime_shape_trace(candidate_model),
    }
    official = {
        "authority": official_measure(authority_model, "authority"),
        "candidate": official_measure(candidate_model, "candidate"),
    }
    sessions = {
        (model_name, config_name): make_session(model, optimization, threads)
        for model_name, model in (("authority", authority_model), ("candidate", candidate_model))
        for config_name, optimization, threads in CONFIGS
    }

    known_iter, known_count, known_splits, known_ids = known_cases()
    known = evaluate_dataset("known", known_iter, sessions, known_count)
    print(json.dumps({
        "dataset": "known",
        "authority": {name: row["right"] for name, row in known["models"]["authority"].items()},
        "candidate": {name: row["right"] for name, row in known["models"]["candidate"].items()},
    }), flush=True)

    accepted_fresh_ids: set[str] = set()
    fresh_runs = []
    fresh_generation = []
    for seed in FRESH_SEEDS:
        iterator, counters = fresh_cases(seed, known_ids, accepted_fresh_ids)
        dataset = evaluate_dataset(
            f"fresh_seed_{seed}", iterator, sessions, FRESH_PER_SEED, progress_every=2_000
        )
        counters["accepted"] = FRESH_PER_SEED
        fresh_generation.append({"seed": seed, **counters})
        fresh_runs.append(dataset)
        print(json.dumps({
            "dataset": dataset["label"],
            "authority": {name: row["right"] for name, row in dataset["models"]["authority"].items()},
            "candidate": {name: row["right"] for name, row in dataset["models"]["candidate"].items()},
        }), flush=True)

    all_datasets = [known, *fresh_runs]
    all_rows = [
        dataset["models"][model_name][config_name]
        for dataset in all_datasets
        for model_name in ("authority", "candidate")
        for config_name, _optimization, _threads in CONFIGS
    ]
    candidate_known_rows = [known["models"]["candidate"][name] for name, _o, _t in CONFIGS]
    candidate_fresh_rows = [
        dataset["models"]["candidate"][name]
        for dataset in fresh_runs for name, _o, _t in CONFIGS
    ]
    candidate_rows = [*candidate_known_rows, *candidate_fresh_rows]
    authority_rows = [
        dataset["models"]["authority"][name]
        for dataset in all_datasets for name, _o, _t in CONFIGS
    ]
    private_catalog = public_private_zero_catalog_contains_task161()
    gates = {
        "immutable_authority_zip_and_member": True,
        "exact_candidate_sha": True,
        "exact_generator_module_from_task_map": TASK_HASH == "6cdd2623",
        "authority_structure_clean": structures["authority"]["pass"],
        "candidate_structure_clean": structures["candidate"]["pass"],
        "candidate_standard_ops_no_lookup_no_giant_ge15": bool(
            structures["candidate"]["standard_domains_only"]
            and structures["candidate"]["no_lookup_or_fixture_table"]
            and not structures["candidate"]["giant_einsum_ge15"]
        ),
        "candidate_finite_initializers": structures["candidate"]["finite_initializers"],
        "candidate_conv_family_ub0": structures["candidate"]["conv_bias"]["ub0"],
        "truthful_runtime_shapes_no_cloak_both": bool(
            shape_traces["authority"]["truthful"] and shape_traces["candidate"]["truthful"]
        ),
        "official_authority_cost190_correct": bool(
            official["authority"] and official["authority"].get("cost") == 190
            and official["authority"].get("correct") is True
        ),
        "official_candidate_cost186": bool(
            official["candidate"] and official["candidate"].get("cost") == 186
        ),
        "strict_cost_reduction_4": bool(
            official["authority"] and official["candidate"]
            and official["authority"]["cost"] - official["candidate"]["cost"] == 4
        ),
        "known_all266_each_config": all(row["total"] == 266 for row in candidate_known_rows),
        "known_policy90_each_config": all(row["policy90"] for row in candidate_known_rows),
        "fresh_disjoint_2x10000": bool(
            len(fresh_runs) == 2 and len(accepted_fresh_ids) == 2 * FRESH_PER_SEED
            and all(row["total"] == FRESH_PER_SEED for row in fresh_runs)
        ),
        "fresh_policy90_each_seed_each_config": all(row["policy90"] for row in candidate_fresh_rows),
        "runtime_errors_nonfinite_shape_zero_both": all(runtime_fault_free(row) for row in all_rows),
        "candidate_sign_and_raw_config_stable": all(config_stable(row) for row in candidate_rows),
        "authority_sign_and_raw_config_stable": all(config_stable(row) for row in authority_rows),
        "candidate_margin_no_0_to_0_25": all(
            row["small_positive_elements_0_to_0_25"] == 0 for row in candidate_rows
        ),
        "not_public_private_zero_catalog": not private_catalog,
        "not_private_zero_behavior": all(row["right"] > 0 for row in candidate_fresh_rows),
    }
    accepted = all(gates.values())
    candidate_known_regressions = {
        name: known["candidate_vs_authority"][name]["authority_correct_candidate_wrong"]
        for name, _o, _t in CONFIGS
    }
    authority_fresh_wrong = {
        dataset["label"]: {
            name: dataset["models"]["authority"][name]["wrong"]
            for name, _o, _t in CONFIGS
        }
        for dataset in fresh_runs
    }
    candidate_fresh_wrong = {
        dataset["label"]: {
            name: dataset["models"]["candidate"][name]["wrong"]
            for name, _o, _t in CONFIGS
        }
        for dataset in fresh_runs
    }
    payload = {
        "task": 161,
        "lane": "root_task161_policy90_275",
        "decision": "PASS_NORMAL_POLICY90_RECOMMEND_PROMOTION" if accepted else "FAIL_CLOSED",
        "accepted": accepted,
        "policy_threshold": POLICY_THRESHOLD,
        "independence": {
            "prior_265_of_266_or_fresh_evidence_used_as_audit_input": False,
            "source_candidate_binary_only": True,
            "known_source": relative(KNOWN_FILE),
            "generator_map_source": relative(TASK_MAP),
            "generator_module": f"inputs/arc-gen-repo/tasks/{GENERATOR_MODULE}.py",
            "authority_source": relative(AUTHORITY_ZIP),
        },
        "authority": {
            "zip": relative(AUTHORITY_ZIP),
            "zip_sha256": AUTHORITY_ZIP_SHA256,
            "member": "task161.onnx",
            "member_sha256": sha256_bytes(authority_bytes),
            "file_bytes": len(authority_bytes),
            "official": official["authority"],
        },
        "candidate": {
            "source": relative(CANDIDATE),
            "sha256": sha256_bytes(candidate_bytes),
            "file_bytes": len(candidate_bytes),
            "official": official["candidate"],
            "cost_reduction": official["authority"]["cost"] - official["candidate"]["cost"],
            "score_gain": official["candidate"]["score"] - official["authority"]["score"],
        },
        "structure": structures,
        "runtime_shape_trace": shape_traces,
        "known": {
            **compact_dataset(known),
            "source_sha256": sha256_path(KNOWN_FILE),
            "split_counts": known_splits,
        },
        "fresh": {
            "seeds": list(FRESH_SEEDS),
            "count_per_seed": FRESH_PER_SEED,
            "generation": fresh_generation,
            "all_accepted_case_ids_unique_and_disjoint_from_known": bool(
                len(accepted_fresh_ids) == 2 * FRESH_PER_SEED
                and not accepted_fresh_ids.intersection(known_ids)
            ),
            "runs": [compact_dataset(dataset) for dataset in fresh_runs],
        },
        "regression_attribution": {
            "candidate_added_known_regressions_each_config": candidate_known_regressions,
            "authority_known_wrong_each_config": {
                name: known["models"]["authority"][name]["wrong"]
                for name, _o, _t in CONFIGS
            },
            "authority_fresh_wrong_each_config": authority_fresh_wrong,
            "candidate_fresh_wrong_each_config": candidate_fresh_wrong,
            "interpretation": (
                "Known failures where authority is correct and candidate is wrong are candidate-added. "
                "Fresh authority failures are authority heuristic weakness; comparison tables separately "
                "record whether candidate repairs, shares, or adds each failure."
            ),
        },
        "private_zero": {
            "task161_in_public_catalog": private_catalog,
            "candidate_has_nonzero_fresh_success": all(row["right"] > 0 for row in candidate_fresh_rows),
            "classification": "normal POLICY90 candidate; no private-zero exception claimed",
        },
        "gates": gates,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"],
        "accepted": accepted,
        "candidate_known_regressions": candidate_known_regressions,
        "authority_fresh_wrong": authority_fresh_wrong,
        "candidate_fresh_wrong": candidate_fresh_wrong,
        "elapsed_seconds": payload["elapsed_seconds"],
        "evidence": relative(OUTPUT),
    }, indent=2), flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

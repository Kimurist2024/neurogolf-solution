#!/usr/bin/env python3
"""Independent fail-closed POLICY90 audit of the pinned task355 cost-249 model."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import re
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
from onnx import AttributeProto, TensorProto, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CANDIDATE = ROOT / (
    "scripts/golf/loop_7999_13/lane_archive_all400/"
    "task355_r04_static249.onnx"
)
CANDIDATE_SHA256 = "7ca617858a19310a433010e6e50da46b4d562d76f3d0688665c8387bdf6f24d8"
AUTHORITY_ZIP = ROOT / "submission_base_8009.46.zip"
AUTHORITY_ZIP_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
KNOWN_FILE = ROOT / "inputs/neurogolf-2026/task355.json"
GENERATOR_FILE = ROOT / "inputs/arc-gen-repo/tasks/task_de1cd16c.py"
TASKS_DIR = GENERATOR_FILE.parent
PRIVATE_ZERO_CATALOG = ROOT / "docs/golf/private_zero_tasks.md"
PUBLIC_OVERFIT_SOURCE = ROOT / "docs/research/discussions/raw/704762.json"
OUTPUT = HERE / "evidence.json"

EXPECTED_IO = (1, 10, 30, 30)
FRESH_SEEDS = (284_355_001, 284_455_001)
FRESH_PER_SEED = 10_000
POLICY_THRESHOLD = 0.90
GIANT_INITIALIZER_MIN_ELEMENTS = 10_000
GIANT_INTERMEDIATE_MIN_ELEMENTS = 10_000
GIANT_NODE_MIN_INPUTS = 15
CONFIGS = (
    ("disable_threads1", "disabled", 1),
    ("default_threads1", "default", 1),
    ("disable_threads4", "disabled", 4),
    ("default_threads4", "default", 4),
)
BANNED_OPS = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}
LOOKUP_OPS = {
    "TfIdfVectorizer", "Hardmax", "Gather", "GatherElements", "GatherND",
    "Scatter", "ScatterElements", "ScatterND", "OneHot", "CategoryMapper",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GENERATOR = importlib.import_module("task_de1cd16c")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def dimensions(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def element_count(shape: list[int | None] | None) -> int | None:
    if shape is None or any(dim is None for dim in shape):
        return None
    return math.prod(shape) if shape else 1


def tensor_dtype(value: onnx.ValueInfoProto) -> str:
    return TensorProto.DataType.Name(value.type.tensor_type.elem_type)


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


def ancestor_initializers(
    value: str,
    producers: dict[str, onnx.NodeProto],
    initializer_names: set[str],
    seen: set[str] | None = None,
) -> set[str]:
    if value in initializer_names:
        return {value}
    if value not in producers:
        return set()
    visited = set() if seen is None else seen
    if value in visited:
        return set()
    visited.add(value)
    result: set[str] = set()
    for source in producers[value].input:
        if source:
            result.update(ancestor_initializers(source, producers, initializer_names, visited))
    return result


def structural_audit(model: onnx.ModelProto) -> dict[str, Any]:
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
        row["strict_shape_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        inferred = model
        row.update(
            strict_shape_data_prop=False,
            strict_shape_error=f"{type(exc).__name__}: {exc}",
        )

    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    node_output_names = [
        name for node in inferred.graph.node for name in node.output if name
    ]
    missing_typed = [name for name in node_output_names if name not in typed]
    nonstatic = [
        name
        for name, value in typed.items()
        if any(dim is None or dim <= 0 for dim in dimensions(value))
    ]
    output_names = {value.name for value in inferred.graph.output}
    intermediate_rows = []
    for name in node_output_names:
        if name in typed and name not in output_names:
            shape = dimensions(typed[name])
            intermediate_rows.append(
                {
                    "name": name,
                    "dtype": tensor_dtype(typed[name]),
                    "shape": shape,
                    "elements": element_count(shape),
                }
            )
    giant_intermediates = [
        item
        for item in intermediate_rows
        if item["elements"] is None
        or item["elements"] >= GIANT_INTERMEDIATE_MIN_ELEMENTS
    ]

    initializer_arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    initializer_rows = {
        name: {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "elements": int(array.size),
            "nonfinite": (
                int(np.count_nonzero(~np.isfinite(array)))
                if array.dtype.kind in "fc"
                else 0
            ),
        }
        for name, array in initializer_arrays.items()
    }
    giant_initializers = [
        {"name": name, **item}
        for name, item in initializer_rows.items()
        if item["elements"] >= GIANT_INITIALIZER_MIN_ELEMENTS
    ]

    producers = {
        name: node for node in inferred.graph.node for name in node.output if name
    }
    initializer_names = set(initializer_rows)
    lookup_rows = []
    lookup_findings = []
    for index, node in enumerate(inferred.graph.node):
        if node.op_type not in LOOKUP_OPS:
            continue
        data_name = node.input[0] if node.input else ""
        data_shape = (
            dimensions(typed[data_name])
            if data_name in typed
            else list(initializer_arrays[data_name].shape)
            if data_name in initializer_arrays
            else None
        )
        data_elements = element_count(data_shape)
        ancestors = sorted(
            ancestor_initializers(data_name, producers, initializer_names)
        )
        giant_ancestors = [
            name
            for name in ancestors
            if initializer_rows[name]["elements"] >= GIANT_INITIALIZER_MIN_ELEMENTS
        ]
        output_shapes = [
            dimensions(typed[name]) if name in typed else None for name in node.output
        ]
        suspicious = bool(
            data_elements is None
            or data_elements >= GIANT_INTERMEDIATE_MIN_ELEMENTS
            or giant_ancestors
            or any(shape is None for shape in output_shapes)
        )
        lookup_row = {
            "node_index": index,
            "op_type": node.op_type,
            "data_input": data_name,
            "data_origin": (
                "initializer"
                if data_name in initializer_names
                else producers[data_name].op_type
                if data_name in producers
                else "graph_input_or_unknown"
            ),
            "data_shape": data_shape,
            "data_elements": data_elements,
            "initializer_ancestors": ancestors,
            "giant_initializer_ancestors": giant_ancestors,
            "output_shapes": output_shapes,
            "suspicious_table_or_cloak": suspicious,
        }
        lookup_rows.append(lookup_row)
        if suspicious:
            lookup_findings.append(lookup_row)

    ops = Counter(node.op_type for node in model.graph.node)
    domains = sorted(
        {
            domain
            for domain in [
                *(item.domain for item in model.opset_import),
                *(node.domain for node in model.graph.node),
            ]
            if domain not in ("", "ai.onnx")
        }
    )
    banned = sorted(
        {
            node.op_type
            for node in model.graph.node
            if node.op_type in BANNED_OPS or "Sequence" in node.op_type
        }
    )
    external = [
        item.name
        for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    maximum_node_inputs = max((len(node.input) for node in model.graph.node), default=0)
    input_shape = (
        dimensions(inferred.graph.input[0]) if len(inferred.graph.input) == 1 else None
    )
    output_shape = (
        dimensions(inferred.graph.output[0]) if len(inferred.graph.output) == 1 else None
    )
    conv_findings = check_conv_bias(model)
    row.update(
        {
            "node_count": len(model.graph.node),
            "node_output_count": len(node_output_names),
            "op_histogram": dict(sorted(ops.items())),
            "opsets": [
                {"domain": item.domain, "version": item.version}
                for item in model.opset_import
            ],
            "input_shape": input_shape,
            "input_dtype": (
                tensor_dtype(inferred.graph.input[0])
                if len(inferred.graph.input) == 1
                else None
            ),
            "output_shape": output_shape,
            "output_dtype": (
                tensor_dtype(inferred.graph.output[0])
                if len(inferred.graph.output) == 1
                else None
            ),
            "canonical_float_input": bool(
                input_shape == list(EXPECTED_IO)
                and len(inferred.graph.input) == 1
                and inferred.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
            ),
            "canonical_output_shape": output_shape == list(EXPECTED_IO),
            "missing_typed_node_outputs": missing_typed,
            "nonstatic_typed_values": nonstatic,
            "all_shapes_static": not missing_typed and not nonstatic,
            "intermediates": intermediate_rows,
            "maximum_declared_intermediate_elements": max(
                (item["elements"] for item in intermediate_rows if item["elements"] is not None),
                default=0,
            ),
            "giant_intermediate_min_elements": GIANT_INTERMEDIATE_MIN_ELEMENTS,
            "giant_intermediates": giant_intermediates,
            "initializer_audit": initializer_rows,
            "initializer_elements": sum(
                item["elements"] for item in initializer_rows.values()
            ),
            "finite_initializers": all(
                item["nonfinite"] == 0 for item in initializer_rows.values()
            ),
            "giant_initializer_min_elements": GIANT_INITIALIZER_MIN_ELEMENTS,
            "giant_initializers": giant_initializers,
            "lookup_ops": lookup_rows,
            "lookup_table_or_cloak_findings": lookup_findings,
            "lookup_audit_clear": not lookup_findings,
            "maximum_node_inputs": maximum_node_inputs,
            "giant_node_min_inputs": GIANT_NODE_MIN_INPUTS,
            "giant_node_arity": maximum_node_inputs >= GIANT_NODE_MIN_INPUTS,
            "nonstandard_domains": domains,
            "standard_domains_only": not domains,
            "banned_ops": banned,
            "nested_graphs": nested_graph_count(model),
            "functions": len(model.functions),
            "sparse_initializers": len(model.graph.sparse_initializer),
            "external_initializers": external,
            "conv_bias_findings": conv_findings,
            "conv_bias_ub0": not conv_findings,
            "file_bytes": len(model.SerializeToString()),
            "under_file_limit": (
                len(model.SerializeToString()) <= scoring.FILESIZE_LIMIT_IN_BYTES
            ),
        }
    )
    row["pass_without_runtime_truth"] = bool(
        row.get("full_check")
        and row.get("strict_shape_data_prop")
        and row["canonical_float_input"]
        and row["canonical_output_shape"]
        and row["all_shapes_static"]
        and row["finite_initializers"]
        and not giant_initializers
        and not giant_intermediates
        and not row["giant_node_arity"]
        and row["lookup_audit_clear"]
        and row["standard_domains_only"]
        and not banned
        and row["nested_graphs"] == 0
        and row["functions"] == 0
        and row["sparse_initializers"] == 0
        and not external
        and row["conv_bias_ub0"]
        and row["under_file_limit"]
    )
    return row


def sanitize(model: onnx.ModelProto) -> onnx.ModelProto:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected candidate")
    return sanitized


def profile_bytes(data: bytes, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"review284_{label}_", dir="/tmp") as work:
        path = Path(work) / "task355.onnx"
        path.write_bytes(data)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def official_profile(
    model: onnx.ModelProto, label: str, require_correct: bool
) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(
        prefix=f"review284_official_{label}_", dir="/tmp"
    ) as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), 355, work, label=label, require_correct=require_correct
        )


def make_session(
    sanitized: onnx.ModelProto, optimization: str, threads: int
) -> ort.InferenceSession:
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


def direct_onehot(grid: list[list[int]]) -> np.ndarray:
    values = np.asarray(grid, dtype=np.int64)
    result = np.zeros(EXPECTED_IO, dtype=np.float32)
    rows, columns = np.indices(values.shape)
    result[0, values, rows, columns] = 1.0
    return result


def cases_digest(examples: list[dict[str, Any]]) -> str:
    current = hashlib.sha256()
    for example in examples:
        for key in ("input", "output"):
            array = np.asarray(example[key], dtype=np.uint8)
            current.update(key.encode("ascii"))
            current.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            current.update(array.tobytes())
    return current.hexdigest()


def converter_crosscheck(examples: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = 0
    first = None
    for index, example in enumerate(examples):
        converted = scoring.convert_to_numpy(example)
        exact = bool(
            converted is not None
            and np.array_equal(converted["input"], direct_onehot(example["input"]))
            and np.array_equal(converted["output"], direct_onehot(example["output"]))
        )
        if not exact:
            mismatches += 1
            first = index if first is None else first
    return {
        "cases": len(examples),
        "mismatches": mismatches,
        "first": first,
        "exact": mismatches == 0,
    }


def fresh_examples(seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return [GENERATOR.generate() for _ in range(FRESH_PER_SEED)]


def empty_runtime() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "errors": 0,
        "nonfinite_cases": 0,
        "nonfinite_elements": 0,
        "shape_mismatches": 0,
        "small_positive_elements_0_to_0_25": 0,
        "minimum_positive": math.inf,
        "maximum_nonpositive": -math.inf,
        "sign_config_mismatch_cases": 0,
        "sign_config_mismatch_cells": 0,
        "raw_config_mismatch_cases": 0,
        "sign_sha256": hashlib.sha256(),
        "raw_sha256": hashlib.sha256(),
        "observed_shapes": set(),
        "first_failure": None,
    }


def evaluate(
    sessions: dict[str, ort.InferenceSession], examples: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    rows = {name: empty_runtime() for name, _optimization, _threads in CONFIGS}
    started = time.monotonic()
    for index, example in enumerate(examples):
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            raise RuntimeError(f"case {index} failed official conversion")
        expected = converted["output"] > 0
        baseline_sign: np.ndarray | None = None
        baseline_raw: bytes | None = None
        for config_index, (name, _optimization, _threads) in enumerate(CONFIGS):
            row = rows[name]
            try:
                raw = np.asarray(
                    sessions[name].run(["output"], {"input": converted["input"]})[0]
                )
            except Exception as exc:  # noqa: BLE001
                row["errors"] += 1
                row["wrong"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "runtime_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                continue
            actual_shape = tuple(int(dim) for dim in raw.shape)
            row["observed_shapes"].add(actual_shape)
            if actual_shape != EXPECTED_IO:
                row["shape_mismatches"] += 1
                row["wrong"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "shape_mismatch",
                        "actual": list(actual_shape),
                    }
                continue
            finite = np.isfinite(raw)
            current_nonfinite = int(np.count_nonzero(~finite))
            row["nonfinite_cases"] += int(current_nonfinite > 0)
            row["nonfinite_elements"] += current_nonfinite
            positive = raw > 0
            raw_bytes = np.ascontiguousarray(raw).tobytes()
            packed_sign = np.packbits(
                positive.reshape(-1), bitorder="little"
            ).tobytes()
            row["raw_sha256"].update(raw_bytes)
            row["sign_sha256"].update(packed_sign)
            if config_index == 0:
                baseline_sign = positive.copy()
                baseline_raw = raw_bytes
            else:
                if baseline_sign is None or baseline_raw is None:
                    row["sign_config_mismatch_cases"] += 1
                    row["sign_config_mismatch_cells"] += math.prod(EXPECTED_IO)
                    row["raw_config_mismatch_cases"] += 1
                else:
                    differing = int(np.count_nonzero(positive != baseline_sign))
                    row["sign_config_mismatch_cases"] += int(differing > 0)
                    row["sign_config_mismatch_cells"] += differing
                    row["raw_config_mismatch_cases"] += int(raw_bytes != baseline_raw)
            positive_values = raw[finite & positive]
            nonpositive_values = raw[finite & ~positive]
            if positive_values.size:
                row["minimum_positive"] = min(
                    row["minimum_positive"], float(np.min(positive_values))
                )
                row["small_positive_elements_0_to_0_25"] += int(
                    np.count_nonzero(positive_values < 0.25)
                )
            if nonpositive_values.size:
                row["maximum_nonpositive"] = max(
                    row["maximum_nonpositive"], float(np.max(nonpositive_values))
                )
            if current_nonfinite == 0 and np.array_equal(positive, expected):
                row["right"] += 1
            else:
                row["wrong"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "case": index,
                        "kind": "wrong_or_nonfinite",
                        "different_cells": int(np.count_nonzero(positive != expected)),
                    }
    elapsed = time.monotonic() - started
    result: dict[str, dict[str, Any]] = {}
    for name, optimization, threads in CONFIGS:
        row = rows[name]
        result[name] = {
            "optimization": optimization,
            "threads": threads,
            "total": len(examples),
            "right": row["right"],
            "wrong": row["wrong"],
            "accounted": row["right"] + row["wrong"],
            "accuracy": row["right"] / len(examples),
            "policy90": row["right"] / len(examples) >= POLICY_THRESHOLD,
            "errors": row["errors"],
            "nonfinite_cases": row["nonfinite_cases"],
            "nonfinite_elements": row["nonfinite_elements"],
            "shape_mismatches": row["shape_mismatches"],
            "observed_shapes": [
                list(value) for value in sorted(row["observed_shapes"])
            ],
            "small_positive_elements_0_to_0_25": row[
                "small_positive_elements_0_to_0_25"
            ],
            "minimum_positive": (
                None
                if row["minimum_positive"] == math.inf
                else row["minimum_positive"]
            ),
            "maximum_nonpositive": (
                None
                if row["maximum_nonpositive"] == -math.inf
                else row["maximum_nonpositive"]
            ),
            "sign_config_mismatch_cases": row["sign_config_mismatch_cases"],
            "sign_config_mismatch_cells": row["sign_config_mismatch_cells"],
            "raw_config_mismatch_cases": row["raw_config_mismatch_cases"],
            "sign_sha256": row["sign_sha256"].hexdigest(),
            "raw_sha256": row["raw_sha256"].hexdigest(),
            "first_failure": row["first_failure"],
            "elapsed_seconds": elapsed,
        }
    return result


def truthful_shape_trace(
    sanitized: onnx.ModelProto, converted_input: np.ndarray
) -> dict[str, Any]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(sanitized), strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in [*inferred.graph.value_info, *inferred.graph.output]
        if value.type.HasField("tensor_type")
    }
    names = [
        name
        for node in inferred.graph.node
        for name in node.output
        if name and name in typed
    ]
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
            runtime = ort.InferenceSession(
                exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            arrays = runtime.run(names, {"input": converted_input})
            mismatches = [
                {
                    "name": name,
                    "declared": dimensions(typed[name]),
                    "actual": list(array.shape),
                }
                for name, array in zip(names, arrays)
                if dimensions(typed[name]) != list(array.shape)
            ]
            nonfinite = sum(
                int(np.count_nonzero(~np.isfinite(array)))
                for array in arrays
                if np.asarray(array).dtype.kind in "fc"
            )
            element_rows = [
                {
                    "name": name,
                    "shape": list(array.shape),
                    "elements": int(np.asarray(array).size),
                }
                for name, array in zip(names, arrays)
            ]
            maximum_elements = max(
                (item["elements"] for item in element_rows), default=0
            )
            giant_runtime = [
                item
                for item in element_rows
                if item["name"] != "output"
                and item["elements"] >= GIANT_INTERMEDIATE_MIN_ELEMENTS
            ]
            result[label] = {
                "session_created": True,
                "traced_outputs": len(names),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "nonfinite_elements": nonfinite,
                "maximum_node_output_elements_including_final_output": maximum_elements,
                "giant_runtime_intermediates": giant_runtime,
                "truthful": not mismatches and nonfinite == 0 and not giant_runtime,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            result[label] = {
                "session_created": False,
                "traced_outputs": 0,
                "mismatch_count": None,
                "mismatches": [],
                "nonfinite_elements": None,
                "maximum_node_output_elements_including_final_output": None,
                "giant_runtime_intermediates": [],
                "truthful": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return result


def runtime_row_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["accounted"] == row["total"]
        and row["policy90"]
        and row["errors"] == 0
        and row["nonfinite_elements"] == 0
        and row["shape_mismatches"] == 0
        and row["small_positive_elements_0_to_0_25"] == 0
        and row["sign_config_mismatch_cases"] == 0
        and row["sign_config_mismatch_cells"] == 0
        and row["raw_config_mismatch_cases"] == 0
        and row["observed_shapes"] == [list(EXPECTED_IO)]
    )


def main() -> None:
    started = time.monotonic()
    candidate_bytes = CANDIDATE.read_bytes()
    if digest(candidate_bytes) != CANDIDATE_SHA256:
        raise RuntimeError("candidate SHA changed")
    authority_zip_bytes = AUTHORITY_ZIP.read_bytes()
    if digest(authority_zip_bytes) != AUTHORITY_ZIP_SHA256:
        raise RuntimeError("authority ZIP SHA changed")
    with zipfile.ZipFile(AUTHORITY_ZIP) as archive:
        authority_bytes = archive.read("task355.onnx")

    candidate_model = onnx.load_model_from_string(candidate_bytes)
    authority_model = onnx.load_model_from_string(authority_bytes)
    sanitized_candidate = sanitize(candidate_model)
    structure = structural_audit(candidate_model)
    candidate_profile = profile_bytes(candidate_bytes, "candidate")
    authority_profile = profile_bytes(authority_bytes, "authority")
    candidate_official = official_profile(
        candidate_model, "candidate", require_correct=False
    )
    authority_official = official_profile(
        authority_model, "authority", require_correct=True
    )

    catalog_text = PRIVATE_ZERO_CATALOG.read_text(encoding="utf-8")
    public_overfit_text = PUBLIC_OVERFIT_SOURCE.read_text(encoding="utf-8")
    policy_classification = {
        "private_zero_catalog_path": relative(PRIVATE_ZERO_CATALOG),
        "private_zero_catalog_sha256": file_digest(PRIVATE_ZERO_CATALOG),
        "task355_in_private_zero_catalog": bool(
            re.search(r"(?<!\d)355(?!\d)", catalog_text)
        ),
        "public_overfit_source_path": relative(PUBLIC_OVERFIT_SOURCE),
        "public_overfit_source_sha256": file_digest(PUBLIC_OVERFIT_SOURCE),
        "overfit_risk_marker_present": "Overfit-risk top 10 task IDs" in public_overfit_text,
        "task355_in_public_overfit_source": "task355" in public_overfit_text,
        "review_policy": "normal_POLICY90_not_exact_correctness",
    }

    known_payload = json.loads(KNOWN_FILE.read_text(encoding="utf-8"))
    known_examples = [
        example
        for split in ("train", "test", "arc-gen")
        for example in known_payload[split]
    ]
    known_meta = {
        "file": relative(KNOWN_FILE),
        "file_sha256": file_digest(KNOWN_FILE),
        "split_counts": {
            split: len(known_payload[split]) for split in ("train", "test", "arc-gen")
        },
        "total": len(known_examples),
        "case_data_sha256": cases_digest(known_examples),
        "converter_crosscheck": converter_crosscheck(known_examples),
    }
    first_converted = scoring.convert_to_numpy(known_examples[0])
    if first_converted is None:
        raise RuntimeError("first known case conversion failed")
    truthful_trace = truthful_shape_trace(
        sanitized_candidate, first_converted["input"]
    )

    sessions = {
        name: make_session(sanitized_candidate, optimization, threads)
        for name, optimization, threads in CONFIGS
    }
    runtime: dict[str, dict[str, Any]] = {}
    runtime["known"] = evaluate(sessions, known_examples)
    print(
        json.dumps(
            {
                "dataset": "known",
                "rates": {
                    name: row["accuracy"] for name, row in runtime["known"].items()
                },
            }
        ),
        flush=True,
    )

    fresh_meta = []
    for seed in FRESH_SEEDS:
        examples = fresh_examples(seed)
        dataset_name = f"fresh_seed_{seed}"
        fresh_meta.append(
            {
                "seed": seed,
                "count": len(examples),
                "case_data_sha256": cases_digest(examples),
                "unique_inputs": len(
                    {
                        (
                            tuple(np.asarray(example["input"], dtype=np.uint8).shape),
                            np.asarray(example["input"], dtype=np.uint8).tobytes(),
                        )
                        for example in examples
                    }
                ),
                "converter_crosscheck": converter_crosscheck(examples),
            }
        )
        runtime[dataset_name] = evaluate(sessions, examples)
        print(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "rates": {
                        name: row["accuracy"]
                        for name, row in runtime[dataset_name].items()
                    },
                }
            ),
            flush=True,
        )

    runtime_rows = [
        row for dataset in runtime.values() for row in dataset.values()
    ]
    aggregate = {
        "dataset_config_evaluations": len(runtime_rows),
        "case_config_executions": sum(row["total"] for row in runtime_rows),
        "errors": sum(row["errors"] for row in runtime_rows),
        "nonfinite_cases": sum(row["nonfinite_cases"] for row in runtime_rows),
        "nonfinite_elements": sum(row["nonfinite_elements"] for row in runtime_rows),
        "output_shape_mismatches": sum(
            row["shape_mismatches"] for row in runtime_rows
        ),
        "small_positive_elements_0_to_0_25": sum(
            row["small_positive_elements_0_to_0_25"] for row in runtime_rows
        ),
        "sign_config_mismatch_cases": sum(
            row["sign_config_mismatch_cases"] for row in runtime_rows
        ),
        "sign_config_mismatch_cells": sum(
            row["sign_config_mismatch_cells"] for row in runtime_rows
        ),
        "raw_config_mismatch_cases": sum(
            row["raw_config_mismatch_cases"] for row in runtime_rows
        ),
        "minimum_positive": min(
            row["minimum_positive"]
            for row in runtime_rows
            if row["minimum_positive"] is not None
        ),
        "maximum_nonpositive": max(
            row["maximum_nonpositive"]
            for row in runtime_rows
            if row["maximum_nonpositive"] is not None
        ),
        "official_converter_crosscheck_cases": (
            known_meta["converter_crosscheck"]["cases"]
            + sum(item["converter_crosscheck"]["cases"] for item in fresh_meta)
        ),
    }

    gates = {
        "candidate_sha_pinned": digest(candidate_bytes) == CANDIDATE_SHA256,
        "authority_zip_sha_pinned": digest(authority_zip_bytes) == AUTHORITY_ZIP_SHA256,
        "official_sanitizer_accepted": sanitized_candidate is not None,
        "candidate_cost249": bool(
            candidate_profile == {"memory": 227, "params": 22, "cost": 249}
            and candidate_official
            and candidate_official.get("cost") == 249
        ),
        "authority_cost250_correct": bool(
            authority_profile == {"memory": 228, "params": 22, "cost": 250}
            and authority_official
            and authority_official.get("cost") == 250
            and authority_official.get("correct")
        ),
        "full_strict_static_standard": structure["pass_without_runtime_truth"],
        "conv_bias_ub0": structure["conv_bias_ub0"],
        "lookup_audit_clear": structure["lookup_audit_clear"],
        "giant_audit_clear": bool(
            not structure["giant_initializers"]
            and not structure["giant_intermediates"]
            and not structure["giant_node_arity"]
        ),
        "truthful_no_cloak_all_four_configs": all(
            item["truthful"] for item in truthful_trace.values()
        ),
        "known_all_cases_all_configs": all(
            row["total"] == len(known_examples) for row in runtime["known"].values()
        ),
        "fresh_each10000_all_configs": bool(
            len(fresh_meta) == 2
            and all(item["count"] == FRESH_PER_SEED for item in fresh_meta)
            and all(
                row["total"] == FRESH_PER_SEED
                for dataset_name, dataset in runtime.items()
                if dataset_name.startswith("fresh_seed_")
                for row in dataset.values()
            )
        ),
        "all_accuracy_at_least_90": all(row["policy90"] for row in runtime_rows),
        "runtime_nonfinite_shape_smallpositive_zero": all(
            runtime_row_pass(row) for row in runtime_rows
        ),
        "config_sign_stable": aggregate["sign_config_mismatch_cases"] == 0,
        "config_raw_stable": aggregate["raw_config_mismatch_cases"] == 0,
        "official_converter_exact": bool(
            known_meta["converter_crosscheck"]["exact"]
            and all(item["converter_crosscheck"]["exact"] for item in fresh_meta)
        ),
        "normal_policy90_classification": bool(
            not policy_classification["task355_in_private_zero_catalog"]
            and policy_classification["overfit_risk_marker_present"]
            and policy_classification["task355_in_public_overfit_source"]
        ),
    }
    accepted = all(gates.values())
    payload = {
        "task": 355,
        "lane": "agent_review_task355_policy90_284",
        "decision": "PASS_POLICY90_INDEPENDENT_REVIEW" if accepted else "FAIL_CLOSED",
        "accepted": accepted,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "independence": {
            "lane283_screen_used_as_audit_input": False,
            "candidate_binary_used": relative(CANDIDATE),
            "generator_used": relative(GENERATOR_FILE),
            "fresh_seeds": list(FRESH_SEEDS),
            "candidate_promoted": False,
            "root_or_others_71407_written": False,
            "kimi_used": False,
        },
        "candidate": {
            "path": relative(CANDIDATE),
            "sha256": digest(candidate_bytes),
            "file_bytes": len(candidate_bytes),
            "profile": candidate_profile,
            "official_profile": candidate_official,
            "sanitized_sha256": digest(sanitized_candidate.SerializeToString()),
            "sanitized_file_bytes": len(sanitized_candidate.SerializeToString()),
            "cost_delta_vs_authority": candidate_profile["cost"] - authority_profile["cost"],
            "score_gain_vs_authority": math.log(
                authority_profile["cost"] / candidate_profile["cost"]
            ),
        },
        "authority": {
            "zip": relative(AUTHORITY_ZIP),
            "zip_sha256": digest(authority_zip_bytes),
            "member": "task355.onnx",
            "member_sha256": digest(authority_bytes),
            "member_bytes": len(authority_bytes),
            "profile": authority_profile,
            "official_profile": authority_official,
        },
        "generator": {
            "path": relative(GENERATOR_FILE),
            "sha256": file_digest(GENERATOR_FILE),
        },
        "policy_classification": policy_classification,
        "structure": structure,
        "truthful_runtime_shape_trace_after_sanitizer": truthful_trace,
        "known": known_meta,
        "fresh_generation": fresh_meta,
        "runtime_configs": [
            {"name": name, "optimization": optimization, "threads": threads}
            for name, optimization, threads in CONFIGS
        ],
        "runtime": runtime,
        "aggregate": aggregate,
        "gates": gates,
        "policy": {
            "threshold": POLICY_THRESHOLD,
            "fail_closed": True,
            "acceptance_kind": "normal_POLICY90_not_exact_correctness",
            "public_overfit_risk_requires_fresh_stress": True,
            "candidate_promoted": False,
            "root_or_others_71407_written": False,
            "kimi_used": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "accepted": accepted,
                "failed_gates": payload["failed_gates"],
                "candidate": payload["candidate"],
                "authority": payload["authority"],
                "aggregate": aggregate,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

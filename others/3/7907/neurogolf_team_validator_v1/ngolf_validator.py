#!/usr/bin/env python3
"""NeuroGolf team validation and submission toolkit.

The goal is reproducible candidate evaluation across local notebooks and GPT sessions.
It mirrors the current competition metric as closely as possible:

    task_cost = runtime_tensor_memory + parameter_elements
    task_score = max(1, 25 - ln(task_cost)), with zero cost scoring 25

The runtime-memory calculation uses ONNX Runtime profiling and keeps the maximum
observed allocation for each tensor across all known examples.

This file intentionally contains the complete workflow in one importable module.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence
import zipfile

import numpy as np
import onnx
import onnxruntime as ort

# -----------------------------------------------------------------------------
# Competition constants
# -----------------------------------------------------------------------------
BATCH = 1
CHANNELS = 10
HEIGHT = 30
WIDTH = 30
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
TASK_IDS = tuple(range(1, 401))
FILE_SIZE_LIMIT = int(1.44 * 1024 * 1024)
EXCLUDED_OP_TYPES = {
    "LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"
}
ALLOWED_OPSET_DOMAINS = {"", "ai.onnx"}
SPLITS = ("train", "test", "arc-gen")
TASK_RE = re.compile(r"task[_-]?(\d{1,3})\.onnx$", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class ValidationCounts:
    right: int = 0
    wrong: int = 0
    skipped: int = 0
    errors: int = 0
    total_seen: int = 0


@dataclass
class DifferentialResult:
    requested: int = 0
    executable: int = 0
    raw_equal: int = 0
    threshold_equal: int = 0
    mismatches: int = 0
    skipped_both_failed: int = 0
    skipped_one_failed: int = 0
    max_abs_difference: float = 0.0
    first_mismatch: dict[str, Any] | None = None


@dataclass
class ModelAudit:
    task: int
    source: str
    sha256: str = ""
    serialized_size: int = 0
    valid: bool = False
    preflight_ok: bool = False
    known: ValidationCounts = field(default_factory=ValidationCounts)
    memory: int | None = None
    params: int | None = None
    cost: int | None = None
    score: float | None = None
    node_count: int = 0
    initializer_count: int = 0
    error: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateDecision:
    task: int
    baseline: ModelAudit
    candidate: ModelAudit
    differential: DifferentialResult | None = None
    cost_reduction: int | None = None
    projected_gain: float | None = None
    verdict: str = "REJECT"
    reasons: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def score_from_cost(cost: int | None) -> float | None:
    if cost is None:
        return None
    if cost == 0:
        return 25.0
    return max(1.0, 25.0 - math.log(cost))


def task_number_from_name(name: str) -> int | None:
    match = TASK_RE.search(Path(name).name)
    return int(match.group(1)) if match else None


def parse_task_list(text: str | None) -> list[int] | None:
    if not text:
        return None
    result: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_safe(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=False))


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, full))
        elif isinstance(value, (list, tuple, set)):
            out[full] = json.dumps(json_safe(value), separators=(",", ":"))
        else:
            out[full] = value
    return out


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    flattened = [flatten_dict(row) for row in rows]
    fields: list[str] = []
    seen: set[str] = set()
    for row in flattened:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flattened)


# -----------------------------------------------------------------------------
# Dataset handling
# -----------------------------------------------------------------------------
def ensure_data_dir(data_dir: str | Path, data_zip: str | Path | None = None) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    present = len(list(data_dir.glob("task*.json")))
    if present >= 400:
        return data_dir
    if data_zip is None:
        raise FileNotFoundError(
            f"Only {present} task JSON files found in {data_dir}; provide --data-zip."
        )
    with zipfile.ZipFile(data_zip) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".json") and task_number_from_name(
                Path(name).with_suffix(".onnx").name
            ) is not None:
                destination = data_dir / Path(name).name
                destination.write_bytes(archive.read(name))
    present = len(list(data_dir.glob("task*.json")))
    if present < 400:
        raise RuntimeError(f"Expected 400 task JSON files; found {present} in {data_dir}.")
    return data_dir


def load_task_examples(data_dir: str | Path, task: int) -> dict[str, list[dict[str, Any]]]:
    path = Path(data_dir) / f"task{task:03d}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def example_fits(example: Mapping[str, Any]) -> bool:
    for key in ("input", "output"):
        grid = example.get(key)
        if not grid or not grid[0]:
            return False
        if len(grid) > HEIGHT or len(grid[0]) > WIDTH:
            return False
        for row in grid:
            if len(row) != len(grid[0]):
                return False
            if any((not isinstance(color, int)) or color < 0 or color >= CHANNELS for color in row):
                return False
    return True


def grid_to_one_hot(grid: Sequence[Sequence[int]]) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"Grid must be rank 2, got {array.shape}.")
    rows, cols = array.shape
    if rows > HEIGHT or cols > WIDTH:
        raise ValueError(f"Grid {array.shape} exceeds {HEIGHT}x{WIDTH}.")
    if array.size and (array.min() < 0 or array.max() >= CHANNELS):
        raise ValueError("Grid contains a color outside 0..9.")
    one_hot = np.zeros(INPUT_SHAPE, dtype=np.float32)
    rr, cc = np.indices(array.shape)
    one_hot[0, array, rr, cc] = 1.0
    return one_hot


def expected_one_hot(example: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return grid_to_one_hot(example["input"]), grid_to_one_hot(example["output"])


def iter_known_examples(
    examples: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    for split in SPLITS:
        for index, example in enumerate(examples.get(split, [])):
            yield split, index, example


# -----------------------------------------------------------------------------
# ZIP handling
# -----------------------------------------------------------------------------
class SubmissionZip:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._entries: dict[int, tuple[str, bytes]] | None = None
        self._order: list[str] | None = None

    def _load(self) -> None:
        if self._entries is not None:
            return
        entries: dict[int, tuple[str, bytes]] = {}
        with zipfile.ZipFile(self.path) as archive:
            self._order = archive.namelist()
            for info in archive.infolist():
                task = task_number_from_name(info.filename)
                if task is None:
                    continue
                if task in entries:
                    raise RuntimeError(f"Duplicate task {task:03d} in {self.path}.")
                entries[task] = (info.filename, archive.read(info.filename))
        self._entries = entries

    @property
    def tasks(self) -> list[int]:
        self._load()
        assert self._entries is not None
        return sorted(self._entries)

    @property
    def order(self) -> list[str]:
        self._load()
        assert self._order is not None
        return list(self._order)

    def model_bytes(self, task: int) -> bytes:
        self._load()
        assert self._entries is not None
        return self._entries[task][1]

    def model_name(self, task: int) -> str:
        self._load()
        assert self._entries is not None
        return self._entries[task][0]

    def changed_tasks(self, other: "SubmissionZip") -> list[int]:
        tasks = sorted(set(self.tasks) | set(other.tasks))
        changed: list[int] = []
        for task in tasks:
            if task not in self.tasks or task not in other.tasks:
                changed.append(task)
                continue
            if sha256_bytes(self.model_bytes(task)) != sha256_bytes(other.model_bytes(task)):
                changed.append(task)
        return changed


def audit_submission_zip(path: str | Path, require_400: bool = True) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "archive_size": path.stat().st_size,
        "entries": 0,
        "onnx_count": 0,
        "tasks": [],
        "duplicates": [],
        "missing_tasks": [],
        "over_limit": [],
        "max_model_size": 0,
        "valid": False,
    }
    seen: set[int] = set()
    with zipfile.ZipFile(path) as archive:
        result["entries"] = len(archive.namelist())
        for info in archive.infolist():
            task = task_number_from_name(info.filename)
            if task is None:
                continue
            result["onnx_count"] += 1
            if task in seen:
                result["duplicates"].append(task)
            seen.add(task)
            data = archive.read(info.filename)
            result["max_model_size"] = max(result["max_model_size"], len(data))
            if len(data) > FILE_SIZE_LIMIT:
                result["over_limit"].append({"task": task, "size": len(data)})
    result["tasks"] = sorted(seen)
    result["missing_tasks"] = sorted(set(TASK_IDS) - seen) if require_400 else []
    result["valid"] = (
        not result["duplicates"]
        and not result["missing_tasks"]
        and not result["over_limit"]
        and (result["onnx_count"] == 400 if require_400 else True)
    )
    return result


def _fixed_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def build_submission(
    baseline_zip: str | Path,
    replacements: Mapping[int, str | Path | bytes],
    output_zip: str | Path,
) -> dict[str, Any]:
    baseline_zip = Path(baseline_zip)
    output_zip = Path(output_zip)
    replacement_bytes: dict[int, bytes] = {}
    for task, value in replacements.items():
        replacement_bytes[int(task)] = value if isinstance(value, bytes) else Path(value).read_bytes()

    changed_entries: list[str] = []
    with zipfile.ZipFile(baseline_zip) as source, zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            task = task_number_from_name(info.filename)
            if task in replacement_bytes:
                candidate = replacement_bytes[task]
                if len(candidate) > FILE_SIZE_LIMIT:
                    raise RuntimeError(
                        f"task{task:03d} is {len(candidate)} bytes, above {FILE_SIZE_LIMIT}."
                    )
                if candidate != data:
                    changed_entries.append(info.filename)
                data = candidate
            target.writestr(_fixed_zip_info(info.filename), data)

    audit = audit_submission_zip(output_zip)
    audit["changed_entries"] = changed_entries
    audit["baseline_sha256"] = sha256_file(baseline_zip)
    return audit


def isolate_task(
    baseline_zip: str | Path,
    source_zip: str | Path,
    task: int,
    output_zip: str | Path,
) -> dict[str, Any]:
    source = SubmissionZip(source_zip)
    return build_submission(baseline_zip, {task: source.model_bytes(task)}, output_zip)


# -----------------------------------------------------------------------------
# Model checks and official-like metric
# -----------------------------------------------------------------------------
def sanitize_model(model: onnx.ModelProto) -> onnx.ModelProto | None:
    model = copy.deepcopy(model)
    for node in model.graph.node:
        if not node.output or not node.output[0]:
            return None
        node.name = node.output[0]
        if "kernel_time" in node.output[0]:
            return None

    name_map: dict[str, str] = {}
    counter = 0

    def safe(old_name: str) -> str:
        nonlocal counter
        if not old_name or old_name in {"input", "output"}:
            return old_name
        if old_name not in name_map:
            name_map[old_name] = f"safe_name_{counter}"
            counter += 1
        return name_map[old_name]

    for item in model.graph.input:
        item.name = safe(item.name)
    for item in model.graph.initializer:
        item.name = safe(item.name)
    for item in model.graph.sparse_initializer:
        item.values.name = safe(item.values.name)
        item.indices.name = safe(item.indices.name)
    for node in model.graph.node:
        for index in range(len(node.input)):
            node.input[index] = safe(node.input[index])
        for index in range(len(node.output)):
            node.output[index] = safe(node.output[index])
        node.name = node.output[0]
    for item in model.graph.output:
        item.name = safe(item.name)
    for item in model.graph.value_info:
        item.name = safe(item.name)
    for node in model.graph.node:
        node.name = node.output[0]
    return model


def preflight_model(model: onnx.ModelProto, serialized_size: int) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if serialized_size > FILE_SIZE_LIMIT:
        errors.append(f"serialized size {serialized_size} exceeds {FILE_SIZE_LIMIT}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        errors.append("graph must have exactly one input and one output")
    if model.functions:
        errors.append("model functions are not allowed")
    for opset in model.opset_import:
        if opset.domain not in ALLOWED_OPSET_DOMAINS:
            errors.append(f"custom opset domain is not allowed: {opset.domain!r}")
    seen_value_info: set[str] = set()
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if item.name in seen_value_info:
            errors.append(f"duplicate graph type entry: {item.name}")
        seen_value_info.add(item.name)
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in EXCLUDED_OP_TYPES or "Sequence" in node.op_type:
            errors.append(f"excluded operator: {node.op_type}")
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                errors.append(f"subgraph attribute is not allowed: {node.name}/{attr.name}")
        if not node.output or not node.output[0]:
            errors.append("every node must have a non-empty first output")
        elif "kernel_time" in node.output[0]:
            errors.append("tensor output contains reserved kernel_time text")
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        errors.append(f"onnx checker failed: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
        for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            if item.type.HasField("sequence_type"):
                errors.append(f"sequence tensor is not allowed: {item.name}")
                continue
            if not item.type.HasField("tensor_type"):
                continue
            tensor_type = item.type.tensor_type
            if not tensor_type.HasField("shape"):
                errors.append(f"missing static shape: {item.name}")
                continue
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                    errors.append(f"non-static or nonpositive shape: {item.name}")
                    break
    except Exception as exc:
        errors.append(f"strict shape inference failed: {exc}")
    if model.ir_version != 10:
        warnings.append(f"IR version is {model.ir_version}; competition baseline often uses 10")
    return not errors, errors, warnings


def calculate_params(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(dim <= 0 for dim in init.dims):
            return None
        params += math.prod(init.dims) if init.dims else 1
    for sparse in model.graph.sparse_initializer:
        if any(dim <= 0 for dim in sparse.values.dims):
            return None
        params += math.prod(sparse.values.dims) if sparse.values.dims else 1
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(dim <= 0 for dim in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims) if attr.t.dims else 1
            elif attr.name == "sparse_value":
                if any(dim <= 0 for dim in attr.sparse_tensor.values.dims):
                    return None
                params += math.prod(attr.sparse_tensor.values.dims) if attr.sparse_tensor.values.dims else 1
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return int(params)


def calculate_runtime_memory(model: onnx.ModelProto, trace_path: str | Path) -> int | None:
    onnx.checker.check_model(model, full_check=True)
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        return None
    initializer_names = {item.name for item in graph.initializer}
    initializer_names.update(item.name for item in graph.sparse_initializer)
    io_names = {item.name for item in list(graph.input) + list(graph.output)}
    if io_names.intersection(initializer_names) or model.functions:
        return None
    for opset in model.opset_import:
        if opset.domain not in ALLOWED_OPSET_DOMAINS:
            return None

    node_outputs: dict[str, list[str]] = {}
    tensor_names: set[str] = set()
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                return None
        node_outputs[node.name] = list(node.output)
        tensor_names.update(name for name in node.output if name)

    tensor_map = {
        item.name: item
        for item in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    tensor_names.update(tensor_map)
    tensor_memory: dict[str, int] = {}
    tensor_dtypes: dict[str, Any] = {}

    for name in tensor_names:
        item = tensor_map.get(name)
        if item is None:
            return None
        if item.type.HasField("sequence_type"):
            return None
        if not item.type.HasField("tensor_type"):
            continue
        tensor_type = item.type.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        elements = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            elements *= int(dim.dim_value)
        if name in {"input", "output"}:
            continue
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        tensor_memory[name] = int(elements * np.dtype(dtype).itemsize)
        tensor_dtypes[name] = dtype

    seen: set[str] = set()
    for item in list(graph.input) + list(graph.value_info) + list(graph.output):
        if item.name in seen:
            return None
        seen.add(item.name)
    for node in graph.node:
        for output_name in node.output:
            if output_name and output_name != "output":
                item = tensor_map.get(output_name)
                if item is None or not item.type.HasField("tensor_type"):
                    return None

    trace_data = json.loads(Path(trace_path).read_text())
    for event in trace_data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        output_shapes = event["args"].get("output_type_shape")
        if output_shapes is None:
            continue
        event_name = event.get("name", "")
        node_name = event_name.replace("_kernel_time", "")
        if node_name not in node_outputs:
            continue
        for index, shape_dict in enumerate(output_shapes):
            if index >= len(node_outputs[node_name]):
                continue
            output_name = node_outputs[node_name][index]
            if output_name not in tensor_dtypes:
                continue
            itemsize = np.dtype(tensor_dtypes[output_name]).itemsize
            memory = int(itemsize * sum(math.prod(dims) for dims in shape_dict.values()))
            tensor_memory[output_name] = max(tensor_memory[output_name], memory)
    return int(sum(tensor_memory.values()))


def make_session(model: onnx.ModelProto, profile_prefix: str | Path | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_output(session: ort.InferenceSession, one_hot_input: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(session.run([output_name], {input_name: one_hot_input})[0])


def validate_known_examples(
    session: ort.InferenceSession,
    examples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[ValidationCounts, list[dict[str, Any]]]:
    counts = ValidationCounts()
    failures: list[dict[str, Any]] = []
    for split, index, example in iter_known_examples(examples):
        counts.total_seen += 1
        if not example_fits(example):
            counts.skipped += 1
            continue
        try:
            one_hot_input, expected = expected_one_hot(example)
            raw = run_output(session, one_hot_input)
            actual = (raw > 0).astype(np.float32)
            if np.array_equal(actual, expected):
                counts.right += 1
            else:
                counts.wrong += 1
                if len(failures) < 5:
                    failures.append({
                        "split": split,
                        "index": index,
                        "actual_positive_count": int(np.count_nonzero(actual)),
                        "expected_positive_count": int(np.count_nonzero(expected)),
                        "different_cells": int(np.count_nonzero(actual != expected)),
                    })
        except Exception as exc:
            counts.wrong += 1
            counts.errors += 1
            if len(failures) < 5:
                failures.append({"split": split, "index": index, "error": repr(exc)})
    return counts, failures


def audit_model_bytes(
    model_bytes: bytes,
    task: int,
    data_dir: str | Path,
    source: str = "candidate",
    keep_trace: bool = False,
    trace_dir: str | Path | None = None,
) -> tuple[ModelAudit, list[dict[str, Any]]]:
    audit = ModelAudit(
        task=task,
        source=source,
        sha256=sha256_bytes(model_bytes),
        serialized_size=len(model_bytes),
    )
    failures: list[dict[str, Any]] = []
    trace_path: Path | None = None
    try:
        original = onnx.load_model_from_string(model_bytes)
        audit.node_count = len(original.graph.node)
        audit.initializer_count = len(original.graph.initializer)
        preflight_ok, errors, warnings = preflight_model(original, len(model_bytes))
        audit.preflight_ok = preflight_ok
        audit.warnings.extend(warnings)
        if errors:
            audit.error = "; ".join(errors)
            return audit, failures

        sanitized = sanitize_model(original)
        if sanitized is None:
            audit.error = "sanitization failed"
            return audit, failures

        profile_root = Path(trace_dir) if trace_dir else Path(tempfile.gettempdir())
        profile_root.mkdir(parents=True, exist_ok=True)
        profile_prefix = profile_root / (
            f"ngolf_{source}_{task:03d}_{os.getpid()}_{time.time_ns()}"
        )
        session = make_session(sanitized, profile_prefix)
        examples = load_task_examples(data_dir, task)
        audit.known, failures = validate_known_examples(session, examples)
        trace_path = Path(session.end_profiling())
        audit.memory = calculate_runtime_memory(sanitized, trace_path)
        audit.params = calculate_params(sanitized)
        if audit.memory is not None and audit.params is not None:
            audit.cost = audit.memory + audit.params
            audit.score = score_from_cost(audit.cost)
        audit.valid = (
            audit.preflight_ok
            and audit.known.wrong == 0
            and audit.cost is not None
        )
    except Exception as exc:
        audit.error = repr(exc)
    finally:
        if trace_path and trace_path.exists() and not keep_trace:
            try:
                trace_path.unlink()
            except OSError:
                pass
    return audit, failures


# -----------------------------------------------------------------------------
# Differential testing
# -----------------------------------------------------------------------------
def _random_grid(rng: np.random.Generator, mode: int) -> np.ndarray:
    height = int(rng.integers(1, HEIGHT + 1))
    width = int(rng.integers(1, WIDTH + 1))
    if mode == 0:  # dense random
        return rng.integers(0, CHANNELS, size=(height, width), dtype=np.int64)
    if mode == 1:  # sparse foreground
        grid = np.zeros((height, width), dtype=np.int64)
        mask = rng.random((height, width)) < rng.uniform(0.01, 0.20)
        grid[mask] = rng.integers(1, CHANNELS, size=int(mask.sum()), dtype=np.int64)
        return grid
    if mode == 2:  # rectangles and lines
        grid = np.zeros((height, width), dtype=np.int64)
        for _ in range(int(rng.integers(1, 8))):
            color = int(rng.integers(1, CHANNELS))
            r0 = int(rng.integers(0, height))
            c0 = int(rng.integers(0, width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            if rng.random() < 0.5:
                grid[r0:r1, c0:c1] = color
            else:
                grid[r0:r1, c0] = color
                grid[r0:r1, c1 - 1] = color
                grid[r0, c0:c1] = color
                grid[r1 - 1, c0:c1] = color
        return grid
    # stripes / repeated palette
    palette = rng.choice(CHANNELS, size=int(rng.integers(2, 6)), replace=False)
    row_pattern = palette[np.arange(width) % len(palette)]
    grid = np.repeat(row_pattern[None, :], height, axis=0)
    if rng.random() < 0.5:
        grid = np.roll(grid, int(rng.integers(0, width)), axis=1)
    return grid.astype(np.int64)


def differential_test(
    baseline_bytes: bytes,
    candidate_bytes: bytes,
    cases: int = 500,
    seed: int = 20260712,
) -> DifferentialResult:
    result = DifferentialResult(requested=cases)
    baseline_model = sanitize_model(onnx.load_model_from_string(baseline_bytes))
    candidate_model = sanitize_model(onnx.load_model_from_string(candidate_bytes))
    if baseline_model is None or candidate_model is None:
        raise RuntimeError("Could not sanitize one of the differential models.")
    baseline_session = make_session(baseline_model)
    candidate_session = make_session(candidate_model)
    rng = np.random.default_rng(seed)

    for case_index in range(cases):
        grid = _random_grid(rng, case_index % 4)
        one_hot = grid_to_one_hot(grid)
        baseline_error = candidate_error = None
        baseline_raw = candidate_raw = None
        try:
            baseline_raw = run_output(baseline_session, one_hot)
        except Exception as exc:
            baseline_error = exc
        try:
            candidate_raw = run_output(candidate_session, one_hot)
        except Exception as exc:
            candidate_error = exc

        if baseline_error is not None and candidate_error is not None:
            result.skipped_both_failed += 1
            continue
        if baseline_error is not None or candidate_error is not None:
            result.skipped_one_failed += 1
            if result.first_mismatch is None:
                result.first_mismatch = {
                    "case": case_index,
                    "grid_shape": list(grid.shape),
                    "baseline_error": repr(baseline_error) if baseline_error else None,
                    "candidate_error": repr(candidate_error) if candidate_error else None,
                }
            continue

        assert baseline_raw is not None and candidate_raw is not None
        result.executable += 1
        raw_equal = np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
        threshold_equal = np.array_equal(baseline_raw > 0, candidate_raw > 0)
        if raw_equal:
            result.raw_equal += 1
        if threshold_equal:
            result.threshold_equal += 1
        if baseline_raw.shape == candidate_raw.shape:
            finite = np.isfinite(baseline_raw) & np.isfinite(candidate_raw)
            if finite.any():
                difference = float(np.max(np.abs(baseline_raw[finite] - candidate_raw[finite])))
                result.max_abs_difference = max(result.max_abs_difference, difference)
        if not threshold_equal:
            result.mismatches += 1
            if result.first_mismatch is None:
                result.first_mismatch = {
                    "case": case_index,
                    "grid_shape": list(grid.shape),
                    "different_threshold_cells": int(
                        np.count_nonzero((baseline_raw > 0) != (candidate_raw > 0))
                    ),
                    "raw_equal": raw_equal,
                }
    return result


# -----------------------------------------------------------------------------
# Candidate decisions and ZIP comparison
# -----------------------------------------------------------------------------
def decide_candidate(
    task: int,
    baseline_audit: ModelAudit,
    candidate_audit: ModelAudit,
    differential: DifferentialResult | None,
    require_random_exact: bool = True,
) -> CandidateDecision:
    decision = CandidateDecision(
        task=task,
        baseline=baseline_audit,
        candidate=candidate_audit,
        differential=differential,
    )
    if baseline_audit.cost is not None and candidate_audit.cost is not None:
        decision.cost_reduction = baseline_audit.cost - candidate_audit.cost
    if baseline_audit.score is not None and candidate_audit.score is not None:
        decision.projected_gain = candidate_audit.score - baseline_audit.score

    if not baseline_audit.valid:
        decision.reasons.append("baseline failed local validation")
    if not candidate_audit.valid:
        decision.reasons.append("candidate failed local validation")
    if decision.cost_reduction is None or decision.cost_reduction <= 0:
        decision.reasons.append("official-like cost did not decrease")
    if differential is not None and require_random_exact:
        if differential.skipped_one_failed:
            decision.reasons.append("candidate and baseline differ in random executability")
        if differential.mismatches:
            decision.reasons.append("random threshold behavior differs")
    if not decision.reasons:
        if differential is not None and differential.executable > 0:
            decision.verdict = "ACCEPT_STRICT"
        else:
            decision.verdict = "ACCEPT_KNOWN_ONLY"
    return decision


def compare_submissions(
    baseline_zip: str | Path,
    candidate_zip: str | Path,
    data_dir: str | Path,
    tasks: Sequence[int] | None = None,
    random_cases: int = 0,
    random_seed: int = 20260712,
    require_random_exact: bool = True,
) -> list[CandidateDecision]:
    baseline = SubmissionZip(baseline_zip)
    candidate = SubmissionZip(candidate_zip)
    selected = list(tasks) if tasks is not None else baseline.changed_tasks(candidate)
    decisions: list[CandidateDecision] = []

    for position, task in enumerate(selected, 1):
        print(f"[{position}/{len(selected)}] task{task:03d}", flush=True)
        baseline_bytes = baseline.model_bytes(task)
        candidate_bytes = candidate.model_bytes(task)
        baseline_audit, _ = audit_model_bytes(
            baseline_bytes, task, data_dir, source="baseline"
        )
        candidate_audit, _ = audit_model_bytes(
            candidate_bytes, task, data_dir, source="candidate"
        )
        differential = None
        if random_cases > 0 and baseline_audit.valid and candidate_audit.valid:
            differential = differential_test(
                baseline_bytes,
                candidate_bytes,
                cases=random_cases,
                seed=random_seed + task,
            )
        decision = decide_candidate(
            task,
            baseline_audit,
            candidate_audit,
            differential,
            require_random_exact=require_random_exact,
        )
        decisions.append(decision)
        print(
            f"  cost {baseline_audit.cost} -> {candidate_audit.cost}; "
            f"gain={decision.projected_gain}; verdict={decision.verdict}",
            flush=True,
        )
    return decisions


def decisions_summary(decisions: Sequence[CandidateDecision]) -> dict[str, Any]:
    accepted = [item for item in decisions if item.verdict.startswith("ACCEPT")]
    return {
        "tasks_checked": len(decisions),
        "accepted_tasks": [item.task for item in accepted],
        "rejected_tasks": [item.task for item in decisions if item not in accepted],
        "total_cost_reduction": sum(item.cost_reduction or 0 for item in accepted),
        "total_projected_gain": sum(item.projected_gain or 0.0 for item in accepted),
        "verdict_counts": {
            verdict: sum(item.verdict == verdict for item in decisions)
            for verdict in sorted({item.verdict for item in decisions})
        },
    }


# -----------------------------------------------------------------------------
# GPT handoff generation
# -----------------------------------------------------------------------------
def write_gpt_handoff(
    output: str | Path,
    baseline_zip: str | Path,
    baseline_score: float | None,
    decisions: Sequence[CandidateDecision] | None = None,
) -> None:
    baseline_zip = Path(baseline_zip)
    lines = [
        "# NeuroGolf GPT Session Handoff",
        "",
        "## Current baseline",
        f"- ZIP: `{baseline_zip.name}`",
        f"- SHA-256: `{sha256_file(baseline_zip)}`",
        f"- Leaderboard score: `{baseline_score if baseline_score is not None else 'UPDATE_ME'}`",
        "- Treat this ZIP as immutable. Build every candidate by replacing only intended task files.",
        "",
        "## Hard acceptance gate",
        "1. All train, test, and arc-gen examples must pass after output thresholding at `> 0`.",
        "2. Standard ONNX domains only; no functions, subgraphs, sequences, or excluded operators.",
        "3. Strict shape inference and ONNX Runtime execution must succeed.",
        "4. Official-like `runtime memory + parameter elements` must strictly decrease.",
        "5. Preserve archive order, all 400 tasks, and the 1.44 MiB per-file limit.",
        "6. Run randomized differential testing against the baseline. Isolate any non-exact candidate.",
        "7. Never batch a candidate that has previously hurt the leaderboard unless it is isolated again.",
        "",
        "## Metric rules",
        "- `cost = max-observed runtime bytes for all non-input/output tensors + initializer/Constant element count`",
        "- `score = max(1, 25 - ln(cost))`; zero cost scores 25.",
        "- Node count, MACs, FLOPs, and serialized ZIP size are not direct score terms.",
        "- A longer graph with tiny tensors can beat a short graph with large tensors.",
        "",
        "## Recommended workflow",
        "```bash",
        "python ngolf_validator.py compare-zips \\",
        "  --baseline baseline.zip --candidate candidate.zip \\",
        "  --data-dir neurogolf_2026_data --random-cases 500 \\",
        "  --out-json comparison.json --out-csv comparison.csv",
        "```",
        "",
    ]
    if decisions:
        summary = decisions_summary(decisions)
        lines.extend([
            "## Latest local comparison",
            f"- Accepted tasks: `{summary['accepted_tasks']}`",
            f"- Total projected gain: `{summary['total_projected_gain']:.9f}`",
            "",
        ])
    lines.extend([
        "## What to give the next GPT session",
        "- The current best ZIP and its leaderboard score.",
        "- The candidate ONNX file or candidate ZIP.",
        "- `comparison.json`, `comparison.csv`, and the latest signal ledger.",
        "- A clear instruction to optimize only from the current best baseline.",
        "",
        "## Proven strategy hierarchy",
        "1. Direct-output synthesis that avoids scored intermediates.",
        "2. Exact in-Einsum factorization with no new activation tensors.",
        "3. Procedural generation of constants when activation memory is cheaper than parameters.",
        "4. Packed integer/bitwise state and direct selection instead of materialized tables.",
        "5. Optional-default input removal and exact scalar initializer deduplication.",
        "6. Sanitized ORT-basic optimization, followed by official profiling and exact validation.",
    ])
    Path(output).write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _decision_rows(decisions: Sequence[CandidateDecision]) -> list[dict[str, Any]]:
    return [json_safe(item) for item in decisions]


def command_audit_zip(args: argparse.Namespace) -> int:
    result = audit_submission_zip(args.zip, require_400=not args.allow_partial)
    print(json.dumps(result, indent=2))
    if args.out_json:
        write_json(args.out_json, result)
    return 0 if result["valid"] else 2


def command_validate_task(args: argparse.Namespace) -> int:
    data_dir = ensure_data_dir(args.data_dir, args.data_zip)
    if args.candidate_model:
        candidate_bytes = Path(args.candidate_model).read_bytes()
    else:
        candidate_bytes = SubmissionZip(args.candidate_zip).model_bytes(args.task)
    candidate_audit, failures = audit_model_bytes(
        candidate_bytes,
        args.task,
        data_dir,
        source="candidate",
        keep_trace=args.keep_trace,
        trace_dir=args.trace_dir,
    )
    payload: dict[str, Any] = {
        "candidate": candidate_audit,
        "known_failures": failures,
    }
    if args.baseline_zip or args.baseline_model:
        baseline_bytes = (
            Path(args.baseline_model).read_bytes()
            if args.baseline_model
            else SubmissionZip(args.baseline_zip).model_bytes(args.task)
        )
        baseline_audit, baseline_failures = audit_model_bytes(
            baseline_bytes, args.task, data_dir, source="baseline"
        )
        differential = None
        if args.random_cases > 0 and baseline_audit.valid and candidate_audit.valid:
            differential = differential_test(
                baseline_bytes,
                candidate_bytes,
                cases=args.random_cases,
                seed=args.seed,
            )
        decision = decide_candidate(
            args.task,
            baseline_audit,
            candidate_audit,
            differential,
            require_random_exact=not args.allow_random_mismatch,
        )
        payload.update({
            "baseline": baseline_audit,
            "baseline_failures": baseline_failures,
            "differential": differential,
            "decision": decision,
        })
    print(json.dumps(json_safe(payload), indent=2))
    if args.out_json:
        write_json(args.out_json, payload)
    decision = payload.get("decision")
    if isinstance(decision, CandidateDecision):
        return 0 if decision.verdict.startswith("ACCEPT") else 3
    return 0 if candidate_audit.valid else 2


def command_compare_zips(args: argparse.Namespace) -> int:
    data_dir = ensure_data_dir(args.data_dir, args.data_zip)
    tasks = parse_task_list(args.tasks)
    decisions = compare_submissions(
        args.baseline,
        args.candidate,
        data_dir,
        tasks=tasks,
        random_cases=args.random_cases,
        random_seed=args.seed,
        require_random_exact=not args.allow_random_mismatch,
    )
    payload = {
        "baseline": str(args.baseline),
        "baseline_sha256": sha256_file(args.baseline),
        "candidate": str(args.candidate),
        "candidate_sha256": sha256_file(args.candidate),
        "summary": decisions_summary(decisions),
        "decisions": decisions,
    }
    print(json.dumps(json_safe(payload["summary"]), indent=2))
    if args.out_json:
        write_json(args.out_json, payload)
    if args.out_csv:
        write_csv(args.out_csv, _decision_rows(decisions))
    if args.handoff:
        write_gpt_handoff(
            args.handoff,
            args.candidate if args.promote_candidate else args.baseline,
            args.baseline_score,
            decisions,
        )
    return 0


def command_build_submission(args: argparse.Namespace) -> int:
    replacements: dict[int, Path] = {}
    for item in args.replace:
        if "=" not in item:
            raise ValueError("Each --replace must be TASK=MODEL_PATH, e.g. 101=task101.onnx")
        task_text, path_text = item.split("=", 1)
        replacements[int(task_text)] = Path(path_text)
    audit = build_submission(args.baseline, replacements, args.output)
    print(json.dumps(audit, indent=2))
    if args.out_json:
        write_json(args.out_json, audit)
    return 0 if audit["valid"] else 2


def command_isolate_task(args: argparse.Namespace) -> int:
    audit = isolate_task(args.baseline, args.source, args.task, args.output)
    print(json.dumps(audit, indent=2))
    if args.out_json:
        write_json(args.out_json, audit)
    return 0 if audit["valid"] else 2


def command_handoff(args: argparse.Namespace) -> int:
    write_gpt_handoff(args.output, args.baseline, args.score, None)
    print(args.output)
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NeuroGolf official-like validator, comparer, and submission builder."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-zip", help="Check archive integrity and file limits.")
    audit.add_argument("--zip", required=True)
    audit.add_argument("--allow-partial", action="store_true")
    audit.add_argument("--out-json")
    audit.set_defaults(func=command_audit_zip)

    validate = subparsers.add_parser("validate-task", help="Profile and validate one task.")
    validate.add_argument("--task", type=int, required=True)
    group = validate.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate-model")
    group.add_argument("--candidate-zip")
    baseline_group = validate.add_mutually_exclusive_group()
    baseline_group.add_argument("--baseline-model")
    baseline_group.add_argument("--baseline-zip")
    validate.add_argument("--data-dir", required=True)
    validate.add_argument("--data-zip")
    validate.add_argument("--random-cases", type=int, default=500)
    validate.add_argument("--seed", type=int, default=20260712)
    validate.add_argument("--allow-random-mismatch", action="store_true")
    validate.add_argument("--keep-trace", action="store_true")
    validate.add_argument("--trace-dir")
    validate.add_argument("--out-json")
    validate.set_defaults(func=command_validate_task)

    compare = subparsers.add_parser("compare-zips", help="Validate changed or selected tasks.")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--data-dir", required=True)
    compare.add_argument("--data-zip")
    compare.add_argument("--tasks", help="Comma/range list, e.g. 36,101,233-235")
    compare.add_argument("--random-cases", type=int, default=0)
    compare.add_argument("--seed", type=int, default=20260712)
    compare.add_argument("--allow-random-mismatch", action="store_true")
    compare.add_argument("--out-json")
    compare.add_argument("--out-csv")
    compare.add_argument("--handoff")
    compare.add_argument("--baseline-score", type=float)
    compare.add_argument("--promote-candidate", action="store_true")
    compare.set_defaults(func=command_compare_zips)

    build = subparsers.add_parser("build-submission", help="Replace selected ONNX files in a baseline ZIP.")
    build.add_argument("--baseline", required=True)
    build.add_argument("--replace", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--out-json")
    build.set_defaults(func=command_build_submission)

    isolate = subparsers.add_parser("isolate-task", help="Copy one task from a source ZIP into a baseline ZIP.")
    isolate.add_argument("--baseline", required=True)
    isolate.add_argument("--source", required=True)
    isolate.add_argument("--task", type=int, required=True)
    isolate.add_argument("--output", required=True)
    isolate.add_argument("--out-json")
    isolate.set_defaults(func=command_isolate_task)

    handoff = subparsers.add_parser("write-handoff", help="Create a GPT session handoff document.")
    handoff.add_argument("--baseline", required=True)
    handoff.add_argument("--score", type=float)
    handoff.add_argument("--output", required=True)
    handoff.set_defaults(func=command_handoff)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

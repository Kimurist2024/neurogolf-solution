#!/usr/bin/env python3
"""All-400 exact/no-op simplification scan against immutable 8009.46."""

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
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper, shape_inference


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
ACTIVE_MANIFEST = ROOT / "others/71407/MANIFEST.json"
CANONICAL_COSTS = ROOT / "scripts/golf/loop_8004_42_plus20/root_mem_census_119/canonical_costs.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
CANDIDATE_DIR = HERE / "candidates"
EVIDENCE = HERE / "evidence.json"
EXPECTED_IO = (1, 10, 30, 30)
PRIORITY_MIN = 150
PRIORITY_MAX = 500
FRESH_PER_SEED = 2_000
CONFIGS = (
    ("disable_threads1", True, 1),
    ("disable_threads4", True, 4),
    ("default_threads1", False, 1),
    ("default_threads4", False, 4),
)
BASE_CONFIG = "disable_threads1"
PROFILES = ("cleanup", "dedupe", "unary", "neutral", "pair_cancel", "combined")
PRIVATE_ZERO_CATALOG = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}
ASSIGNED_ACTIVE22 = {
    7, 12, 13, 66, 90, 101, 134, 158, 161, 175, 192,
    205, 209, 226, 245, 310, 319, 328, 333, 344, 349, 366,
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "exact_noop285_screen",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/screen_all.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def case_id(example: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("input", "output"):
        array = np.asarray(example[key], dtype=np.uint8)
        digest.update(np.asarray(array.shape, dtype=np.int16).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_shape(value: onnx.ValueInfoProto | None) -> tuple[int, ...] | None:
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims = value.type.tensor_type.shape.dim
    if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
        return None
    return tuple(int(dim.dim_value) for dim in dims)


def inferred_values(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    return {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }


def constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for item in model.graph.initializer:
        try:
            arrays[item.name] = numpy_helper.to_array(item)
        except Exception:
            pass
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1 or not node.output[0]:
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        try:
            if "value" in attrs:
                arrays[node.output[0]] = numpy_helper.to_array(attrs["value"])
            elif "value_float" in attrs:
                arrays[node.output[0]] = np.asarray(attrs["value_float"], dtype=np.float32)
            elif "value_floats" in attrs:
                arrays[node.output[0]] = np.asarray(attrs["value_floats"], dtype=np.float32)
            elif "value_int" in attrs:
                arrays[node.output[0]] = np.asarray(attrs["value_int"], dtype=np.int64)
            elif "value_ints" in attrs:
                arrays[node.output[0]] = np.asarray(attrs["value_ints"], dtype=np.int64)
        except Exception:
            pass
    return arrays


def remove_dead_value_info(model: onnx.ModelProto) -> list[str]:
    live = {item.name for item in model.graph.input}
    live.update(item.name for item in model.graph.output)
    live.update(item.name for item in model.graph.initializer)
    live.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
    removed = [item.name for item in model.graph.value_info if item.name not in live]
    if removed:
        keep = [item for item in model.graph.value_info if item.name in live]
        del model.graph.value_info[:]
        model.graph.value_info.extend(keep)
    return removed


def cleanup_dead(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
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
        pending.extend(name for name in model.graph.node[index].input if name)
    nodes = list(model.graph.node)
    dead = [index for index in range(len(nodes)) if index not in live_nodes]
    for index in dead:
        actions.append({
            "kind": "dead_node", "index": index, "op": nodes[index].op_type,
            "outputs": list(nodes[index].output),
        })
    if dead:
        keep = [node for index, node in enumerate(nodes) if index in live_nodes]
        del model.graph.node[:]
        model.graph.node.extend(keep)
    used = {name for node in model.graph.node for name in node.input if name}
    used.update(item.name for item in model.graph.output)
    used.update(item.name for item in model.graph.input)
    unused = [item.name for item in model.graph.initializer if item.name not in used]
    for name in unused:
        actions.append({"kind": "unused_initializer", "name": name})
    if unused:
        unused_set = set(unused)
        keep = [item for item in model.graph.initializer if item.name not in unused_set]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    remove_dead_value_info(model)
    return actions


def initializer_key(item: onnx.TensorProto) -> tuple[Any, ...]:
    array = np.ascontiguousarray(numpy_helper.to_array(item))
    return str(array.dtype), tuple(array.shape), array.tobytes()


def dedupe_initializers(model: onnx.ModelProto) -> list[dict[str, Any]]:
    protected = {item.name for item in model.graph.input}
    protected.update(item.name for item in model.graph.output)
    first: dict[tuple[Any, ...], str] = {}
    replacements: dict[str, str] = {}
    for item in model.graph.initializer:
        if item.name in protected:
            continue
        try:
            key = initializer_key(item)
        except Exception:
            continue
        if key in first:
            replacements[item.name] = first[key]
        else:
            first[key] = item.name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
    if replacements:
        keep = [item for item in model.graph.initializer if item.name not in replacements]
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    return [
        {"kind": "duplicate_initializer", "removed": old, "reused": new}
        for old, new in sorted(replacements.items())
    ]


def rename_tensor(model: onnx.ModelProto, source: str, target: str) -> None:
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == source:
                node.input[index] = target
        for index, name in enumerate(node.output):
            if name == source:
                node.output[index] = target
    for item in model.graph.initializer:
        if item.name == source:
            item.name = target
    # Avoid duplicate advisory records when the bypass target was already typed.
    keep_vi = [item for item in model.graph.value_info if item.name != target]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)
    for item in model.graph.value_info:
        if item.name == source:
            item.name = target


def bypass_node(model: onnx.ModelProto, index: int, source: str, proof: str) -> dict[str, Any] | None:
    nodes = list(model.graph.node)
    if not (0 <= index < len(nodes)):
        return None
    node = nodes[index]
    if len(node.output) != 1 or not node.output[0]:
        return None
    target = node.output[0]
    graph_outputs = {item.name for item in model.graph.output}
    graph_inputs = {item.name for item in model.graph.input}
    if target in graph_outputs and source in graph_inputs:
        # Keeping canonical input="input" and output="output" requires an alias node.
        return None
    keep = [item for node_index, item in enumerate(nodes) if node_index != index]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    if target in graph_outputs:
        rename_tensor(model, source, target)
    else:
        for consumer in model.graph.node:
            for input_index, name in enumerate(consumer.input):
                if name == target:
                    consumer.input[input_index] = source
    remove_dead_value_info(model)
    return {
        "kind": "bypass", "op": node.op_type, "output": target,
        "source": source, "proof": proof, "was_graph_output": target in graph_outputs,
    }


def can_bypass(model: onnx.ModelProto, target: str, source: str) -> bool:
    graph_outputs = {item.name for item in model.graph.output}
    graph_inputs = {item.name for item in model.graph.input}
    return not (target in graph_outputs and source in graph_inputs)


def transpose_perm(node: onnx.NodeProto, rank: int) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == "perm":
            return [int(value) for value in attribute.ints]
    return list(reversed(range(rank)))


def unary_noops(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    while True:
        try:
            values = inferred_values(model)
        except Exception:
            break
        arrays = constant_arrays(model)
        chosen = None
        for index, node in enumerate(model.graph.node):
            if len(node.output) != 1 or not node.output[0] or not node.input:
                continue
            source = node.input[0]
            proof = None
            if node.op_type == "Identity" and len(node.input) == 1:
                proof = "Identity(x)=x"
            elif node.op_type == "Cast" and len(node.input) == 1:
                before, after = values.get(source), values.get(node.output[0])
                if (
                    before is not None and after is not None
                    and before.type.tensor_type.elem_type == after.type.tensor_type.elem_type
                ):
                    proof = "same-dtype Cast"
            elif node.op_type == "Transpose" and len(node.input) == 1:
                shape = tensor_shape(values.get(source))
                if shape is not None:
                    perm = transpose_perm(node, len(shape))
                    if perm == list(range(len(shape))):
                        proof = "identity Transpose permutation"
            elif node.op_type in {"Reshape", "Expand", "Squeeze", "Unsqueeze"}:
                before = tensor_shape(values.get(source))
                after = tensor_shape(values.get(node.output[0]))
                if before is not None and before == after:
                    proof = f"identity {node.op_type}: inferred input/output shapes equal"
            elif node.op_type == "Pad" and len(node.input) >= 2:
                pads = arrays.get(node.input[1])
                if pads is not None and np.all(pads == 0):
                    proof = "zero Pad"
            if proof is not None and can_bypass(model, node.output[0], source):
                chosen = (index, source, proof)
                break
        if chosen is None:
            break
        action = bypass_node(model, *chosen)
        if action is None:
            break
        actions.append(action)
        actions.extend(cleanup_dead(model))
    return actions


def neutral_arithmetic(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    while True:
        arrays = constant_arrays(model)
        chosen = None
        for index, node in enumerate(model.graph.node):
            if len(node.input) != 2 or len(node.output) != 1 or not node.output[0]:
                continue
            source = proof = None
            if node.op_type in {"Add", "Mul"}:
                neutral = 0 if node.op_type == "Add" else 1
                for constant_index in (0, 1):
                    array = arrays.get(node.input[constant_index])
                    if array is not None and array.size and np.all(array == neutral):
                        source = node.input[1 - constant_index]
                        proof = f"{node.op_type} neutral {neutral}"
                        break
            elif node.op_type in {"Sub", "Div"}:
                neutral = 0 if node.op_type == "Sub" else 1
                array = arrays.get(node.input[1])
                if array is not None and array.size and np.all(array == neutral):
                    source = node.input[0]
                    proof = f"{node.op_type} right-neutral {neutral}"
            if (
                source is not None and proof is not None
                and can_bypass(model, node.output[0], source)
            ):
                chosen = (index, source, proof)
                break
        if chosen is None:
            break
        action = bypass_node(model, *chosen)
        if action is None:
            break
        actions.append(action)
        actions.extend(cleanup_dead(model))
    return actions


def pair_cancellations(model: onnx.ModelProto) -> list[dict[str, Any]]:
    actions = []
    while True:
        try:
            values = inferred_values(model)
        except Exception:
            break
        producers = {
            output: index
            for index, node in enumerate(model.graph.node)
            for output in node.output if output
        }
        chosen = None
        for outer_index, outer in enumerate(model.graph.node):
            if not outer.input or len(outer.output) != 1 or not outer.output[0]:
                continue
            inner_index = producers.get(outer.input[0])
            if inner_index is None:
                continue
            inner = model.graph.node[inner_index]
            if not inner.input or len(inner.output) != 1:
                continue
            source = inner.input[0]
            before = tensor_shape(values.get(source))
            after = tensor_shape(values.get(outer.output[0]))
            if before is None or before != after:
                continue
            proof = None
            if inner.op_type == outer.op_type == "Transpose":
                first = transpose_perm(inner, len(before))
                second = transpose_perm(outer, len(before))
                if len(first) == len(second) and [first[index] for index in second] == list(range(len(before))):
                    proof = "composed Transpose permutation is identity"
            elif {inner.op_type, outer.op_type} == {"Squeeze", "Unsqueeze"}:
                proof = "adjacent Squeeze/Unsqueeze restore identical inferred shape"
            if proof is not None and can_bypass(model, outer.output[0], source):
                chosen = (outer_index, source, proof)
                break
        if chosen is None:
            break
        action = bypass_node(model, *chosen)
        if action is None:
            break
        actions.append(action)
        actions.extend(cleanup_dead(model))
    return actions


def transform(base: onnx.ModelProto, profile: str) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    model = copy.deepcopy(base)
    actions: list[dict[str, Any]] = []
    if profile == "cleanup":
        actions.extend(cleanup_dead(model))
    elif profile == "dedupe":
        actions.extend(dedupe_initializers(model))
        actions.extend(cleanup_dead(model))
    elif profile == "unary":
        actions.extend(unary_noops(model))
        actions.extend(cleanup_dead(model))
    elif profile == "neutral":
        actions.extend(neutral_arithmetic(model))
        actions.extend(cleanup_dead(model))
    elif profile == "pair_cancel":
        actions.extend(pair_cancellations(model))
        actions.extend(cleanup_dead(model))
    elif profile == "combined":
        actions.extend(cleanup_dead(model))
        actions.extend(dedupe_initializers(model))
        # Repeat because one family can expose another.
        for _ in range(3):
            before = len(actions)
            actions.extend(pair_cancellations(model))
            actions.extend(unary_noops(model))
            actions.extend(neutral_arithmetic(model))
            actions.extend(cleanup_dead(model))
            actions.extend(dedupe_initializers(model))
            if len(actions) == before:
                break
    else:
        raise ValueError(profile)
    remove_dead_value_info(model)
    return model, actions


def extended_structure(task: int, data: bytes) -> dict[str, Any]:
    audit = SCREEN.static_audit(data, [rel(AUTHORITY)], task)
    try:
        model = onnx.load_model_from_string(data)
        arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
        nonfinite = [
            item.name for item, array in zip(model.graph.initializer, arrays)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
        ]
        audit["nonfinite_initializers"] = nonfinite
        if nonfinite:
            audit["reasons"].append("nonfinite_initializer")
        audit["reasons"] = sorted(set(audit["reasons"]))
        audit["pass"] = bool(audit["pass"] and not nonfinite)
    except Exception as exc:  # noqa: BLE001
        audit["pass"] = False
        audit.setdefault("reasons", []).append(f"extended_structure:{type(exc).__name__}")
        audit["extended_structure_error"] = str(exc)
    return audit


def official_profile(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"noop285_{task:03d}_{label}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, work, label=label, require_correct=False
        )


def truthful_trace(task: int, data: bytes) -> dict[str, Any]:
    try:
        result = runtime_shape_trace(task, onnx.load_model_from_string(data))
        result["truthful"] = not result.get("declared_actual_mismatches") and not result.get("error")
        return result
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def sign_difference(left: bytes, right: bytes) -> int:
    xor = np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    )
    return int(np.unpackbits(xor).sum())


def raw_fingerprint(raw: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(raw.dtype).encode())
    digest.update(np.asarray(raw.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(raw).tobytes())
    return digest.hexdigest()


def evaluate_config(
    runtime: ort.InferenceSession,
    cases: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    baseline_signs: list[bytes | None] | None,
    baseline_raw: list[str | None] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = {
        "total": len(cases), "right": 0, "wrong": 0, "errors": 0,
        "nonfinite_cases": 0, "nonfinite_elements": 0,
        "output_shape_mismatches": 0, "small_positive_0_to_0_25": 0,
        "minimum_positive": math.inf, "maximum_nonpositive": -math.inf,
        "sign_config_mismatch_cases": 0, "sign_config_mismatch_cells": 0,
        "raw_config_mismatch_cases": 0,
        "first_wrong": None, "first_error": None, "first_shape": None,
    }
    signs: list[bytes | None] = []
    raw_ids: list[str | None] = []
    correctness: list[bool | None] = []
    for index, (example, location) in enumerate(zip(cases, locations)):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"converter rejected prefiltered case: {location}")
        expected = benchmark["output"] > 0
        try:
            raw = np.asarray(runtime.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            row["errors"] += 1
            signs.append(None); raw_ids.append(None); correctness.append(None)
            if row["first_error"] is None:
                row["first_error"] = {**location, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if tuple(raw.shape) != EXPECTED_IO:
            row["output_shape_mismatches"] += 1
            signs.append(None); raw_ids.append(None); correctness.append(None)
            if row["first_shape"] is None:
                row["first_shape"] = {**location, "actual": list(raw.shape)}
            continue
        finite = np.isfinite(raw)
        count_nonfinite = int(np.count_nonzero(~finite))
        row["nonfinite_cases"] += int(count_nonfinite > 0)
        row["nonfinite_elements"] += count_nonfinite
        positive = raw > 0
        packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
        raw_id = raw_fingerprint(raw)
        signs.append(packed); raw_ids.append(raw_id)
        correct = bool(np.array_equal(positive, expected))
        correctness.append(correct)
        row["right"] += int(correct); row["wrong"] += int(not correct)
        if not correct and row["first_wrong"] is None:
            row["first_wrong"] = {**location, "different_cells": int(np.count_nonzero(positive != expected))}
        if np.any(positive):
            row["minimum_positive"] = min(row["minimum_positive"], float(raw[positive].min()))
            row["small_positive_0_to_0_25"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        nonpositive = finite & ~positive
        if np.any(nonpositive):
            row["maximum_nonpositive"] = max(row["maximum_nonpositive"], float(raw[nonpositive].max()))
        if baseline_signs is not None:
            baseline_sign = baseline_signs[index]
            difference = math.prod(EXPECTED_IO) if baseline_sign is None else sign_difference(packed, baseline_sign)
            row["sign_config_mismatch_cases"] += int(difference > 0)
            row["sign_config_mismatch_cells"] += difference
            row["raw_config_mismatch_cases"] += int(raw_id != baseline_raw[index])
    row["accuracy"] = row["right"] / len(cases)
    row["minimum_positive"] = None if row["minimum_positive"] == math.inf else row["minimum_positive"]
    row["maximum_nonpositive"] = None if row["maximum_nonpositive"] == -math.inf else row["maximum_nonpositive"]
    return row, {"signs": signs, "raw": raw_ids, "correctness": correctness}


def evaluate_four(data: bytes, cases: list[dict[str, Any]], locations: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = {}
    aux = {}
    baseline_signs = baseline_raw = None
    for name, disable_all, threads in CONFIGS:
        try:
            runtime = make_session(data, disable_all, threads)
            row, detail = evaluate_config(
                runtime, cases, locations,
                None if name == BASE_CONFIG else baseline_signs,
                None if name == BASE_CONFIG else baseline_raw,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "total": len(cases), "right": 0, "wrong": 0, "accuracy": 0.0,
                "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}",
                "nonfinite_cases": 0, "nonfinite_elements": 0,
                "output_shape_mismatches": 0, "small_positive_0_to_0_25": 0,
                "sign_config_mismatch_cases": 0, "sign_config_mismatch_cells": 0,
                "raw_config_mismatch_cases": 0,
            }
            detail = {"signs": [None] * len(cases), "raw": [None] * len(cases), "correctness": [None] * len(cases)}
        row["disable_all"] = disable_all; row["threads"] = threads
        rows[name] = row; aux[name] = detail
        if name == BASE_CONFIG:
            baseline_signs = detail["signs"]; baseline_raw = detail["raw"]
    return rows, aux


def compare_models(candidate: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for name, _disable, _threads in CONFIGS:
        cand = candidate[name]; auth = authority[name]
        row = {
            "total_comparable": 0, "raw_equal_cases": 0, "sign_equal_cases": 0,
            "sign_mismatch_cells": 0, "both_correct": 0,
            "authority_correct_candidate_wrong": 0,
            "candidate_correct_authority_wrong": 0, "both_wrong": 0,
        }
        for c_raw, c_sign, c_right, a_raw, a_sign, a_right in zip(
            cand["raw"], cand["signs"], cand["correctness"],
            auth["raw"], auth["signs"], auth["correctness"],
        ):
            if None in (c_raw, c_sign, c_right, a_raw, a_sign, a_right):
                continue
            row["total_comparable"] += 1
            row["raw_equal_cases"] += int(c_raw == a_raw)
            difference = sign_difference(c_sign, a_sign)
            row["sign_equal_cases"] += int(difference == 0)
            row["sign_mismatch_cells"] += difference
            if c_right and a_right: row["both_correct"] += 1
            elif a_right: row["authority_correct_candidate_wrong"] += 1
            elif c_right: row["candidate_correct_authority_wrong"] += 1
            else: row["both_wrong"] += 1
        output[name] = row
    return output


def runtime_clean(row: dict[str, Any]) -> bool:
    return bool(
        row.get("errors") == 0 and not row.get("session_error")
        and row.get("nonfinite_cases") == 0 and row.get("nonfinite_elements") == 0
        and row.get("output_shape_mismatches") == 0
        and row.get("small_positive_0_to_0_25") == 0
    )


def config_stable(row: dict[str, Any]) -> bool:
    return bool(
        row.get("sign_config_mismatch_cases") == 0
        and row.get("sign_config_mismatch_cells") == 0
        and row.get("raw_config_mismatch_cases") == 0
    )


def exact_comparison(rows: dict[str, Any], total: int) -> bool:
    return all(
        row["total_comparable"] == total
        and row["raw_equal_cases"] == total
        and row["sign_equal_cases"] == total
        and row["sign_mismatch_cells"] == 0
        for row in rows.values()
    )


def known_cases(task: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples = scoring.load_examples(task)
    cases = []; locations = []
    meta = {"raw": {}, "converted": {}, "skipped": {}}
    for subset in ("train", "test", "arc-gen"):
        meta["raw"][subset] = len(examples[subset]); meta["converted"][subset] = 0; meta["skipped"][subset] = 0
        for index, example in enumerate(examples[subset]):
            if scoring.convert_to_numpy(example) is None:
                meta["skipped"][subset] += 1
                continue
            meta["converted"][subset] += 1
            cases.append(example)
            locations.append({"subset": subset, "index": index, "case_id": case_id(example)})
    meta["converted_total"] = len(cases)
    return cases, locations, meta


def fresh_cases(task: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    module_name = f"task_{task_map[f'{task:03d}']}"
    generator = importlib.import_module(module_name)
    common = importlib.import_module("common")
    random.seed(seed); np.random.seed(seed & 0xFFFFFFFF); common.random.seed(seed)
    cases = []; locations = []; attempts = generation_errors = conversion_skips = 0
    stream_digest = hashlib.sha256(); seen = set()
    while len(cases) < FRESH_PER_SEED:
        attempts += 1
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if scoring.convert_to_numpy(example) is None:
            conversion_skips += 1
            continue
        identifier = case_id(example); stream_digest.update(bytes.fromhex(identifier)); seen.add(identifier)
        locations.append({"seed": seed, "index": len(cases), "case_id": identifier})
        cases.append(example)
    return cases, locations, {
        "seed": seed, "generator_module": module_name, "attempts": attempts,
        "accepted": len(cases), "generation_errors": generation_errors,
        "conversion_skips": conversion_skips, "unique_case_ids": len(seen),
        "case_stream_sha256": stream_digest.hexdigest(),
    }


def main() -> int:
    started = time.monotonic()
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority changed")
    if len(ASSIGNED_ACTIVE22) != 22:
        raise RuntimeError("assigned active22 snapshot is malformed")
    active_manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    observed_active = {int(item["task"]) for item in active_manifest["active_candidates"]}
    concurrent_active = observed_active - ASSIGNED_ACTIVE22
    active = ASSIGNED_ACTIVE22 | observed_active
    canonical = json.loads(CANONICAL_COSTS.read_text(encoding="utf-8"))
    costs = {int(item["task"]): int(item["cost"]) for item in canonical["ranked"]}
    if len(costs) != 400 or canonical.get("authority_zip") != "submission_base_8009.46.zip":
        raise RuntimeError("canonical costs are not the 400-task 8009.46 census")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = {task: archive.read(f"task{task:03d}.onnx") for task in range(1, 401)}

    ordered_tasks = sorted(
        range(1, 401),
        key=lambda task: (not PRIORITY_MIN <= costs[task] <= PRIORITY_MAX, costs[task], task),
    )
    task_rows = []
    variant_rows = []
    strict_actual: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for ordinal, task in enumerate(ordered_tasks, start=1):
        base_data = authority_data[task]
        base = onnx.load_model_from_string(base_data)
        task_row = {
            "task": task, "authority_sha256": sha256(base_data),
            "authority_cost": costs[task],
            "priority_cost_150_to_500": PRIORITY_MIN <= costs[task] <= PRIORITY_MAX,
            "active_excluded": task in active,
            "assigned_active22_excluded": task in ASSIGNED_ACTIVE22,
            "concurrent_active_excluded": task in concurrent_active,
            "private_route_excluded": task in PRIVATE_ZERO_CATALOG,
            "profiles_with_actions": 0, "unique_variants": 0,
            "strict_actual_variants": 0,
        }
        if task in active:
            task_row["classification"] = (
                "EXCLUDED_ACTIVE22" if task in ASSIGNED_ACTIVE22
                else "EXCLUDED_CONCURRENT_ACTIVE"
            )
            task_rows.append(task_row); continue
        if task in PRIVATE_ZERO_CATALOG:
            task_row["classification"] = "EXCLUDED_PRIVATE_OR_UNSOUND_ROUTE"
            task_rows.append(task_row); continue
        authority_structure = extended_structure(task, base_data)
        task_row["authority_structure"] = authority_structure
        if not authority_structure["pass"]:
            task_row["classification"] = "EXCLUDED_AUTHORITY_LOOKUP_GIANT_OR_STRUCTURE"
            task_rows.append(task_row); continue
        for profile in PROFILES:
            model, actions = transform(base, profile)
            if not actions:
                continue
            task_row["profiles_with_actions"] += 1
            data = model.SerializeToString()
            digest = sha256(data)
            if digest == sha256(base_data) or (task, digest) in seen:
                continue
            seen.add((task, digest)); task_row["unique_variants"] += 1
            row = {
                "task": task, "profile": profile, "sha256": digest,
                "authority_sha256": sha256(base_data), "authority_cost": costs[task],
                "priority_cost_150_to_500": task_row["priority_cost_150_to_500"],
                "actions": actions,
            }
            structure = extended_structure(task, data)
            row["structure"] = structure
            if not structure["pass"]:
                row["classification"] = "REJECT_CANDIDATE_STRUCTURE"
                variant_rows.append(row); continue
            profile_result = official_profile(task, data, f"{profile}_{digest[:8]}")
            row["official_profile"] = profile_result
            if profile_result is None or int(profile_result["cost"]) >= costs[task]:
                row["classification"] = "REJECT_NOT_STRICT_LOWER_ACTUAL"
                variant_rows.append(row); continue
            trace = truthful_trace(task, data)
            row["runtime_shape_trace"] = trace
            if not trace.get("truthful"):
                row["classification"] = "REJECT_RUNTIME_SHAPE_CLOAK"
                variant_rows.append(row); continue
            row["classification"] = "STRICT_ACTUAL_TRUTHFUL_PRE_KNOWN"
            row["projected_gain"] = math.log(costs[task] / int(profile_result["cost"]))
            row["_data"] = data
            strict_actual[task].append(row)
            task_row["strict_actual_variants"] += 1
            variant_rows.append(row)
            print(json.dumps({
                "strict_actual": task, "profile": profile, "cost": profile_result["cost"],
                "authority": costs[task], "sha": digest[:12],
            }), flush=True)
        task_row["classification"] = "SCANNED"
        task_rows.append(task_row)
        if ordinal % 50 == 0:
            print(json.dumps({"task_scan_progress": ordinal, "of": 400}), flush=True)

    # Test candidates cheapest-first; a task reports only its cheapest complete pass.
    known_cache = {}
    finalists = []
    for task in sorted(strict_actual):
        candidates = sorted(
            strict_actual[task],
            key=lambda row: (int(row["official_profile"]["cost"]), row["sha256"]),
        )
        if task not in known_cache:
            known_cache[task] = known_cases(task)
        cases, locations, known_meta = known_cache[task]
        authority_four, authority_aux = evaluate_four(authority_data[task], cases, locations)
        for candidate in candidates:
            candidate_four, candidate_aux = evaluate_four(candidate["_data"], cases, locations)
            comparison = compare_models(candidate_aux, authority_aux)
            candidate["known"] = {
                "case_meta": known_meta,
                "candidate": candidate_four,
                "authority": authority_four,
                "candidate_vs_authority": comparison,
            }
            total = len(cases)
            pass_known = bool(
                exact_comparison(comparison, total)
                and all(runtime_clean(row) and config_stable(row) for row in candidate_four.values())
                and all(runtime_clean(row) and config_stable(row) for row in authority_four.values())
                and all(row["accuracy"] >= 0.90 for row in candidate_four.values())
            )
            candidate["known_raw_sign_exact_all_four"] = pass_known
            if not pass_known:
                candidate["classification"] = "REJECT_KNOWN_RAW_SIGN_OR_RUNTIME"
                continue
            candidate["classification"] = "KNOWN_RAW_EXACT_PRE_FRESH"
            # Fresh is conditional and uses the exact generator mapping.
            seed_rows = []
            for seed in (285_000_000 + task, 285_100_000 + task):
                fresh, fresh_locations, generation = fresh_cases(task, seed)
                cand_four, cand_aux = evaluate_four(candidate["_data"], fresh, fresh_locations)
                auth_four, auth_aux = evaluate_four(authority_data[task], fresh, fresh_locations)
                fresh_comparison = compare_models(cand_aux, auth_aux)
                seed_rows.append({
                    "seed": seed, "generation": generation,
                    "candidate": cand_four, "authority": auth_four,
                    "candidate_vs_authority": fresh_comparison,
                })
                print(json.dumps({
                    "fresh_task": task, "seed": seed,
                    "candidate_right": {name: row["right"] for name, row in cand_four.items()},
                    "raw_equal": {name: row["raw_equal_cases"] for name, row in fresh_comparison.items()},
                }), flush=True)
            pass_fresh = all(
                exact_comparison(seed_row["candidate_vs_authority"], FRESH_PER_SEED)
                and all(runtime_clean(row) and config_stable(row) and row["accuracy"] >= 0.90 for row in seed_row["candidate"].values())
                and all(runtime_clean(row) and config_stable(row) for row in seed_row["authority"].values())
                and seed_row["generation"]["accepted"] == FRESH_PER_SEED
                and seed_row["generation"]["generation_errors"] == 0
                and seed_row["generation"]["conversion_skips"] == 0
                for seed_row in seed_rows
            )
            candidate["fresh"] = {
                "count_per_seed": FRESH_PER_SEED,
                "seeds": [item["seed"] for item in seed_rows],
                "runs": seed_rows,
                "raw_sign_exact_truth90_runtime_clean": pass_fresh,
            }
            if not pass_fresh:
                candidate["classification"] = "REJECT_FRESH_RAW_SIGN_TRUTH_OR_RUNTIME"
                continue
            candidate["classification"] = "PASS_EXACT_NOOP"
            candidate["same_task_cheapest_complete_pass"] = True
            path = CANDIDATE_DIR / (
                f"task{task:03d}_{candidate['profile']}_cost{candidate['official_profile']['cost']}_"
                f"{candidate['sha256'][:12]}.onnx"
            )
            path.write_bytes(candidate["_data"])
            candidate["path"] = rel(path)
            finalists.append(candidate)
            break

    for row in variant_rows:
        row.pop("_data", None)
    classifications = Counter(row["classification"] for row in variant_rows)
    priority_tasks = [row for row in task_rows if row["priority_cost_150_to_500"]]
    payload = {
        "lane": "agent_exact_noop_scan_285",
        "decision": "PASS_CANDIDATES_FOUND" if finalists else "NO_EXACT_NOOP_WINNER",
        "authority": {
            "zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256,
            "canonical_costs": rel(CANONICAL_COSTS), "models": 400,
        },
        "active_exclusion": {
            "manifest": rel(ACTIVE_MANIFEST),
            "assigned_active22_count": len(ASSIGNED_ACTIVE22),
            "assigned_active22_tasks": sorted(ASSIGNED_ACTIVE22),
            "observed_manifest_count_at_run": len(observed_active),
            "observed_manifest_tasks_at_run": sorted(observed_active),
            "concurrent_additions_excluded": sorted(concurrent_active),
            "total_safety_exclusion_count": len(active),
        },
        "policy": {
            "profiles": list(PROFILES),
            "priority_cost_range": [PRIORITY_MIN, PRIORITY_MAX],
            "fresh_count_per_seed": FRESH_PER_SEED,
            "lookup_giant_private_routes_allowed": False,
            "automatic_promotion": False,
        },
        "summary": {
            "authority_models": 400,
            "assigned_active22_excluded": sum(row["assigned_active22_excluded"] for row in task_rows),
            "concurrent_active_excluded": sum(row["concurrent_active_excluded"] for row in task_rows),
            "total_active_safety_excluded": sum(row["active_excluded"] for row in task_rows),
            "private_route_excluded": sum(row["private_route_excluded"] and not row["active_excluded"] for row in task_rows),
            "priority_tasks_total": len(priority_tasks),
            "priority_tasks_scanned_nonexcluded": sum(row["classification"] == "SCANNED" for row in priority_tasks),
            "profiles_with_actions": sum(row["profiles_with_actions"] for row in task_rows),
            "unique_variants": len(variant_rows),
            "strict_actual_truthful_pre_known": sum(row["classification"] in {"STRICT_ACTUAL_TRUTHFUL_PRE_KNOWN", "KNOWN_RAW_EXACT_PRE_FRESH", "PASS_EXACT_NOOP", "REJECT_KNOWN_RAW_SIGN_OR_RUNTIME", "REJECT_FRESH_RAW_SIGN_TRUTH_OR_RUNTIME"} for row in variant_rows),
            "known_raw_exact_pre_fresh": sum(row.get("known_raw_sign_exact_all_four", False) for row in variant_rows),
            "finalists": len(finalists),
            "finalist_tasks": [int(row["task"]) for row in finalists],
            "classification_counts": dict(classifications),
        },
        "task_inventory": sorted(task_rows, key=lambda row: row["task"]),
        "variants": sorted(variant_rows, key=lambda row: (row["task"], row["profile"], row["sha256"])),
        "winners": [
            {key: value for key, value in row.items() if key != "_data"}
            for row in sorted(finalists, key=lambda row: row["task"])
        ],
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only scripts/golf/agent_exact_noop_scan_285",
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": payload["decision"], "summary": payload["summary"],
        "elapsed_seconds": payload["elapsed_seconds"], "evidence": rel(EVIDENCE),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

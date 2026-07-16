#!/usr/bin/env python3
"""Exact residual scan for Tile, Expand, and OneHot in all 400 authorities.

Only a rewrite with a strictly lower official known-corpus profile may become
a candidate.  Equal-cost controls and positive-cost algebraic equivalents are
recorded as census evidence but are not emitted as ONNX candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
TARGET_OPS = {"Tile", "Expand", "OneHot"}
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def dtype_name(value: onnx.ValueInfoProto) -> str:
    return TensorProto.DataType.Name(value.type.tensor_type.elem_type)


def value_maps(model: onnx.ModelProto) -> tuple[dict[str, list[int | None]], dict[str, str]]:
    shapes: dict[str, list[int | None]] = {}
    dtypes: dict[str, str] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        shapes[value.name] = dims(value)
        dtypes[value.name] = dtype_name(value)
    for item in model.graph.initializer:
        shapes[item.name] = [int(value) for value in item.dims]
        dtypes[item.name] = TensorProto.DataType.Name(item.data_type)
    return shapes, dtypes


def product(shape: tuple[int, ...] | list[int]) -> int:
    return int(math.prod(shape))


def known_cases(task: int) -> list[dict[str, Any]]:
    examples = scoring.load_examples(task)
    return [
        example
        for subset in ("train", "test", "arc-gen")
        for example in examples[subset]
    ]


def profile_known(
    model: onnx.ModelProto, task: int
) -> tuple[dict[str, Any], dict[str, set[tuple[int, ...]]]]:
    """Official-style known profile plus every observed node-output shape."""
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError(f"task{task:03d}: sanitize_model rejected authority")
    with tempfile.TemporaryDirectory(prefix=f"tile_expand_onehot_{task:03d}_", dir=HERE) as workdir:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.profile_file_prefix = str(Path(workdir) / "known")
        session = ort.InferenceSession(sanitized.SerializeToString(), options)
        right = wrong = errors = nonfinite = shape_errors = 0
        output_shapes: set[tuple[int, ...]] = set()
        for example in known_cases(task):
            converted = scoring.convert_to_numpy(example)
            if converted is None:
                errors += 1
                continue
            try:
                raw = session.run(["output"], {"input": converted["input"]})[0]
                output_shapes.add(tuple(int(value) for value in raw.shape))
                if tuple(raw.shape) != (1, 10, 30, 30):
                    shape_errors += 1
                if not np.isfinite(raw).all():
                    nonfinite += 1
                if np.array_equal(raw > 0.0, converted["output"] > 0.0):
                    right += 1
                else:
                    wrong += 1
            except Exception:
                errors += 1
        trace_path = session.end_profiling()
        trace = json.loads(Path(trace_path).read_text())
        memory, params = scoring.score_network(sanitized, trace_path)
        if memory is None or params is None:
            raise RuntimeError(f"task{task:03d}: official profile unscorable")

    node_by_name = {node.name: node for node in sanitized.graph.node}
    observed_safe: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for event in trace:
        if event.get("cat") != "Node":
            continue
        node_name = event.get("name", "").replace("_kernel_time", "")
        node = node_by_name.get(node_name)
        output_type_shape = event.get("args", {}).get("output_type_shape")
        if node is None or not output_type_shape:
            continue
        for index, item in enumerate(output_type_shape):
            if index >= len(node.output) or not item:
                continue
            shape = tuple(int(value) for value in next(iter(item.values())))
            observed_safe[node.output[index]].add(shape)
    safe_to_original = {
        safe_output: original_output
        for original_node, safe_node in zip(model.graph.node, sanitized.graph.node, strict=True)
        for original_output, safe_output in zip(original_node.output, safe_node.output, strict=True)
    }
    observed: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for safe_name, runtime_shapes in observed_safe.items():
        original_name = safe_to_original.get(safe_name)
        if original_name is not None:
            observed[original_name].update(runtime_shapes)
    return (
        {
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "nonfinite": nonfinite,
            "output_shape_errors": shape_errors,
            "total": len(known_cases(task)),
            "output_shapes": [list(shape) for shape in sorted(output_shapes)],
            "memory": int(memory),
            "params": int(params),
            "cost": int(memory) + int(params),
        },
        observed,
    )


def initializer_users(model: onnx.ModelProto) -> Counter[str]:
    names = {item.name for item in model.graph.initializer}
    return Counter(name for node in model.graph.node for name in node.input if name in names)


def array_record(array: np.ndarray) -> dict[str, Any]:
    flat = array.reshape(-1)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "size": int(array.size),
        "values": flat.tolist() if flat.size <= 64 else flat[:64].tolist(),
        "values_truncated": bool(flat.size > 64),
    }


def exact_initializer_names(arrays: dict[str, np.ndarray], target: np.ndarray) -> list[str]:
    return sorted(
        name
        for name, array in arrays.items()
        if array.dtype == target.dtype
        and array.shape == target.shape
        and np.array_equal(array, target)
    )


def identity_basis_names(arrays: dict[str, np.ndarray], depth: int) -> list[str]:
    result = []
    for name, array in arrays.items():
        squeezed = np.squeeze(array)
        if squeezed.shape != (depth, depth):
            continue
        try:
            numeric = squeezed.astype(np.float64)
        except (TypeError, ValueError):
            continue
        if np.array_equal(numeric, np.eye(depth, dtype=np.float64)):
            result.append(name)
    return sorted(result)


def consumers(model: onnx.ModelProto, value_name: str) -> list[dict[str, Any]]:
    result = []
    for index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name == value_name:
                result.append(
                    {"node_index": index, "op": node.op_type, "input_index": input_index}
                )
    return result


def runtime_input_shapes(
    model: onnx.ModelProto,
    sanitized: onnx.ModelProto,
    node_index: int,
    input_index: int,
    observed: dict[str, set[tuple[int, ...]]],
) -> list[list[int]]:
    source = model.graph.node[node_index].input[input_index]
    produced = {name for node in model.graph.node for name in node.output}
    if source in produced:
        return [list(shape) for shape in sorted(observed.get(source, set()))]
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if source in arrays:
        return [list(arrays[source].shape)]
    return []


def analyze_tile(
    model: onnx.ModelProto,
    sanitized: onnx.ModelProto,
    node_index: int,
    arrays: dict[str, np.ndarray],
    users: Counter[str],
    observed: dict[str, set[tuple[int, ...]]],
) -> dict[str, Any]:
    node = model.graph.node[node_index]
    repeats = arrays.get(node.input[1])
    input_shapes = runtime_input_shapes(model, sanitized, node_index, 0, observed)
    output_name = model.graph.node[node_index].output[0]
    output_shapes = [list(shape) for shape in sorted(observed.get(output_name, set()))]
    if repeats is None:
        return {"family": "Tile", "decision": "reject_dynamic_repeats"}
    repeats_list = [int(value) for value in repeats.reshape(-1)]
    all_ones = all(value == 1 for value in repeats_list)
    singleton_repeat_axes = bool(input_shapes)
    for shape in input_shapes:
        if len(shape) != len(repeats_list):
            singleton_repeat_axes = False
            break
        for dimension, repeat in zip(shape, repeats_list, strict=True):
            if repeat != 1 and dimension != 1:
                singleton_repeat_axes = False
                break
    repeat_param_removable = users[node.input[1]] == 1
    duplicate_shape_names = [
        name
        for name in exact_initializer_names(arrays, repeats)
        if name != node.input[1]
    ]
    if all_ones:
        output_elements = max((product(shape) for shape in output_shapes), default=0)
        return {
            "family": "Tile",
            "repeats": repeats_list,
            "runtime_input_shapes": input_shapes,
            "runtime_output_shapes": output_shapes,
            "all_repeats_one": True,
            "singleton_repeat_axes": True,
            "rewrite": "direct_alias_or_Identity",
            "strict_lower_possible": True,
            "optimistic_memory_delta": -output_elements,
            "optimistic_param_delta": -int(repeats.size) if repeat_param_removable else 0,
        }
    if singleton_repeat_axes:
        return {
            "family": "Tile",
            "repeats": repeats_list,
            "runtime_input_shapes": input_shapes,
            "runtime_output_shapes": output_shapes,
            "all_repeats_one": False,
            "singleton_repeat_axes": True,
            "rewrite": "Expand",
            "strict_lower_possible": bool(duplicate_shape_names),
            "duplicate_shape_initializers": duplicate_shape_names,
            "memory_delta": 0,
            "param_delta": -int(repeats.size) if duplicate_shape_names and repeat_param_removable else 0,
            "reason": "Tile and Expand materialize the same output; only an already-live identical shape can lower params.",
        }
    return {
        "family": "Tile",
        "repeats": repeats_list,
        "runtime_input_shapes": input_shapes,
        "runtime_output_shapes": output_shapes,
        "all_repeats_one": False,
        "singleton_repeat_axes": False,
        "rewrite": None,
        "strict_lower_possible": False,
        "reason": (
            "At least one repeated runtime axis is non-singleton. Expand broadcasts each element "
            "and is not equal to Tile's whole-axis repetition; reshape/expand/reshape adds activations."
        ),
    }


def analyze_expand(
    model: onnx.ModelProto,
    sanitized: onnx.ModelProto,
    node_index: int,
    arrays: dict[str, np.ndarray],
    users: Counter[str],
    observed: dict[str, set[tuple[int, ...]]],
) -> dict[str, Any]:
    node = model.graph.node[node_index]
    shape_array = arrays.get(node.input[1])
    input_shapes = runtime_input_shapes(model, sanitized, node_index, 0, observed)
    output_name = model.graph.node[node_index].output[0]
    output_shapes = [list(shape) for shape in sorted(observed.get(output_name, set()))]
    duplicate_shape_names = []
    if shape_array is not None:
        duplicate_shape_names = [
            name
            for name in exact_initializer_names(arrays, shape_array)
            if name != node.input[1]
        ]
    identity = input_shapes == output_shapes and bool(input_shapes)
    consumer_rows = consumers(model, node.output[0])
    concat_consumer = any(row["op"] == "Concat" for row in consumer_rows)
    return {
        "family": "Expand",
        "shape": array_record(shape_array) if shape_array is not None else None,
        "runtime_input_shapes": input_shapes,
        "runtime_output_shapes": output_shapes,
        "identity": identity,
        "consumers": consumer_rows,
        "duplicate_shape_initializers": duplicate_shape_names,
        "shape_initializer_unique": bool(shape_array is not None and users[node.input[1]] == 1),
        "strict_lower_possible": bool(identity or duplicate_shape_names),
        "rewrite": "direct_alias" if identity else None,
        "reason": (
            "The sole Concat consumer does not broadcast. The expanded color-coordinate tensor "
            "must have the same non-concat dimensions as the row/column tensors. A producer-side "
            "broadcast enlarges its input activation/initializer and is not cheaper."
            if concat_consumer and not identity and not duplicate_shape_names
            else "No exact lower-cost residual form found."
        ),
    }


def np_dtype_for_tensor_type(elem_type: int) -> np.dtype:
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))


def analyze_onehot(
    model: onnx.ModelProto,
    sanitized: onnx.ModelProto,
    inferred: onnx.ModelProto,
    node_index: int,
    arrays: dict[str, np.ndarray],
    users: Counter[str],
    observed: dict[str, set[tuple[int, ...]]],
) -> dict[str, Any]:
    node = model.graph.node[node_index]
    depth_array = arrays.get(node.input[1])
    values_array = arrays.get(node.input[2])
    if depth_array is None or values_array is None:
        return {"family": "OneHot", "decision": "reject_dynamic_depth_or_values"}
    depth = int(depth_array.reshape(-1)[0])
    axis = next(
        (int(onnx.helper.get_attribute_value(attr)) for attr in node.attribute if attr.name == "axis"),
        -1,
    )
    output_name = model.graph.node[node_index].output[0]
    output_shapes = [tuple(shape) for shape in observed.get(output_name, set())]
    output_elements = max((product(shape) for shape in output_shapes), default=0)
    inferred_shapes, inferred_dtypes = value_maps(inferred)
    index_dtype_name = inferred_dtypes.get(node.input[0])
    index_value = next(
        (
            value
            for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
            if value.name == node.input[0]
        ),
        None,
    )
    if index_value is None:
        raise RuntimeError(f"missing inferred OneHot index type: {node.input[0]}")
    index_dtype = np_dtype_for_tensor_type(index_value.type.tensor_type.elem_type)
    if not output_shapes:
        raise RuntimeError("OneHot output was not observed")
    output_rank = len(output_shapes[0])
    normalized_axis = axis if axis >= 0 else output_rank + axis
    range_shape = [1] * output_rank
    range_shape[normalized_axis] = depth
    range_basis = np.arange(depth, dtype=index_dtype).reshape(range_shape)
    existing_range = exact_initializer_names(arrays, range_basis)
    existing_identity = identity_basis_names(arrays, depth)
    value_dtype_size = int(values_array.dtype.itemsize)
    output_consumers = consumers(model, node.output[0])
    numeric_ops = {"Einsum", "Conv", "RoiAlign"}
    numeric_required = any(row["op"] in numeric_ops for row in output_consumers)
    values = values_array.reshape(-1)
    zero_one = bool(values.size == 2 and values[0] == 0 and values[1] == 1)
    removable_params = 0
    for name in (node.input[1], node.input[2]):
        if users[name] == 1:
            removable_params += int(arrays[name].size)
    range_params = 0 if existing_range else depth
    mapping_params = 0 if zero_one else 2
    # Optimistic bound: broadcasting the index directly against the shaped range
    # requires no Unsqueeze. Numeric consumers still require a mapped tensor of
    # the original dtype/shape, so Equal's bool output is pure added memory.
    optimistic_memory_delta = output_elements if numeric_required else 0
    optimistic_param_delta = range_params + mapping_params - removable_params
    lower_bound_delta = optimistic_memory_delta + optimistic_param_delta
    return {
        "family": "OneHot",
        "depth": depth,
        "axis": axis,
        "normalized_axis": normalized_axis,
        "indices_inferred_shape": inferred_shapes.get(node.input[0]),
        "indices_dtype": index_dtype_name,
        "runtime_output_shapes": [list(shape) for shape in sorted(output_shapes)],
        "output_elements": output_elements,
        "output_value_dtype": str(values_array.dtype),
        "output_value_bytes": value_dtype_size,
        "values": values.tolist(),
        "consumers": output_consumers,
        "numeric_output_required": numeric_required,
        "existing_exact_range_basis": existing_range,
        "existing_identity_basis": existing_identity,
        "unique_removable_depth_values_params": removable_params,
        "equal_to_range_optimistic_memory_delta": optimistic_memory_delta,
        "equal_to_range_optimistic_param_delta": optimistic_param_delta,
        "equal_to_range_optimistic_cost_delta": lower_bound_delta,
        "strict_lower_possible": lower_bound_delta < 0,
        "rewrite": None,
        "reason": (
            "All consumers require the numeric OneHot tensor. Equal-to-range therefore retains "
            "an equally-sized numeric Cast/Where output and adds a bool selector. No exact range "
            "or identity basis initializer is already live."
        ),
    }


ABSORPTION_NOTES = {
    66: (
        "The selector is shared by two Einsums. Replacing it with selected row/column profiles "
        "requires at least two 30-float intermediates versus one shared 30-float selector."
    ),
    200: (
        "OneHot feeds a padded Conv. Gathering precomputed responses needs a 10x2x30 table (600 "
        "parameters); coordinate/parity construction adds a 30-element arithmetic intermediate."
    ),
    247: (
        "RoiAlign simultaneously selects up to three colors and realizes the variable height/width "
        "as the free graph output. Externalizing the selector needs an intermediate color mask and "
        "a separate spatial realization, so it cannot undercut the fused path."
    ),
    300: (
        "The selector is a numeric factor of an Einsum and is also cast to dynamic weights. No "
        "identity/range basis exists; Equal must be cast back to float16 for the Einsum."
    ),
}


def structural(
    model: onnx.ModelProto,
    inferred: onnx.ModelProto,
    observed: dict[str, set[tuple[int, ...]]],
) -> dict[str, Any]:
    checker = True
    checker_error = None
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    except Exception as exc:
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    shape_map, _ = value_maps(inferred)
    mismatches = []
    for name, runtime_shapes in sorted(observed.items()):
        declared = shape_map.get(name)
        if declared is None or any(list(shape) != declared for shape in runtime_shapes):
            mismatches.append(
                {
                    "value": name,
                    "declared_or_inferred": declared,
                    "runtime": [list(shape) for shape in sorted(runtime_shapes)],
                }
            )
    ops = Counter(node.op_type for node in model.graph.node)
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    nested = [
        node.op_type
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
    ]
    return {
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_shape_inference_data_prop": True,
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "nested_graph_ops": nested,
        "banned_ops": sorted(
            {
                node.op_type
                for node in model.graph.node
                if node.op_type in BANNED or "Sequence" in node.op_type
            }
        ),
        "conv_bias_findings": check_conv_bias(model),
        "op_histogram": dict(sorted(ops.items())),
        "runtime_shape_mismatch_count": len(mismatches),
        "runtime_shapes_truthful": not mismatches,
        "runtime_shape_mismatches": mismatches,
    }


def main() -> None:
    authority_bytes = AUTHORITY.read_bytes()
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        census = Counter()
        occurrences: list[tuple[int, int, str]] = []
        member_bytes: dict[int, bytes] = {}
        for name in members:
            task = int(Path(name).stem.removeprefix("task"))
            data = archive.read(name)
            member_bytes[task] = data
            model = onnx.load_model_from_string(data)
            for index, node in enumerate(model.graph.node):
                if node.op_type in TARGET_OPS:
                    census[node.op_type] += 1
                    occurrences.append((task, index, node.op_type))

    target_tasks = sorted({task for task, _, _ in occurrences})
    rows: dict[str, Any] = {}
    all_rewrites: list[dict[str, Any]] = []
    for task in target_tasks:
        model = onnx.load_model_from_string(member_bytes[task])
        try:
            inferred = onnx.shape_inference.infer_shapes(
                copy.deepcopy(model), strict_mode=True, data_prop=True
            )
        except Exception as exc:
            raise RuntimeError(f"task{task:03d}: strict data_prop failed: {exc}") from exc
        known, observed = profile_known(model, task)
        sanitized = scoring.sanitize_model(copy.deepcopy(model))
        if sanitized is None:
            raise RuntimeError(f"task{task:03d}: sanitize failed")
        arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        users = initializer_users(model)
        special_rows = []
        for index, node in enumerate(model.graph.node):
            if node.op_type not in TARGET_OPS:
                continue
            if node.op_type == "Tile":
                analysis = analyze_tile(model, sanitized, index, arrays, users, observed)
            elif node.op_type == "Expand":
                analysis = analyze_expand(model, sanitized, index, arrays, users, observed)
            else:
                analysis = analyze_onehot(
                    model, sanitized, inferred, index, arrays, users, observed
                )
                analysis["downstream_absorption"] = ABSORPTION_NOTES[task]
            item = {
                "task": task,
                "node_index": index,
                "op": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                **analysis,
            }
            special_rows.append(item)
            all_rewrites.append(item)
        rows[str(task)] = {
            "task": task,
            "authority_member": f"task{task:03d}.onnx",
            "sha256": sha256(member_bytes[task]),
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "known_official_profile": known,
            "structural": structural(model, inferred, observed),
            "special_nodes": special_rows,
        }

    lower = [row for row in all_rewrites if row.get("strict_lower_possible")]
    payload = {
        "scan": "root_tile_expand_onehot_scan_261",
        "authority": {
            "path": "submission.zip",
            "sha256": sha256(authority_bytes),
            "onnx_member_count": len(members),
        },
        "census": dict(sorted(census.items())),
        "occurrence_count": len(occurrences),
        "target_tasks": target_tasks,
        "rows": rows,
        "rewrite_summary": {
            "tile_all_repeats_one": sum(
                row.get("all_repeats_one") is True for row in all_rewrites if row["op"] == "Tile"
            ),
            "tile_singleton_repeat_axes": sum(
                row.get("singleton_repeat_axes") is True
                for row in all_rewrites
                if row["op"] == "Tile" and not row.get("all_repeats_one")
            ),
            "expand_identity_or_shared_shape": sum(
                bool(row.get("identity") or row.get("duplicate_shape_initializers"))
                for row in all_rewrites
                if row["op"] == "Expand"
            ),
            "onehot_equal_lower_bound_negative": sum(
                row.get("equal_to_range_optimistic_cost_delta", 0) < 0
                for row in all_rewrites
                if row["op"] == "OneHot"
            ),
        },
        "lower_survivors": lower,
        "winner": None,
        "candidate_files": [],
        "survivor_gate": {
            "known_four_configs_raw": False,
            "fresh_two_seeds_x1000": False,
            "skip_reason": (
                "No algebraically exact rewrite has a negative official cost delta before candidate creation."
            ),
        },
        "policy": {
            "approximation": False,
            "lookup": False,
            "new_shape_cloak": False,
            "root_or_other_paths_modified": False,
        },
    }
    if lower:
        raise RuntimeError(f"unhandled strict-lower survivors require candidate gates: {lower}")
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "authority_sha256": payload["authority"]["sha256"],
                "members": len(members),
                "census": payload["census"],
                "target_tasks": target_tasks,
                "rewrite_summary": payload["rewrite_summary"],
                "lower_survivors": len(lower),
                "output": str((HERE / "scan.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

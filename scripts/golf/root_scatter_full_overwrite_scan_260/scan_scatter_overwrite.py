#!/usr/bin/env python3
"""All-400 exact full-overwrite ScatterND/ScatterElements scan.

The rewrite is deliberately finite and proof-driven:

* ScatterND indices are normalized and expanded into the complete static
  prefix-block coordinate map.  Only duplicate-free, gap-free full overwrites
  whose update permutation is exactly Identity, Reshape, or Transpose survive.
* ScatterElements indices are normalized at every coordinate.  Only the same
  full permutation on every axis-line survives, and it is replaced by the
  exact inverse Identity, Slice, or Gather.
* An indices initializer is removable only when every one of its graph uses is
  a proved index input selected in the same rewrite group.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import random
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
RESULT = HERE / "scan_result.json"
CANDIDATES = HERE / "candidates"
FRESH_COUNT = 1000
FRESH_SEEDS = (260_000_001, 260_000_002)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "scatter_overwrite_screen_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "scatter_overwrite_trace_helpers",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    result: dict[str, tuple[int, ...]] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                break
            dims.append(int(dim.dim_value))
        else:
            result[value.name] = tuple(dims)
    for item in inferred.graph.initializer:
        if item.dims and all(dim > 0 for dim in item.dims):
            result[item.name] = tuple(int(dim) for dim in item.dims)
    return result


def string_attr(node: onnx.NodeProto, name: str, default: str) -> str:
    for attr in node.attribute:
        if attr.name == name:
            return attr.s.decode("utf-8")
    return default


def int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def normalize_index(value: int, length: int) -> int | None:
    value = value + length if value < 0 else value
    return value if 0 <= value < length else None


def matching_transpose(
    update_ids: np.ndarray, output_ids: np.ndarray
) -> list[int] | None:
    if update_ids.ndim != output_ids.ndim:
        return None
    rank = update_ids.ndim
    candidates: list[list[int]] = [[]]
    for output_axis, wanted in enumerate(output_ids.shape):
        next_candidates = []
        for prefix in candidates:
            for source_axis, length in enumerate(update_ids.shape):
                if source_axis not in prefix and length == wanted:
                    next_candidates.append(prefix + [source_axis])
        candidates = next_candidates
        if not candidates:
            return None
        # Static ARC graphs are low-rank.  Fail closed on a pathological all-1
        # factorial search rather than silently choosing an unproved map.
        if len(candidates) > 100_000:
            return None
    for perm in candidates:
        if np.array_equal(np.transpose(update_ids, perm), output_ids):
            return perm
    return None


def prove_scatter_nd(
    node: onnx.NodeProto,
    index: int,
    shapes: dict[str, tuple[int, ...]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "ScatterND",
        "node_index": index,
        "output": node.output[0] if node.output else "",
        "data": node.input[0] if node.input else "",
        "indices": node.input[1] if len(node.input) > 1 else "",
        "updates": node.input[2] if len(node.input) > 2 else "",
        "convertible": False,
    }
    if len(node.input) != 3 or len(node.output) != 1:
        row["reason"] = "nonstandard_input_or_output_arity"
        return row
    if string_attr(node, "reduction", "none") != "none":
        row["reason"] = "reduction_not_none"
        return row
    data_shape = shapes.get(node.input[0])
    index_shape = shapes.get(node.input[1])
    update_shape = shapes.get(node.input[2])
    output_shape = shapes.get(node.output[0])
    indices = arrays.get(node.input[1])
    if None in (data_shape, index_shape, update_shape, output_shape):
        row["reason"] = "nonstatic_shape"
        return row
    assert data_shape is not None and index_shape is not None
    assert update_shape is not None and output_shape is not None
    if indices is None:
        row["reason"] = "indices_not_initializer"
        return row
    if indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        row["reason"] = "indices_not_int32_or_int64"
        return row
    if tuple(indices.shape) != index_shape or indices.ndim < 1:
        row["reason"] = "initializer_shape_mismatch_or_rank_zero"
        return row
    rank = len(data_shape)
    k = int(index_shape[-1])
    if k <= 0 or k > rank:
        row["reason"] = "index_tuple_length_out_of_range"
        return row
    expected_updates = index_shape[:-1] + data_shape[k:]
    if update_shape != expected_updates:
        row["reason"] = "updates_shape_not_scatternd_formula"
        return row
    if output_shape != data_shape:
        row["reason"] = "output_shape_not_data_shape"
        return row

    prefix_shape = index_shape[:-1]
    prefix_count = math.prod(prefix_shape) if prefix_shape else 1
    expected_prefixes = math.prod(data_shape[:k]) if data_shape[:k] else 1
    if prefix_count != expected_prefixes:
        row.update(
            {
                "reason": "prefix_block_count_not_full",
                "prefix_count": prefix_count,
                "expected_prefix_count": expected_prefixes,
            }
        )
        return row

    normalized: list[tuple[int, ...]] = []
    for raw_tuple in indices.reshape((-1, k)):
        coordinate = []
        for axis, raw in enumerate(raw_tuple):
            value = normalize_index(int(raw), data_shape[axis])
            if value is None:
                row["reason"] = "normalized_index_out_of_range"
                return row
            coordinate.append(value)
        normalized.append(tuple(coordinate))
    unique = set(normalized)
    complete = set(itertools.product(*(range(length) for length in data_shape[:k])))
    if len(unique) != len(normalized):
        row.update(
            {
                "reason": "duplicate_normalized_prefix",
                "duplicate_count": len(normalized) - len(unique),
            }
        )
        return row
    if unique != complete:
        row.update(
            {
                "reason": "missing_or_extra_normalized_prefix",
                "missing_count": len(complete - unique),
                "extra_count": len(unique - complete),
            }
        )
        return row

    update_ids = np.arange(math.prod(update_shape), dtype=np.int64).reshape(update_shape)
    output_ids = np.full(data_shape, -1, dtype=np.int64)
    for prefix_pos, target in zip(np.ndindex(prefix_shape), normalized):
        output_ids[target] = update_ids[prefix_pos]
    if np.any(output_ids < 0) or np.unique(output_ids).size != output_ids.size:
        row["reason"] = "internal_expanded_coordinate_bijection_failure"
        return row

    if update_shape == output_shape and np.array_equal(update_ids, output_ids):
        replacement = {"op_type": "Identity"}
    elif np.array_equal(update_ids.reshape(-1), output_ids.reshape(-1)):
        replacement = {"op_type": "Reshape", "shape": list(output_shape)}
    else:
        perm = matching_transpose(update_ids, output_ids)
        if perm is None:
            row["reason"] = "full_overwrite_but_not_identity_reshape_or_transpose"
            return row
        replacement = {"op_type": "Transpose", "perm": perm}
    row.update(
        {
            "convertible": True,
            "reason": "exhaustive_duplicate_free_full_prefix_block_overwrite",
            "data_shape": list(data_shape),
            "indices_shape": list(index_shape),
            "updates_shape": list(update_shape),
            "output_shape": list(output_shape),
            "index_tuple_length": k,
            "prefix_block_count": prefix_count,
            "normalized_prefixes": [list(value) for value in normalized],
            "replacement": replacement,
        }
    )
    return row


def arithmetic_slice(indices: np.ndarray, axis_length: int) -> dict[str, int] | None:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return None
    if values.size == 1:
        step = 1
    else:
        diffs = np.diff(values)
        step = int(diffs[0])
        if step == 0 or not np.all(diffs == step):
            return None
    start = int(values[0])
    last = int(values[-1])
    if step > 0:
        end = last + 1
    elif last > 0:
        end = last - 1
    else:
        end = -axis_length - 1
    reproduced = np.arange(axis_length, dtype=np.int64)[slice(start, end, step)]
    if not np.array_equal(reproduced, values):
        return None
    return {"start": start, "end": end, "step": step}


def prove_scatter_elements(
    node: onnx.NodeProto,
    index: int,
    shapes: dict[str, tuple[int, ...]],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "ScatterElements",
        "node_index": index,
        "output": node.output[0] if node.output else "",
        "data": node.input[0] if node.input else "",
        "indices": node.input[1] if len(node.input) > 1 else "",
        "updates": node.input[2] if len(node.input) > 2 else "",
        "convertible": False,
    }
    if len(node.input) != 3 or len(node.output) != 1:
        row["reason"] = "nonstandard_input_or_output_arity"
        return row
    if string_attr(node, "reduction", "none") != "none":
        row["reason"] = "reduction_not_none"
        return row
    data_shape = shapes.get(node.input[0])
    index_shape = shapes.get(node.input[1])
    update_shape = shapes.get(node.input[2])
    output_shape = shapes.get(node.output[0])
    indices = arrays.get(node.input[1])
    if None in (data_shape, index_shape, update_shape, output_shape):
        row["reason"] = "nonstatic_shape"
        return row
    assert data_shape is not None and index_shape is not None
    assert update_shape is not None and output_shape is not None
    if indices is None:
        row["reason"] = "indices_not_initializer"
        return row
    if indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        row["reason"] = "indices_not_int32_or_int64"
        return row
    if tuple(indices.shape) != index_shape:
        row["reason"] = "indices_initializer_shape_mismatch"
        return row
    if not (index_shape == update_shape == data_shape == output_shape):
        row["reason"] = "not_full_shape_overwrite"
        return row
    rank = len(data_shape)
    axis = int_attr(node, "axis", 0)
    axis = axis + rank if axis < 0 else axis
    if not 0 <= axis < rank:
        row["reason"] = "axis_out_of_range"
        return row
    axis_length = data_shape[axis]

    normalized = np.empty(index_shape, dtype=np.int64)
    for coordinate in np.ndindex(index_shape):
        value = normalize_index(int(indices[coordinate]), axis_length)
        if value is None:
            row["reason"] = "normalized_index_out_of_range"
            return row
        normalized[coordinate] = value

    moved = np.moveaxis(normalized, axis, -1).reshape((-1, axis_length))
    expected = set(range(axis_length))
    first = moved[0]
    for line_index, line in enumerate(moved):
        if len(set(int(value) for value in line)) != axis_length or set(
            int(value) for value in line
        ) != expected:
            row.update(
                {
                    "reason": "axis_line_has_duplicate_or_missing_target",
                    "failed_line": line_index,
                }
            )
            return row
        if not np.array_equal(line, first):
            row.update(
                {
                    "reason": "axis_lines_do_not_share_one_permutation",
                    "failed_line": line_index,
                }
            )
            return row

    permutation = first.astype(np.int64)
    inverse = np.empty(axis_length, dtype=np.int64)
    inverse[permutation] = np.arange(axis_length, dtype=np.int64)
    update_ids = np.arange(math.prod(update_shape), dtype=np.int64).reshape(update_shape)
    output_ids = np.full(output_shape, -1, dtype=np.int64)
    for coordinate in np.ndindex(index_shape):
        target = list(coordinate)
        target[axis] = int(normalized[coordinate])
        output_ids[tuple(target)] = update_ids[coordinate]
    reproduced = np.take(update_ids, inverse, axis=axis)
    if np.any(output_ids < 0) or not np.array_equal(reproduced, output_ids):
        row["reason"] = "internal_inverse_permutation_self_check_failed"
        return row

    if np.array_equal(inverse, np.arange(axis_length, dtype=np.int64)):
        replacement: dict[str, Any] = {"op_type": "Identity"}
    else:
        slice_spec = arithmetic_slice(inverse, axis_length)
        if slice_spec is not None:
            replacement = {"op_type": "Slice", "axis": axis, **slice_spec}
        else:
            replacement = {
                "op_type": "Gather",
                "axis": axis,
                "inverse_indices": inverse.tolist(),
            }
    row.update(
        {
            "convertible": True,
            "reason": "exhaustive_same_full_permutation_on_every_axis_line",
            "data_shape": list(data_shape),
            "indices_shape": list(index_shape),
            "updates_shape": list(update_shape),
            "output_shape": list(output_shape),
            "axis": axis,
            "line_count": int(moved.shape[0]),
            "normalized_permutation": permutation.tolist(),
            "inverse_permutation": inverse.tolist(),
            "replacement": replacement,
        }
    )
    return row


def synthetic_self_tests() -> list[dict[str, Any]]:
    """Exercise every supported map family plus duplicate rejection."""

    cases: list[tuple[str, dict[str, Any], bool, str | None]] = []

    def nd_case(
        name: str,
        data_shape: tuple[int, ...],
        indices: np.ndarray,
        update_shape: tuple[int, ...],
        expected: str | None,
    ) -> None:
        node = helper.make_node("ScatterND", ["d", "i", "u"], ["o"])
        proof = prove_scatter_nd(
            node,
            0,
            {"d": data_shape, "i": indices.shape, "u": update_shape, "o": data_shape},
            {"i": indices},
        )
        cases.append((name, proof, expected is not None, expected))

    def elements_case(name: str, indices: np.ndarray, expected: str | None) -> None:
        shape = tuple(int(value) for value in indices.shape)
        node = helper.make_node("ScatterElements", ["d", "i", "u"], ["o"], axis=1)
        proof = prove_scatter_elements(
            node,
            0,
            {"d": shape, "i": shape, "u": shape, "o": shape},
            {"i": indices},
        )
        cases.append((name, proof, expected is not None, expected))

    nd_case(
        "scatternd_identity",
        (2, 3),
        np.asarray([[0], [1]], dtype=np.int64),
        (2, 3),
        "Identity",
    )
    nd_case(
        "scatternd_reshape",
        (1, 2),
        np.asarray([0], dtype=np.int64),
        (2,),
        "Reshape",
    )
    transpose_indices = np.empty((3, 2, 2), dtype=np.int64)
    for a, b in np.ndindex((3, 2)):
        transpose_indices[a, b] = [b, a]
    nd_case(
        "scatternd_transpose",
        (2, 3),
        transpose_indices,
        (3, 2),
        "Transpose",
    )
    nd_case(
        "scatternd_duplicate_reject",
        (2, 3),
        np.asarray([[0], [0]], dtype=np.int64),
        (2, 3),
        None,
    )
    elements_case(
        "scatterelements_identity",
        np.tile(np.arange(3, dtype=np.int64), (2, 1)),
        "Identity",
    )
    elements_case(
        "scatterelements_reverse_slice",
        np.tile(np.asarray([2, 1, 0], dtype=np.int64), (2, 1)),
        "Slice",
    )
    elements_case(
        "scatterelements_inverse_gather",
        np.tile(np.asarray([2, 0, 1], dtype=np.int64), (2, 1)),
        "Gather",
    )
    duplicate = np.tile(np.asarray([0, 0, 2], dtype=np.int64), (2, 1))
    elements_case("scatterelements_duplicate_reject", duplicate, None)

    results = []
    for name, proof, expected_convertible, expected_op in cases:
        actual_convertible = bool(proof["convertible"])
        actual_op = proof.get("replacement", {}).get("op_type")
        passed = actual_convertible == expected_convertible and actual_op == expected_op
        results.append(
            {
                "name": name,
                "passed": passed,
                "expected_convertible": expected_convertible,
                "actual_convertible": actual_convertible,
                "expected_op_type": expected_op,
                "actual_op_type": actual_op,
                "reason": proof["reason"],
            }
        )
    if not all(row["passed"] for row in results):
        raise AssertionError(f"scatter proof self-test failed: {results}")
    return results


def discover(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    shapes = tensor_shapes(model)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                uses[name].append((node_index, input_index))

    proofs = []
    convertible_by_node: dict[int, dict[str, Any]] = {}
    for node_index, node in enumerate(model.graph.node):
        if node.op_type == "ScatterND":
            proof = prove_scatter_nd(node, node_index, shapes, arrays)
        elif node.op_type == "ScatterElements":
            proof = prove_scatter_elements(node, node_index, shapes, arrays)
        else:
            continue
        proofs.append(proof)
        if proof["convertible"]:
            convertible_by_node[node_index] = proof

    groups = []
    grouped_nodes: set[int] = set()
    for name, array in arrays.items():
        user_slots = uses.get(name, [])
        if not user_slots:
            continue
        if all(
            input_index == 1
            and node_index in convertible_by_node
            and model.graph.node[node_index].input[1] == name
            for node_index, input_index in user_slots
        ):
            groups.append(
                {
                    "initializer": name,
                    "initializer_params": int(array.size),
                    "initializer_dtype": str(array.dtype),
                    "remove_initializer": True,
                    "node_indices": [node_index for node_index, _ in user_slots],
                    "proofs": [convertible_by_node[node_index] for node_index, _ in user_slots],
                }
            )
            grouped_nodes.update(node_index for node_index, _ in user_slots)
    # A full-overwrite node can still be profitable when its indices tensor is
    # shared with an unrelated use: the initializer is retained and only the
    # now-dead data branch is pruned.  Such nodes are independent rewrite units;
    # parameter deletion is never credited to them.
    retained_index_groups = []
    for node_index, proof in convertible_by_node.items():
        if node_index in grouped_nodes:
            continue
        name = proof["indices"]
        retained_index_groups.append(
            {
                "initializer": name,
                "initializer_params": int(arrays[name].size),
                "initializer_dtype": str(arrays[name].dtype),
                "remove_initializer": False,
                "shared_uses": [list(slot) for slot in uses[name]],
                "node_indices": [node_index],
                "proofs": [proof],
            }
        )
    rewrite_groups = groups + retained_index_groups
    return {
        "task": task,
        "authority_sha256": sha256(data),
        "scatter_nd_count": sum(proof["kind"] == "ScatterND" for proof in proofs),
        "scatter_elements_count": sum(
            proof["kind"] == "ScatterElements" for proof in proofs
        ),
        "convertible_node_count": sum(proof["convertible"] for proof in proofs),
        "proofs": proofs,
        "all_use_convertible_groups": groups,
        "retained_index_rewrite_groups": retained_index_groups,
        "rewrite_groups": rewrite_groups,
        "candidate_possible": bool(rewrite_groups),
    }


def unique_name(existing: set[str], base: str) -> str:
    name = base
    suffix = 0
    while name in existing:
        suffix += 1
        name = f"{base}_{suffix}"
    existing.add(name)
    return name


def find_vector_initializer(
    model: onnx.ModelProto,
    values: Iterable[int],
    excluded: set[str],
) -> str | None:
    wanted = np.asarray(list(values), dtype=np.int64)
    for item in model.graph.initializer:
        if item.name in excluded or item.data_type not in (TensorProto.INT32, TensorProto.INT64):
            continue
        array = np.asarray(numpy_helper.to_array(item), dtype=np.int64)
        if array.ndim == 1 and np.array_equal(array, wanted):
            return item.name
    return None


def prune_unreachable(model: onnx.ModelProto) -> None:
    live = {value.name for value in model.graph.output}
    keep_indices: set[int] = set()
    for node_index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[node_index]
        if any(name and name in live for name in node.output):
            keep_indices.add(node_index)
            live.update(name for name in node.input if name)
    kept_nodes = [
        copy.deepcopy(node)
        for node_index, node in enumerate(model.graph.node)
        if node_index in keep_indices
    ]
    retained_outputs = {
        name for node in kept_nodes for name in node.output if name
    }
    graph_inputs = {value.name for value in model.graph.input}
    kept_initializers = [
        copy.deepcopy(item)
        for item in model.graph.initializer
        if item.name in live or item.name in graph_inputs
    ]
    kept_value_info = [
        copy.deepcopy(value)
        for value in model.graph.value_info
        if value.name in retained_outputs
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)


def build_candidate(
    data: bytes, groups: list[dict[str, Any]]
) -> tuple[bytes, dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    remove_indices = {
        group["initializer"] for group in groups if group["remove_initializer"]
    }
    proof_by_node = {
        int(proof["node_index"]): proof for group in groups for proof in group["proofs"]
    }
    existing_names = {
        name
        for node in model.graph.node
        for name in list(node.input) + list(node.output)
        if name
    } | {item.name for item in model.graph.initializer}
    added_initializers: list[dict[str, Any]] = []
    constant_cache: dict[tuple[int, ...], str] = {}

    # Reuse constants that do not belong to a removed all-use indices group.
    for item in model.graph.initializer:
        if item.name in remove_indices or item.data_type not in (
            TensorProto.INT32,
            TensorProto.INT64,
        ):
            continue
        array = np.asarray(numpy_helper.to_array(item), dtype=np.int64)
        if array.ndim == 1:
            constant_cache.setdefault(tuple(int(value) for value in array), item.name)

    def vector_constant(values: Iterable[int], label: str) -> str:
        key = tuple(int(value) for value in values)
        if key in constant_cache:
            return constant_cache[key]
        name = unique_name(existing_names, f"scatter_overwrite_{label}")
        model.graph.initializer.append(
            numpy_helper.from_array(np.asarray(key, dtype=np.int64), name=name)
        )
        constant_cache[key] = name
        added_initializers.append({"name": name, "values": list(key)})
        return name

    for node_index, proof in proof_by_node.items():
        node = model.graph.node[node_index]
        updates = proof["updates"]
        output = proof["output"]
        replacement = proof["replacement"]
        op_type = replacement["op_type"]
        del node.input[:]
        del node.output[:]
        del node.attribute[:]
        node.op_type = op_type
        node.domain = ""
        node.output.extend([output])
        if op_type == "Identity":
            node.input.extend([updates])
        elif op_type == "Reshape":
            shape_name = vector_constant(replacement["shape"], "shape")
            node.input.extend([updates, shape_name])
        elif op_type == "Transpose":
            node.input.extend([updates])
            node.attribute.extend([helper.make_attribute("perm", replacement["perm"])])
        elif op_type == "Gather":
            inverse_name = vector_constant(replacement["inverse_indices"], "inverse")
            node.input.extend([updates, inverse_name])
            node.attribute.extend([helper.make_attribute("axis", replacement["axis"])])
        elif op_type == "Slice":
            starts = vector_constant([replacement["start"]], "slice_scalar")
            ends = vector_constant([replacement["end"]], "slice_scalar")
            axes = vector_constant([replacement["axis"]], "slice_scalar")
            steps = vector_constant([replacement["step"]], "slice_scalar")
            node.input.extend([updates, starts, ends, axes, steps])
        else:  # pragma: no cover - all proof constructors are enumerated above.
            raise AssertionError(op_type)
        node.doc_string = "exact proved full-overwrite Scatter replacement"

    remaining = [
        copy.deepcopy(item)
        for item in model.graph.initializer
        if item.name not in remove_indices
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(remaining)
    before_prune_nodes = len(model.graph.node)
    before_prune_params = scoring.calculate_params(model)
    prune_unreachable(model)
    after_prune_params = scoring.calculate_params(model)
    metadata = {
        "removed_indices_initializers": sorted(remove_indices),
        "added_initializers": added_initializers,
        "dead_nodes_pruned": before_prune_nodes - len(model.graph.node),
        "params_before_prune": before_prune_params,
        "params_after_prune": after_prune_params,
        "replacement_nodes": [
            {
                "node_index": node_index,
                "kind": proof_by_node[node_index]["kind"],
                "replacement": proof_by_node[node_index]["replacement"],
            }
            for node_index in sorted(proof_by_node)
        ],
    }
    return model.SerializeToString(), metadata


def declared_cost(data: bytes) -> dict[str, int] | None:
    model = onnx.load_model_from_string(data)
    params = scoring.calculate_params(model)
    if params is None:
        return None
    shapes = tensor_shapes(model)
    type_map = {
        value.name: value.type.tensor_type.elem_type
        for value in list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
        if value.type.HasField("tensor_type")
    }
    memory = 0
    for node in model.graph.node:
        for output in node.output:
            if not output or output == "output":
                continue
            shape = shapes.get(output)
            elem_type = type_map.get(output)
            if shape is None or elem_type is None:
                return None
            memory += math.prod(shape) * np.dtype(
                helper.tensor_dtype_to_np_dtype(elem_type)
            ).itemsize
    return {"memory_bytes": int(memory), "params": int(params), "cost": int(memory + params)}


def choose_groups(
    data: bytes, groups: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    authority_proxy = declared_cost(data)
    if not groups:
        return [], {"authority_declared_cost": authority_proxy, "subsets_evaluated": 0}
    if len(groups) <= 16:
        subsets = itertools.chain.from_iterable(
            itertools.combinations(range(len(groups)), count)
            for count in range(1, len(groups) + 1)
        )
    else:
        # This fallback was not reached on the pinned archive.  Each group is
        # semantically independent and full overwrite never increases live
        # activation shape; evaluate the combined graph fail-closed.
        subsets = [tuple(range(len(groups)))]
    best_indices: tuple[int, ...] = ()
    best_cost: dict[str, int] | None = None
    best_metadata: dict[str, Any] | None = None
    evaluated = 0
    for subset in subsets:
        evaluated += 1
        candidate, metadata = build_candidate(data, [groups[i] for i in subset])
        cost = declared_cost(candidate)
        if cost is None:
            continue
        if best_cost is None or cost["cost"] < best_cost["cost"] or (
            cost["cost"] == best_cost["cost"] and len(subset) > len(best_indices)
        ):
            best_indices = tuple(subset)
            best_cost = cost
            best_metadata = metadata
    selected = [groups[index] for index in best_indices]
    return selected, {
        "authority_declared_cost": authority_proxy,
        "candidate_declared_cost": best_cost,
        "selected_group_indices": list(best_indices),
        "subsets_evaluated": evaluated,
        "build_metadata": best_metadata,
    }


def full_validation(data: bytes) -> dict[str, Any]:
    row: dict[str, Any] = {"checker_full": False, "shape_inference_strict_data_prop": False}
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        row["checker_error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        row["shape_inference_strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["shape_inference_error"] = f"{type(exc).__name__}: {exc}"
    return row


def score(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"scatter_overwrite_{task:03d}_{label}_") as wd:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, wd, label=label, require_correct=False
        )


def known_raw_four(task: int, authority: bytes, candidate: bytes) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    result: dict[str, Any] = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            stats = {
                "total": 0,
                "candidate_right": 0,
                "raw_equal": 0,
                "threshold_equal": 0,
                "authority_errors": 0,
                "candidate_errors": 0,
            }
            try:
                base_session = SCREEN.make_session(authority, disabled, threads)
                cand_session = SCREEN.make_session(candidate, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["session_error"] = f"{type(exc).__name__}: {exc}"
                result[key] = stats
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    stats["total"] += 1
                    try:
                        base = base_session.run(
                            None, {base_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["authority_errors"] += 1
                        continue
                    try:
                        cand = cand_session.run(
                            None, {cand_session.get_inputs()[0].name: benchmark["input"]}
                        )[0]
                    except Exception:  # noqa: BLE001
                        stats["candidate_errors"] += 1
                        continue
                    stats["candidate_right"] += int(
                        np.array_equal(cand > 0, benchmark["output"] > 0)
                    )
                    stats["raw_equal"] += int(np.array_equal(cand, base))
                    stats["threshold_equal"] += int(np.array_equal(cand > 0, base > 0))
            result[key] = stats
    return result


def known_pass(report: dict[str, Any]) -> bool:
    return len(report) == 4 and all(
        row.get("total", 0) > 0
        and row.get("candidate_right") == row.get("total")
        and row.get("raw_equal") == row.get("total")
        and row.get("threshold_equal") == row.get("total")
        and row.get("authority_errors") == 0
        and row.get("candidate_errors") == 0
        and not row.get("session_error")
        for row in report.values()
    )


def fresh_four(task: int, candidate: bytes) -> dict[str, Any]:
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    configs = (
        (True, 1, "disable_all_threads1"),
        (True, 4, "disable_all_threads4"),
        (False, 1, "default_threads1"),
        (False, 4, "default_threads4"),
    )
    sessions = {
        name: SCREEN.make_session(candidate, disabled, threads)
        for disabled, threads, name in configs
    }
    runs = []
    for seed in FRESH_SEEDS:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {name: {"right": 0, "wrong": 0, "errors": 0} for _, _, name in configs}
        valid = attempts = generation_errors = conversion_skips = 0
        while valid < FRESH_COUNT:
            attempts += 1
            try:
                benchmark = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if benchmark is None:
                conversion_skips += 1
                continue
            valid += 1
            want = benchmark["output"] > 0
            for _, _, name in configs:
                try:
                    session = sessions[name]
                    raw = session.run(
                        None, {session.get_inputs()[0].name: benchmark["input"]}
                    )[0]
                    if np.array_equal(raw > 0, want):
                        stats[name]["right"] += 1
                    else:
                        stats[name]["wrong"] += 1
                except Exception:  # noqa: BLE001
                    stats[name]["errors"] += 1
        runs.append(
            {
                "seed": seed,
                "valid": valid,
                "attempts": attempts,
                "generation_errors": generation_errors,
                "conversion_skips": conversion_skips,
                "configs": stats,
            }
        )
    return {"count_per_seed": FRESH_COUNT, "seeds": list(FRESH_SEEDS), "runs": runs}


def main() -> None:
    ort.set_default_logger_severity(4)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority archive changed")
    self_tests = synthetic_self_tests()
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    discoveries = []
    payloads: dict[int, bytes] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in range(1, 401):
            data = archive.read(f"task{task:03d}.onnx")
            payloads[task] = data
            row = discover(task, data)
            if row["scatter_nd_count"] or row["scatter_elements_count"]:
                selected, selection = choose_groups(
                    data, row["rewrite_groups"]
                )
                row["selected_groups"] = selected
                row["selection"] = selection
                row["candidate_possible"] = bool(selected)
                discoveries.append(row)

    candidate_rows = []
    for discovery in discoveries:
        if not discovery["candidate_possible"]:
            continue
        task = int(discovery["task"])
        authority = payloads[task]
        candidate, build_metadata = build_candidate(
            authority, discovery["selected_groups"]
        )
        path = CANDIDATES / f"task{task:03d}_scatter_overwrite.onnx"
        onnx.save_model(onnx.load_model_from_string(candidate), path)
        candidate = path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "authority_sha256": sha256(authority),
            "candidate_sha256": sha256(candidate),
            "discovery": discovery,
            "build_metadata": build_metadata,
            "authority_score": score(task, authority, "authority"),
            "candidate_score": score(task, candidate, "candidate"),
            "full_validation": full_validation(candidate),
            "strict": SCREEN.structural_audit(candidate),
        }
        try:
            row["authority_runtime_shape"] = TRACE.runtime_shape_trace(
                task, onnx.load_model_from_string(authority)
            )
            trace = TRACE.runtime_shape_trace(task, onnx.load_model_from_string(candidate))
            row["runtime_shape"] = trace
            row["truthful"] = not trace["declared_actual_mismatches"]
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_error"] = f"{type(exc).__name__}: {exc}"
            row["truthful"] = False
        base_score = row["authority_score"]
        cand_score = row["candidate_score"]
        row["strictly_lower"] = bool(
            base_score and cand_score and cand_score["cost"] < base_score["cost"]
        )
        full_ok = all(row["full_validation"].get(key, False) for key in (
            "checker_full",
            "shape_inference_strict_data_prop",
        ))
        if not row["strictly_lower"]:
            row["decision"] = "REJECT_NOT_STRICTLY_LOWER_OR_UNSCORABLE"
        elif not full_ok or not row["strict"]["pass"] or not row["strict"].get("conv_bias_ub0"):
            row["decision"] = "REJECT_FULL_STRICT_SCHEMA_OR_UB"
        else:
            known = known_raw_four(task, authority, candidate)
            row["known_raw_four"] = known
            row["known_raw_four_pass"] = known_pass(known)
            if not cand_score.get("correct"):
                row["decision"] = "REJECT_OFFICIAL_KNOWN"
            elif not row["truthful"]:
                row["decision"] = "REJECT_RUNTIME_SHAPE"
            elif not row["known_raw_four_pass"]:
                row["decision"] = "REJECT_KNOWN_RAW_OR_RUNTIME"
            else:
                fresh = fresh_four(task, candidate)
                row["fresh"] = fresh
                row["fresh_pass"] = all(
                    config["right"] == FRESH_COUNT
                    and config["wrong"] == 0
                    and config["errors"] == 0
                    for run in fresh["runs"]
                    for config in run["configs"].values()
                )
                row["decision"] = "ACCEPT" if row["fresh_pass"] else "REJECT_FRESH"
        candidate_rows.append(row)
        print(
            f"task{task:03d} groups={len(discovery['selected_groups'])} "
            f"cost={base_score}->{cand_score} decision={row['decision']}",
            flush=True,
        )

    accepted = [row for row in candidate_rows if row["decision"] == "ACCEPT"]
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "synthetic_self_tests": self_tests,
        "tasks_with_scatter": len(discoveries),
        "scatter_nd_nodes": sum(row["scatter_nd_count"] for row in discoveries),
        "scatter_elements_nodes": sum(
            row["scatter_elements_count"] for row in discoveries
        ),
        "convertible_nodes": sum(row["convertible_node_count"] for row in discoveries),
        "all_use_groups": sum(
            len(row["all_use_convertible_groups"]) for row in discoveries
        ),
        "retained_index_rewrite_groups": sum(
            len(row["retained_index_rewrite_groups"]) for row in discoveries
        ),
        "candidate_tasks": len(candidate_rows),
        "discoveries": discoveries,
        "candidates": candidate_rows,
        "accepted": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["candidate_sha256"],
                "authority_cost": row["authority_score"]["cost"],
                "candidate_cost": row["candidate_score"]["cost"],
            }
            for row in accepted
        ],
        "decision": "ACCEPT" if accepted else "NO_SAFE_SCATTER_FULL_OVERWRITE_WINNER",
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks_with_scatter": report["tasks_with_scatter"],
                "scatter_nd_nodes": report["scatter_nd_nodes"],
                "scatter_elements_nodes": report["scatter_elements_nodes"],
                "convertible_nodes": report["convertible_nodes"],
                "all_use_groups": report["all_use_groups"],
                "candidate_tasks": report["candidate_tasks"],
                "accepted": len(accepted),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

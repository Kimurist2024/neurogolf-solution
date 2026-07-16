#!/usr/bin/env python3
"""All-400 exact binary reduction/predicate fusion scan."""

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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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
FRESH_SEEDS = (266_000_001, 266_000_002)

REDUCTIONS = {"ReduceSum", "ReduceMax", "ReduceMin"}
COMPARISONS = {"Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"}
BOOL_LOGIC = {"And", "Or", "Xor", "Not"}
BOOL_PASSTHROUGH = {
    "Identity",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Slice",
    "Gather",
    "GatherElements",
    "GatherND",
    "Concat",
    "CenterCropPad",
    "Pad",
    "Expand",
    "Tile",
}

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
    "reduction_predicate_screen_helpers",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20i_94/screen_incremental.py",
)
TRACE = load_module(
    "reduction_predicate_trace_helpers",
    ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_opset(model: onnx.ModelProto) -> int:
    return next(
        int(item.version)
        for item in model.opset_import
        if item.domain in ("", "ai.onnx")
    )


def tensor_info(
    model: onnx.ModelProto,
) -> tuple[dict[str, tuple[int, ...]], dict[str, int]]:
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=False, data_prop=True
    )
    shapes: dict[str, tuple[int, ...]] = {}
    types: dict[str, int] = {}
    for value in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        tensor = value.type.tensor_type
        if not value.type.HasField("tensor_type"):
            continue
        types[value.name] = int(tensor.elem_type)
        dims = []
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                break
            dims.append(int(dim.dim_value))
        else:
            shapes[value.name] = tuple(dims)
    for item in inferred.graph.initializer:
        types[item.name] = int(item.data_type)
        if item.dims and all(dim > 0 for dim in item.dims):
            shapes[item.name] = tuple(int(dim) for dim in item.dims)
        elif not item.dims:
            shapes[item.name] = ()
    return shapes, types


def int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    return next((int(attr.i) for attr in node.attribute if attr.name == name), default)


def arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }


def producer_and_uses(model: onnx.ModelProto):
    producers = {
        output: (index, node)
        for index, node in enumerate(model.graph.node)
        for output in node.output
        if output
    }
    uses: dict[str, list[tuple[int, int, onnx.NodeProto]]] = defaultdict(list)
    for index, node in enumerate(model.graph.node):
        for slot, name in enumerate(node.input):
            if name:
                uses[name].append((index, slot, node))
    return producers, uses


def prove_bool_lineage(
    name: str,
    types: dict[str, int],
    producers: dict[str, tuple[int, onnx.NodeProto]],
    memo: dict[str, dict[str, Any] | None],
    active: set[str] | None = None,
) -> dict[str, Any] | None:
    if name in memo:
        return memo[name]
    if types.get(name) != TensorProto.BOOL:
        memo[name] = None
        return None
    active = set() if active is None else active
    if name in active:
        memo[name] = None
        return None
    active.add(name)
    source = producers.get(name)
    if source is None:
        memo[name] = None
        return None
    index, node = source
    proof: dict[str, Any] | None = None
    if node.op_type in COMPARISONS:
        proof = {
            "bool_source": name,
            "producer_index": index,
            "producer_op": node.op_type,
            "lineage": [node.op_type],
        }
    elif node.op_type in BOOL_LOGIC:
        parents = [
            prove_bool_lineage(value, types, producers, memo, active)
            for value in node.input
            if types.get(value) == TensorProto.BOOL
        ]
        if parents and all(parent is not None for parent in parents):
            proof = {
                "bool_source": name,
                "producer_index": index,
                "producer_op": node.op_type,
                "lineage": [node.op_type]
                + sorted({op for parent in parents for op in parent["lineage"]}),
            }
    elif node.op_type in BOOL_PASSTHROUGH:
        parents = [
            prove_bool_lineage(value, types, producers, memo, active)
            for value in node.input
            if types.get(value) == TensorProto.BOOL
        ]
        if parents and all(parent is not None for parent in parents):
            proof = {
                "bool_source": name,
                "producer_index": index,
                "producer_op": node.op_type,
                "lineage": [node.op_type]
                + sorted({op for parent in parents for op in parent["lineage"]}),
            }
    active.remove(name)
    memo[name] = proof
    return proof


def prove_binary_carrier(
    name: str,
    types: dict[str, int],
    producers: dict[str, tuple[int, onnx.NodeProto]],
    values: dict[str, np.ndarray],
    bool_memo: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    elem_type = types.get(name)
    if elem_type == TensorProto.BOOL:
        proof = prove_bool_lineage(name, types, producers, bool_memo)
        if proof is not None:
            return {**proof, "carrier": name, "carrier_kind": "bool", "cast_index": None}
        return None
    source = producers.get(name)
    if source is None:
        return None
    index, node = source
    if node.op_type in ("Cast", "CastLike") and node.input:
        proof = prove_bool_lineage(node.input[0], types, producers, bool_memo)
        if proof is not None:
            return {
                **proof,
                "carrier": name,
                "carrier_kind": "cast_from_proved_bool",
                "cast_index": index,
                "cast_op": node.op_type,
            }
    if node.op_type == "OneHot" and len(node.input) >= 3 and node.input[2] in values:
        onehot_values = np.asarray(values[node.input[2]]).reshape(-1)
        if onehot_values.size == 2 and set(onehot_values.tolist()).issubset({0, 1}):
            return {
                "carrier": name,
                "carrier_kind": "proved_binary_onehot",
                "cast_index": None,
                "bool_source": None,
                "producer_index": index,
                "producer_op": "OneHot",
                "lineage": ["OneHot(values_subset_0_1)"],
            }
    return None


def reduction_axes(
    node: onnx.NodeProto,
    input_shape: tuple[int, ...],
    values: dict[str, np.ndarray],
) -> tuple[list[int], int] | None:
    rank = len(input_shape)
    axes: list[int] | None = None
    if len(node.input) > 1 and node.input[1]:
        array = values.get(node.input[1])
        if array is None or not np.issubdtype(array.dtype, np.integer):
            return None
        raw = [int(value) for value in np.asarray(array).reshape(-1)]
        if not raw and int_attr(node, "noop_with_empty_axes", 0):
            axes = []
        elif not raw:
            axes = list(range(rank))
        else:
            axes = raw
    else:
        attr = next((attr for attr in node.attribute if attr.name == "axes"), None)
        axes = list(attr.ints) if attr is not None else list(range(rank))
    normalized = []
    for axis in axes:
        axis = axis + rank if axis < 0 else axis
        if not 0 <= axis < rank or axis in normalized:
            return None
        normalized.append(axis)
    count = math.prod(input_shape[axis] for axis in normalized) if normalized else 1
    return normalized, int(count)


def compare_value(op_type: str, left: int | float, right: int | float) -> bool:
    if op_type == "Equal":
        return left == right
    if op_type == "Greater":
        return left > right
    if op_type == "GreaterOrEqual":
        return left >= right
    if op_type == "Less":
        return left < right
    if op_type == "LessOrEqual":
        return left <= right
    raise AssertionError(op_type)


def classify_predicate(
    reduction: onnx.NodeProto,
    comparison: onnx.NodeProto,
    reduced_count: int,
    values: dict[str, np.ndarray],
    invert: bool,
) -> dict[str, Any] | None:
    reduction_output = reduction.output[0]
    slots = [slot for slot, name in enumerate(comparison.input) if name == reduction_output]
    if len(slots) != 1 or len(comparison.input) != 2:
        return None
    slot = slots[0]
    constant_name = comparison.input[1 - slot]
    constant = values.get(constant_name)
    if constant is None or constant.size != 1:
        return None
    threshold = np.asarray(constant).reshape(-1)[0].item()
    aggregate_values = range(reduced_count + 1) if reduction.op_type == "ReduceSum" else range(2)
    actual = []
    for value in aggregate_values:
        result = compare_value(
            comparison.op_type,
            value if slot == 0 else threshold,
            threshold if slot == 0 else value,
        )
        actual.append(not result if invert else result)
    if reduction.op_type == "ReduceMax":
        target_any = [False, True]
        return {
            "predicate": "Any",
            "desired_reduction": "ReduceMax",
            "truth_table": actual,
            "constant": constant_name,
            "threshold": threshold,
        } if actual == target_any else None
    if reduction.op_type == "ReduceMin":
        target_all = [False, True]
        return {
            "predicate": "All",
            "desired_reduction": "ReduceMin",
            "truth_table": actual,
            "constant": constant_name,
            "threshold": threshold,
        } if actual == target_all else None
    any_table = [False] + [True] * reduced_count
    all_table = [False] * reduced_count + [True]
    if actual == any_table:
        predicate, desired = "Any", "ReduceMax"
    elif actual == all_table:
        predicate, desired = "All", "ReduceMin"
    else:
        return None
    return {
        "predicate": predicate,
        "desired_reduction": desired,
        "truth_table": actual,
        "constant": constant_name,
        "threshold": threshold,
    }


def make_predicate_unit(
    model: onnx.ModelProto,
    reduction_index: int,
    shapes: dict[str, tuple[int, ...]],
    types: dict[str, int],
    values: dict[str, np.ndarray],
    producers,
    uses,
    bool_memo,
) -> tuple[dict[str, Any] | None, str]:
    reduction = model.graph.node[reduction_index]
    if not reduction.input or not reduction.output:
        return None, "missing_reduction_io"
    carrier_shape = shapes.get(reduction.input[0])
    output_shape = shapes.get(reduction.output[0])
    if carrier_shape is None or output_shape is None:
        return None, "nonstatic_reduction_shape"
    binary = prove_binary_carrier(
        reduction.input[0], types, producers, values, bool_memo
    )
    if binary is None:
        return None, "input_domain_not_formally_binary"
    axis_info = reduction_axes(reduction, carrier_shape, values)
    if axis_info is None:
        return None, "dynamic_or_invalid_axes"
    axes, reduced_count = axis_info
    reducer_uses = uses.get(reduction.output[0], [])
    if len(reducer_uses) != 1:
        return None, "reduction_output_not_single_use"
    compare_index, _, comparison = reducer_uses[0]
    if comparison.op_type not in COMPARISONS or not comparison.output:
        return None, "single_consumer_not_comparison"
    predicate_output = comparison.output[0]
    invert_index = None
    final_predicate = predicate_output
    invert = False
    comparison_uses = uses.get(predicate_output, [])
    if (
        len(comparison_uses) == 1
        and comparison_uses[0][2].op_type == "Not"
        and comparison_uses[0][1] == 0
    ):
        invert_index = comparison_uses[0][0]
        final_predicate = comparison_uses[0][2].output[0]
        invert = True
    classification = classify_predicate(
        reduction, comparison, reduced_count, values, invert
    )
    if classification is None:
        return None, "comparison_truth_table_not_any_or_all"
    if shapes.get(final_predicate) != output_shape:
        return None, "predicate_output_shape_not_reduction_shape"

    final_uses = uses.get(final_predicate, [])
    graph_outputs = {value.name for value in model.graph.output}
    opset = default_opset(model)
    carrier_type = types[reduction.input[0]]
    bool_source = binary.get("bool_source")
    cast_index = binary.get("cast_index")
    cast_only_this_reducer = bool(
        cast_index is not None
        and len(uses.get(reduction.input[0], [])) == 1
        and uses[reduction.input[0]][0][0] == reduction_index
    )

    # Bool ReduceMax/Min entered the standard type constraint in opset 20.
    if opset >= 20 and bool_source is not None and cast_only_this_reducer:
        mode = "bool_reduce_fusion"
        deleted = [compare_index] + ([invert_index] if invert_index is not None else [])
        target_output = final_predicate
        rewired_casts: list[int] = []
        delete_final_cast = None
    else:
        if final_predicate in graph_outputs or not final_uses:
            return None, "carrier_fusion_needs_cast_only_predicate_consumers"
        if not all(
            slot == 0 and node.op_type in ("Cast", "CastLike")
            for _, slot, node in final_uses
        ):
            return None, "predicate_has_noncast_consumer"
        mode = "numeric_carrier_cast_absorption"
        rewired_casts = [index for index, _, _ in final_uses]
        deleted = [compare_index] + ([invert_index] if invert_index is not None else [])
        delete_final_cast = None
        target_output = reduction.output[0]
        if len(final_uses) == 1:
            final_cast_index, _, final_cast = final_uses[0]
            final_type = types.get(final_cast.output[0])
            if final_type == carrier_type:
                delete_final_cast = final_cast_index
                deleted.append(final_cast_index)
                rewired_casts = []
                target_output = final_cast.output[0]
                mode = "numeric_carrier_direct_output"

    return {
        "kind": "predicate_fusion",
        "mode": mode,
        "reduction_index": reduction_index,
        "comparison_index": compare_index,
        "invert_index": invert_index,
        "delete_node_indices": sorted(index for index in deleted if index is not None),
        "rewire_cast_indices": rewired_casts,
        "delete_final_cast_index": delete_final_cast,
        "carrier": reduction.input[0],
        "carrier_type": int(carrier_type),
        "carrier_shape": list(carrier_shape),
        "bool_source": bool_source,
        "binary_proof": binary,
        "axes": axes,
        "reduced_count": reduced_count,
        "old_reduction": reduction.op_type,
        "desired_reduction": classification["desired_reduction"],
        "predicate": classification["predicate"],
        "threshold": classification["threshold"],
        "truth_table": classification["truth_table"],
        "constant": classification["constant"],
        "old_reduction_output": reduction.output[0],
        "predicate_output": final_predicate,
        "target_output": target_output,
        "carrier_cast_index": cast_index,
        "delete_carrier_cast": mode == "bool_reduce_fusion" and cast_only_this_reducer,
        "opset": opset,
    }, "proved_any_all_predicate"


def make_squeeze_unit(
    model: onnx.ModelProto,
    reduction_index: int,
    shapes: dict[str, tuple[int, ...]],
    types: dict[str, int],
    values: dict[str, np.ndarray],
    producers,
    uses,
    bool_memo,
) -> tuple[dict[str, Any] | None, str]:
    reduction = model.graph.node[reduction_index]
    if reduction.op_type not in ("ReduceMax", "ReduceMin"):
        return None, "not_max_min_for_squeeze"
    carrier_shape = shapes.get(reduction.input[0])
    old_output_shape = shapes.get(reduction.output[0])
    if carrier_shape is None or old_output_shape is None:
        return None, "nonstatic_squeeze_chain_shape"
    binary = prove_binary_carrier(
        reduction.input[0], types, producers, values, bool_memo
    )
    if binary is None:
        return None, "squeeze_input_not_formally_binary"
    reducer_uses = uses.get(reduction.output[0], [])
    if len(reducer_uses) != 1 or reducer_uses[0][2].op_type != "Squeeze":
        return None, "reduction_not_single_use_squeeze"
    squeeze_index, _, squeeze = reducer_uses[0]
    if not squeeze.output or shapes.get(squeeze.output[0]) != ():
        return None, "squeeze_output_not_scalar"
    axis_info = reduction_axes(reduction, carrier_shape, values)
    if axis_info is None:
        return None, "dynamic_or_invalid_axes"
    axes, _ = axis_info
    if any(carrier_shape[axis] != 1 for axis in range(len(carrier_shape)) if axis not in axes):
        return None, "unreduced_axis_not_static_singleton"
    return {
        "kind": "reduce_squeeze_all_axes",
        "reduction_index": reduction_index,
        "squeeze_index": squeeze_index,
        "delete_node_indices": [squeeze_index],
        "carrier": reduction.input[0],
        "carrier_shape": list(carrier_shape),
        "binary_proof": binary,
        "old_axes": axes,
        "old_output": reduction.output[0],
        "target_output": squeeze.output[0],
        "desired_reduction": reduction.op_type,
        "opset": default_opset(model),
    }, "proved_singleton_unreduced_axes"


def discover(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    shapes, types = tensor_info(model)
    values = arrays(model)
    producers, uses = producer_and_uses(model)
    bool_memo: dict[str, dict[str, Any] | None] = {}
    reasons = Counter()
    predicate_units = []
    squeeze_units = []
    reduction_count = 0
    binary_reduction_count = 0
    occupied: set[int] = set()
    for index, node in enumerate(model.graph.node):
        if node.op_type not in REDUCTIONS:
            continue
        reduction_count += 1
        if prove_binary_carrier(node.input[0], types, producers, values, bool_memo):
            binary_reduction_count += 1
        unit, reason = make_predicate_unit(
            model, index, shapes, types, values, producers, uses, bool_memo
        )
        if unit is not None:
            nodes = {index, *unit["delete_node_indices"]}
            if not nodes & occupied:
                predicate_units.append(unit)
                occupied.update(nodes)
                continue
        reasons[reason] += 1
        unit, squeeze_reason = make_squeeze_unit(
            model, index, shapes, types, values, producers, uses, bool_memo
        )
        if unit is not None:
            nodes = {index, *unit["delete_node_indices"]}
            if not nodes & occupied:
                squeeze_units.append(unit)
                occupied.update(nodes)
                continue
        reasons[squeeze_reason] += 1
    units = predicate_units + squeeze_units
    return {
        "task": task,
        "authority_sha256": sha256(data),
        "opset": default_opset(model),
        "reduction_count": reduction_count,
        "binary_reduction_count": binary_reduction_count,
        "predicate_units": predicate_units,
        "squeeze_units": squeeze_units,
        "units": units,
        "reason_counts": dict(sorted(reasons.items())),
        "candidate_possible": bool(units),
    }


def replace_reduction_kind(
    node: onnx.NodeProto,
    desired: str,
    axes: list[int] | None,
    opset: int,
    reduce_all: bool = False,
) -> None:
    if node.op_type == desired and not reduce_all:
        return
    keepdims = int_attr(node, "keepdims", 1)
    data_input = node.input[0]
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute("keepdims", 0 if reduce_all else keepdims)])
    node.op_type = desired
    if reduce_all:
        del node.input[:]
        node.input.extend([data_input])
    elif opset >= 18:
        # Axes input representation is shared by Sum/Max/Min at these opsets.
        # If the source had no axes input, absence continues to mean reduce-all.
        pass
    else:
        del node.input[:]
        node.input.extend([data_input])
        if axes is not None:
            node.attribute.extend([helper.make_attribute("axes", axes)])


def prune_unreachable(model: onnx.ModelProto) -> None:
    live = {value.name for value in model.graph.output}
    keep: set[int] = set()
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(name and name in live for name in node.output):
            keep.add(index)
            live.update(name for name in node.input if name)
    nodes = [copy.deepcopy(node) for index, node in enumerate(model.graph.node) if index in keep]
    node_outputs = {name for node in nodes for name in node.output if name}
    graph_inputs = {value.name for value in model.graph.input}
    initializers = [
        copy.deepcopy(item)
        for item in model.graph.initializer
        if item.name in live or item.name in graph_inputs
    ]
    value_info = [
        copy.deepcopy(value)
        for value in model.graph.value_info
        if value.name in node_outputs
    ]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)


def build_candidate(data: bytes, units: list[dict[str, Any]]) -> tuple[bytes, dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    delete_nodes = {index for unit in units for index in unit["delete_node_indices"]}
    delete_nodes.update(
        unit["carrier_cast_index"]
        for unit in units
        if unit.get("delete_carrier_cast") and unit.get("carrier_cast_index") is not None
    )
    rewire_casts: dict[int, str] = {}
    for unit in units:
        for index in unit.get("rewire_cast_indices", []):
            rewire_casts[index] = unit["old_reduction_output"]

    for unit in units:
        node = model.graph.node[unit["reduction_index"]]
        if unit["kind"] == "predicate_fusion":
            replace_reduction_kind(
                node,
                unit["desired_reduction"],
                unit["axes"],
                unit["opset"],
            )
            if unit["mode"] == "bool_reduce_fusion":
                node.input[0] = unit["bool_source"]
            node.output[0] = unit["target_output"]
            node.doc_string = f"exact binary {unit['predicate']} predicate fusion"
        else:
            replace_reduction_kind(
                node,
                unit["desired_reduction"],
                unit["old_axes"],
                unit["opset"],
                reduce_all=True,
            )
            node.output[0] = unit["target_output"]
            node.doc_string = "exact singleton-axis Reduce/Squeeze fusion"
    for index, input_name in rewire_casts.items():
        model.graph.node[index].input[0] = input_name

    remaining = [
        copy.deepcopy(node)
        for index, node in enumerate(model.graph.node)
        if index not in delete_nodes
    ]
    del model.graph.node[:]
    model.graph.node.extend(remaining)
    before_nodes = len(remaining)
    before_params = scoring.calculate_params(model)
    prune_unreachable(model)
    metadata = {
        "unit_count": len(units),
        "delete_original_node_indices": sorted(delete_nodes),
        "nodes_after_rewrite_before_prune": before_nodes,
        "nodes_after_prune": len(model.graph.node),
        "dead_nodes_pruned": before_nodes - len(model.graph.node),
        "params_before_prune": before_params,
        "params_after_prune": scoring.calculate_params(model),
    }
    return model.SerializeToString(), metadata


def declared_cost(data: bytes) -> dict[str, int] | None:
    model = onnx.load_model_from_string(data)
    params = scoring.calculate_params(model)
    if params is None:
        return None
    shapes, types = tensor_info(model)
    memory = 0
    for node in model.graph.node:
        for output in node.output:
            if not output or output == "output" or output not in shapes or output not in types:
                if output and output != "output":
                    return None
                continue
            memory += math.prod(shapes[output]) * np.dtype(
                helper.tensor_dtype_to_np_dtype(types[output])
            ).itemsize
    return {"memory_bytes": int(memory), "params": int(params), "cost": int(memory + params)}


def choose_units(data: bytes, units: list[dict[str, Any]]):
    authority = declared_cost(data)
    if not units:
        return [], {"authority_declared_cost": authority, "subsets_evaluated": 0}
    subsets = (
        itertools.chain.from_iterable(
            itertools.combinations(range(len(units)), count)
            for count in range(1, len(units) + 1)
        )
        if len(units) <= 16
        else [tuple(range(len(units)))]
    )
    best_indices: tuple[int, ...] = ()
    best_cost = None
    evaluated = 0
    for subset in subsets:
        evaluated += 1
        candidate, _ = build_candidate(data, [units[index] for index in subset])
        cost = declared_cost(candidate)
        if cost is None:
            continue
        if best_cost is None or cost["cost"] < best_cost["cost"] or (
            cost["cost"] == best_cost["cost"] and len(subset) > len(best_indices)
        ):
            best_indices, best_cost = tuple(subset), cost
    return [units[index] for index in best_indices], {
        "authority_declared_cost": authority,
        "candidate_declared_cost": best_cost,
        "selected_unit_indices": list(best_indices),
        "subsets_evaluated": evaluated,
    }


def synthetic_self_tests() -> list[dict[str, Any]]:
    cases = [
        ("sum_gt_zero_any", "ReduceSum", "Greater", 4, 0, 0, False, "Any"),
        ("sum_ge_one_any", "ReduceSum", "GreaterOrEqual", 4, 1, 0, False, "Any"),
        ("sum_eq_full_all", "ReduceSum", "Equal", 4, 4, 0, False, "All"),
        ("max_gt_zero_any", "ReduceMax", "Greater", 4, 0, 0, False, "Any"),
        ("min_eq_one_all", "ReduceMin", "Equal", 4, 1, 0, False, "All"),
        ("reverse_operand_any", "ReduceSum", "Less", 4, 0, 1, False, "Any"),
        ("not_sum_eq_zero_any", "ReduceSum", "Equal", 4, 0, 0, True, "Any"),
        ("sum_eq_one_reject", "ReduceSum", "Equal", 4, 1, 0, False, None),
    ]
    results = []
    for name, reduction_op, compare_op, count, threshold, slot, invert, expected in cases:
        reduction = helper.make_node(reduction_op, ["x", "axes"], ["r"], keepdims=0)
        comparison_inputs = ["r", "c"] if slot == 0 else ["c", "r"]
        comparison = helper.make_node(compare_op, comparison_inputs, ["p"])
        result = classify_predicate(
            reduction,
            comparison,
            count,
            {"c": np.asarray(threshold, dtype=np.int64)},
            invert,
        )
        actual = result["predicate"] if result else None
        results.append(
            {"name": name, "expected": expected, "actual": actual, "passed": actual == expected}
        )
    if not all(row["passed"] for row in results):
        raise AssertionError(results)
    return results


def full_validation(data: bytes) -> dict[str, Any]:
    row = {"checker_full": False, "shape_inference_strict_data_prop": False}
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        row["checker_full"] = True
    except Exception as exc:  # noqa: BLE001
        row["checker_error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        row["shape_inference_strict_data_prop"] = True
    except Exception as exc:  # noqa: BLE001
        row["shape_inference_error"] = f"{type(exc).__name__}: {exc}"
    return row


def score(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"reduction_predicate_{task:03d}_{label}_") as wd:
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
            if row["reduction_count"]:
                selected, selection = choose_units(data, row["units"])
                row["selected_units"] = selected
                row["selection"] = selection
                row["candidate_possible"] = bool(selected)
                discoveries.append(row)

    candidates = []
    for discovery in discoveries:
        if not discovery["candidate_possible"]:
            continue
        task = int(discovery["task"])
        authority = payloads[task]
        candidate, build = build_candidate(authority, discovery["selected_units"])
        path = CANDIDATES / f"task{task:03d}_reduction_predicate.onnx"
        onnx.save_model(onnx.load_model_from_string(candidate), path)
        candidate = path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "authority_sha256": sha256(authority),
            "candidate_sha256": sha256(candidate),
            "discovery": discovery,
            "build_metadata": build,
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
        base, cand = row["authority_score"], row["candidate_score"]
        row["strictly_lower"] = bool(base and cand and cand["cost"] < base["cost"])
        full_ok = row["full_validation"]["checker_full"] and row["full_validation"][
            "shape_inference_strict_data_prop"
        ]
        if not row["strictly_lower"]:
            row["decision"] = "REJECT_NOT_STRICTLY_LOWER_OR_UNSCORABLE"
        elif not full_ok or not row["strict"]["pass"] or not row["strict"].get("conv_bias_ub0"):
            row["decision"] = "REJECT_FULL_STRICT_SCHEMA_OR_UB"
        else:
            known = known_raw_four(task, authority, candidate)
            row["known_raw_four"] = known
            row["known_raw_four_pass"] = known_pass(known)
            if not cand.get("correct"):
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
        candidates.append(row)
        print(
            f"task{task:03d} units={len(discovery['selected_units'])} "
            f"cost={base}->{cand} decision={row['decision']}",
            flush=True,
        )

    accepted = [row for row in candidates if row["decision"] == "ACCEPT"]
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "tasks_scanned": 400,
        "synthetic_self_tests": self_tests,
        "tasks_with_reduction": len(discoveries),
        "reduction_nodes": sum(row["reduction_count"] for row in discoveries),
        "binary_reduction_nodes": sum(row["binary_reduction_count"] for row in discoveries),
        "predicate_units": sum(len(row["predicate_units"]) for row in discoveries),
        "squeeze_units": sum(len(row["squeeze_units"]) for row in discoveries),
        "candidate_tasks": len(candidates),
        "discoveries": discoveries,
        "candidates": candidates,
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
        "decision": "ACCEPT" if accepted else "NO_SAFE_REDUCTION_PREDICATE_WINNER",
    }
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks_with_reduction": report["tasks_with_reduction"],
                "reduction_nodes": report["reduction_nodes"],
                "binary_reduction_nodes": report["binary_reduction_nodes"],
                "predicate_units": report["predicate_units"],
                "squeeze_units": report["squeeze_units"],
                "candidate_tasks": report["candidate_tasks"],
                "accepted": len(accepted),
                "decision": report["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

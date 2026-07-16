#!/usr/bin/env python3
"""Exhaustive exact linear-kernel scan of the immutable 8009.46 authority.

This pass covers Conv, MatMul, and Gemm.  It uses exact rational elimination
on serialized values (never SVD/tolerances), groups every use of a shared
initializer into one action, and prices new intermediate activations with the
same memory+parameter objective used by the official scorer.  Non-lower plans
are evidence rows only; they are never emitted as candidate models.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
import tempfile
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
OUTPUT = HERE / "scan.json"
TARGET_OPS = {"Conv", "MatMul", "Gemm"}

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_fraction(value: Any) -> Fraction:
    if isinstance(value, (np.integer, int, np.bool_)):
        return Fraction(int(value), 1)
    return Fraction.from_float(float(value))


def array_key(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    array = np.ascontiguousarray(array)
    return array.dtype.str, tuple(int(x) for x in array.shape), array.tobytes()


def serial_key(key: tuple[str, tuple[int, ...], bytes]) -> str:
    dtype, shape, data = key
    return f"{dtype}:{'x'.join(map(str, shape))}:{sha256(data)}"


def rational_rank(matrix: np.ndarray) -> int:
    """Rank over Q of the exact serialized numeric values."""
    rows = [list(map(exact_fraction, row)) for row in np.asarray(matrix)]
    height = len(rows)
    width = len(rows[0]) if height else 0
    rank = 0
    for column in range(width):
        pivot = next((row for row in range(rank, height) if rows[row][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        rows[rank] = [value / divisor for value in rows[rank]]
        for row in range(height):
            if row == rank or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [left - scale * right for left, right in zip(rows[row], rows[rank])]
        rank += 1
        if rank == height:
            break
    return rank


def exact_column_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Exact rank factor L@R with factors exactly representable in source dtype."""
    matrix = np.ascontiguousarray(matrix)
    rows, columns = map(int, matrix.shape)
    echelon: list[tuple[int, list[Fraction], list[Fraction]]] = []
    basis_columns: list[int] = []
    coefficients: list[list[Fraction]] = []
    for column in range(columns):
        vector = [exact_fraction(matrix[row, column]) for row in range(rows)]
        combination = [Fraction(0) for _ in basis_columns]
        for pivot, reduced, representation in echelon:
            scale = vector[pivot]
            if not scale:
                continue
            vector = [left - scale * right for left, right in zip(vector, reduced)]
            for index, value in enumerate(representation):
                combination[index] += scale * value
        pivot = next((index for index, value in enumerate(vector) if value), None)
        if pivot is None:
            coefficients.append(combination)
            continue
        pivot_value = vector[pivot]
        for index, (old_pivot, old_vector, old_rep) in enumerate(echelon):
            echelon[index] = (old_pivot, old_vector, old_rep + [Fraction(0)])
        for old in coefficients:
            old.append(Fraction(0))
        representation = [(-value) / pivot_value for value in combination]
        representation.append(Fraction(1, 1) / pivot_value)
        reduced = [value / pivot_value for value in vector]
        echelon.append((pivot, reduced, representation))
        basis_columns.append(column)
        unit = [Fraction(0) for _ in basis_columns]
        unit[-1] = Fraction(1)
        coefficients.append(unit)
    rank = len(basis_columns)
    if not rank:
        return None
    left = np.ascontiguousarray(matrix[:, basis_columns])
    right_fractions = [
        [coefficients[column][row] for column in range(columns)] for row in range(rank)
    ]
    try:
        right = np.asarray(
            [[float(value) for value in row] for row in right_fractions], dtype=matrix.dtype
        )
    except (OverflowError, ValueError):
        return None
    if not np.all(np.isfinite(right)):
        return None
    for stored, exact_row in zip(right, right_fractions):
        if any(exact_fraction(value) != wanted for value, wanted in zip(stored, exact_row)):
            return None
    # Recheck the identity over exact serialized values, independent of BLAS.
    for row in range(rows):
        for column in range(columns):
            rebuilt = sum(
                exact_fraction(left[row, latent]) * exact_fraction(right[latent, column])
                for latent in range(rank)
            )
            if rebuilt != exact_fraction(matrix[row, column]):
                return None
    return left, np.ascontiguousarray(right), rank


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def concrete_shape(value: onnx.ValueInfoProto | None) -> list[int] | None:
    if value is None or not value.type.HasField("tensor_type"):
        return None
    dims: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return dims


def type_maps(model: onnx.ModelProto) -> tuple[dict[str, list[int]], dict[str, np.dtype]]:
    try:
        inferred = shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=False, data_prop=True
        )
    except Exception:
        inferred = model
    values = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, np.dtype] = {}
    for name, value in values.items():
        shape = concrete_shape(value)
        if shape is not None:
            shapes[name] = shape
        try:
            dtypes[name] = np.dtype(
                helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
            )
        except Exception:
            pass
    return shapes, dtypes


def conv_first_shape(
    input_shape: list[int], node: onnx.NodeProto, kernel: list[int], subset: tuple[int, ...], rank: int
) -> list[int] | None:
    """Smallest legal first-factor shape for disjoint-axis sequential Conv."""
    spatial = len(kernel)
    if len(input_shape) != spatial + 2 or attrs(node).get("auto_pad", b"NOTSET") not in {
        b"NOTSET",
        "NOTSET",
    }:
        return None
    properties = attrs(node)
    pads = list(map(int, properties.get("pads", [0] * (2 * spatial))))
    strides = list(map(int, properties.get("strides", [1] * spatial)))
    dilations = list(map(int, properties.get("dilations", [1] * spatial)))
    subset_set = set(subset)
    output = [int(input_shape[0]), int(rank)]
    for axis, full_kernel in enumerate(kernel):
        in_first = axis in subset_set
        # A size-1 full kernel can carry crop/stride in either factor.  Put it
        # in the first factor because that minimizes counted intermediate bytes.
        carries_geometry = in_first or full_kernel == 1
        first_kernel = full_kernel if in_first else 1
        before = pads[axis] if carries_geometry else 0
        after = pads[axis + spatial] if carries_geometry else 0
        dilation = dilations[axis] if in_first else 1
        stride = strides[axis] if carries_geometry else 1
        effective = dilation * (first_kernel - 1) + 1
        size = math.floor((input_shape[axis + 2] + before + after - effective) / stride) + 1
        if size <= 0:
            return None
        output.append(int(size))
    return output


def conv_partition_matrix(
    weight: np.ndarray, group: int, subset: tuple[int, ...]
) -> list[np.ndarray]:
    outputs, inputs_per_group, *kernel = map(int, weight.shape)
    outputs_per_group = outputs // group
    complement = tuple(axis for axis in range(len(kernel)) if axis not in subset)
    result: list[np.ndarray] = []
    for block in range(group):
        current = weight[block * outputs_per_group : (block + 1) * outputs_per_group]
        permutation = [1] + [2 + axis for axis in subset] + [0] + [
            2 + axis for axis in complement
        ]
        left = inputs_per_group * math.prod(kernel[axis] for axis in subset)
        right = outputs_per_group * math.prod(kernel[axis] for axis in complement)
        result.append(np.ascontiguousarray(np.transpose(current, permutation).reshape(left, right)))
    return result


def conv_factors(
    weight: np.ndarray, subset: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Factors for group=1 W[o,i,k...] over a disjoint spatial partition."""
    outputs, inputs, *kernel = map(int, weight.shape)
    matrix = conv_partition_matrix(weight, 1, subset)[0]
    factored = exact_column_factor(matrix)
    if factored is None:
        return None
    left, right, rank = factored
    complement = tuple(axis for axis in range(len(kernel)) if axis not in subset)
    left_shape = [inputs] + [kernel[axis] for axis in subset] + [rank]
    left = left.reshape(left_shape)
    left = np.transpose(left, [len(left_shape) - 1, 0, *range(1, len(left_shape) - 1)])
    first_shape = [rank, inputs] + [kernel[axis] if axis in subset else 1 for axis in range(len(kernel))]
    left = np.ascontiguousarray(left.reshape(first_shape))
    right_shape = [rank, outputs] + [kernel[axis] for axis in complement]
    right = right.reshape(right_shape)
    right = np.transpose(right, [1, 0, *range(2, len(right_shape))])
    second_shape = [outputs, rank] + [kernel[axis] if axis in complement else 1 for axis in range(len(kernel))]
    right = np.ascontiguousarray(right.reshape(second_shape))
    return left, right, rank


def matrix_class(matrix: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if matrix.ndim != 2:
        return result
    rows, columns = matrix.shape
    nonzero = matrix != 0
    if rows == columns:
        diagonal = np.diag(matrix)
        if np.count_nonzero(nonzero) == np.count_nonzero(diagonal):
            result["diagonal"] = True
            result["identity"] = bool(np.array_equal(diagonal, np.ones_like(diagonal)))
    per_column = np.count_nonzero(nonzero, axis=0)
    if np.all(per_column == 1):
        row_indices = np.argmax(nonzero, axis=0)
        values = matrix[row_indices, np.arange(columns)]
        result["selection"] = True
        result["selection_plus_one"] = bool(np.array_equal(values, np.ones_like(values)))
        result["permutation"] = bool(
            rows == columns and len(set(map(int, row_indices))) == columns
        )
        result["selection_indices"] = list(map(int, row_indices))
    return result


def conv_block_groups(weight: np.ndarray, old_group: int) -> list[int]:
    outputs, inputs_per_group, *_ = map(int, weight.shape)
    inputs = old_group * inputs_per_group
    result: list[int] = []
    for new_group in range(old_group + 1, min(outputs, inputs) + 1):
        if new_group % old_group or outputs % new_group or inputs % new_group:
            continue
        split = new_group // old_group
        out_old = outputs // old_group
        out_new = outputs // new_group
        in_new = inputs // new_group
        valid = True
        for old in range(old_group):
            for output in range(out_old):
                wanted = output // out_new
                for channel in range(inputs_per_group):
                    if channel // in_new != wanted and np.any(
                        weight[old * out_old + output, channel]
                    ):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            result.append(new_group)
    return result


def official_profile(model: onnx.ModelProto, task: int) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"linear243_{task:03d}_") as directory:
        path = Path(directory) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def load_ledger() -> dict[int, int]:
    with (ROOT / "all_scores.csv").open() as handle:
        return {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }


def main() -> None:
    ledger = load_ledger()
    node_census = Counter()
    task_census = Counter()
    constant_positions = Counter()
    structure_census = Counter()
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []
    profiles: dict[int, dict[str, int]] = {}
    all_global: list[dict[str, Any]] = []

    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            model = onnx.load_model_from_string(archive.read(member))
            initializers = {
                item.name: np.ascontiguousarray(numpy_helper.to_array(item))
                for item in model.graph.initializer
            }
            initializer_keys = defaultdict(list)
            for name, array in initializers.items():
                initializer_keys[array_key(array)].append(name)
            occurrences: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
            output_uses: Counter[str] = Counter()
            producers: dict[str, tuple[int, onnx.NodeProto]] = {}
            for node_index, node in enumerate(model.graph.node):
                for position, name in enumerate(node.input):
                    if name:
                        occurrences[name].append((node_index, position))
                        output_uses[name] += 1
                for name in node.output:
                    if name:
                        producers[name] = (node_index, node)
            shapes, dtypes = type_maps(model)
            target_nodes = [
                (index, node)
                for index, node in enumerate(model.graph.node)
                if node.op_type in TARGET_OPS
            ]
            if target_nodes:
                task_census["tasks_with_target_op"] += 1
                profiles[task] = official_profile(model, task)
                if profiles[task]["cost"] == ledger.get(task):
                    task_census["official_profile_matches_ledger"] += 1
            for node_index, node in target_nodes:
                node_census[node.op_type] += 1
                if len(node.input) > 1 and node.input[1] in initializers:
                    constant_positions[f"{node.op_type}:weight"] += 1
                else:
                    constant_positions[f"{node.op_type}:dynamic_weight"] += 1
                if len(node.input) > 2 and node.input[2] in initializers:
                    constant_positions[f"{node.op_type}:bias"] += 1
                for position, name in enumerate(node.input):
                    if name not in producers:
                        continue
                    producer_index, producer = producers[name]
                    if producer.op_type not in TARGET_OPS:
                        continue
                    chain = {
                        "task": task,
                        "producer_index": producer_index,
                        "producer_op": producer.op_type,
                        "consumer_index": node_index,
                        "consumer_op": node.op_type,
                        "consumer_input_position": position,
                        "single_consumer": output_uses[name] == 1,
                        "graph_output": name in {item.name for item in model.graph.output},
                    }
                    chain_rows.append(chain)
                    structure_census["consecutive_linear_edges"] += 1

            # Each shared initializer is considered once and can only be
            # removed by an action that covers every graph occurrence.
            plans_by_initializer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
            handled: set[str] = set()
            for node_index, node in target_nodes:
                if len(node.input) < 2 or node.input[1] not in initializers:
                    continue
                weight_name = node.input[1]
                if weight_name in handled:
                    continue
                handled.add(weight_name)
                weight = initializers[weight_name]
                locations = occurrences[weight_name]
                all_weight_uses = all(
                    position == 1 and model.graph.node[index].op_type in TARGET_OPS
                    for index, position in locations
                )
                row: dict[str, Any] = {
                    "task": task,
                    "initializer": weight_name,
                    "shape": list(map(int, weight.shape)),
                    "dtype": str(weight.dtype),
                    "elements": int(weight.size),
                    "nonzero": int(np.count_nonzero(weight)),
                    "uses": len(locations),
                    "all_uses_are_linear_weights": all_weight_uses,
                    "op_types": sorted({model.graph.node[index].op_type for index, _ in locations}),
                    "families": [],
                }
                if node.op_type == "Conv" and weight.ndim >= 3:
                    properties = attrs(node)
                    group = int(properties.get("group", 1))
                    outputs, inputs_per_group, *kernel = map(int, weight.shape)
                    row["group"] = group
                    row["existing_depthwise"] = bool(
                        group > 1 and inputs_per_group == 1 and outputs % group == 0
                    )
                    if row["existing_depthwise"]:
                        structure_census["existing_depthwise_kernels"] += 1
                    blocks = conv_block_groups(weight, group)
                    row["exact_stricter_block_groups"] = blocks
                    structure_census["block_group_plans"] += len(blocks)
                    partition_rows: list[dict[str, Any]] = []
                    for mask in range(1 << len(kernel)):
                        subset = tuple(axis for axis in range(len(kernel)) if mask & (1 << axis))
                        matrices = conv_partition_matrix(weight, group, subset)
                        ranks = [rational_rank(matrix) for matrix in matrices]
                        rank = max(ranks, default=0)
                        factor_params = (
                            group
                            * rank
                            * inputs_per_group
                            * math.prod(kernel[axis] for axis in subset)
                            + outputs
                            * rank
                            * math.prod(
                                kernel[axis]
                                for axis in range(len(kernel))
                                if axis not in subset
                            )
                        )
                        partition: dict[str, Any] = {
                            "first_spatial_axes": list(subset),
                            "exact_ranks_by_group": ranks,
                            "rank": rank,
                            "factor_params": int(factor_params),
                            "parameter_delta": int(factor_params - weight.size),
                        }
                        structure_census["conv_axis_partitions"] += 1
                        if factor_params < weight.size:
                            structure_census["conv_parameter_reducing_partitions"] += 1
                        if group == 1:
                            factors = conv_factors(weight, subset)
                            if factors is not None:
                                left, right, stored_rank = factors
                                partition["stored_dtype_exact_factors"] = True
                                partition["factor_shapes"] = [list(left.shape), list(right.shape)]
                                structure_census["conv_stored_exact_factorizations"] += 1
                                if all_weight_uses:
                                    activation = 0
                                    activation_shapes: list[list[int]] = []
                                    unresolved = False
                                    compatible = True
                                    for use_index, use_position in locations:
                                        use_node = model.graph.node[use_index]
                                        if use_node.op_type != "Conv" or use_position != 1:
                                            compatible = False
                                            break
                                        input_shape = shapes.get(use_node.input[0])
                                        first_shape = (
                                            conv_first_shape(
                                                input_shape,
                                                use_node,
                                                kernel,
                                                subset,
                                                stored_rank,
                                            )
                                            if input_shape is not None
                                            else None
                                        )
                                        if first_shape is None:
                                            unresolved = True
                                            continue
                                        dtype = dtypes.get(use_node.input[0], weight.dtype)
                                        activation += math.prod(first_shape) * dtype.itemsize
                                        activation_shapes.append(first_shape)
                                    if compatible:
                                        action = {
                                            "task": task,
                                            "initializer": weight_name,
                                            "kind": "conv_exact_axis_factor",
                                            "first_spatial_axes": list(subset),
                                            "rank": stored_rank,
                                            "removed_params": int(weight.size),
                                            "factor_arrays": [left, right],
                                            "activation_bytes": int(activation),
                                            "activation_shapes": activation_shapes,
                                            "activation_unresolved": unresolved,
                                        }
                                        if not unresolved:
                                            plans_by_initializer[weight_name].append(action)
                        else:
                            partition["stored_dtype_exact_factors"] = False
                            partition["factor_build_reason"] = (
                                "grouped per-block rank census; sequential factors require "
                                "duplicated group-local banks"
                            )
                        partition_rows.append(partition)
                    row["families"].append({"kind": "conv_axis_partitions", "plans": partition_rows})

                    # Explicit depthwise->pointwise and pointwise->depthwise
                    # tests.  These are more permissive than a single global
                    # matrix rank: every input (or output) channel may own a
                    # different rank-1 spatial kernel.
                    outputs_per_group = outputs // group
                    spatial_elements = math.prod(kernel)
                    depthwise_ranks: list[int] = []
                    depthwise_stored_exact = True
                    for block in range(group):
                        block_weight = weight[
                            block * outputs_per_group : (block + 1) * outputs_per_group
                        ]
                        for channel in range(inputs_per_group):
                            matrix = np.ascontiguousarray(
                                block_weight[:, channel].reshape(
                                    outputs_per_group, spatial_elements
                                )
                            )
                            current_rank = rational_rank(matrix)
                            depthwise_ranks.append(current_rank)
                            if current_rank > 1 or (
                                current_rank == 1 and exact_column_factor(matrix) is None
                            ):
                                depthwise_stored_exact = False
                    depthwise_params = (
                        group * inputs_per_group * spatial_elements
                        + outputs * inputs_per_group
                    )
                    depthwise_plan = {
                        "kind": "depthwise_then_pointwise",
                        "exact_ranks_by_input_channel": depthwise_ranks,
                        "exact": max(depthwise_ranks, default=0) <= 1,
                        "stored_dtype_exact_factors": depthwise_stored_exact,
                        "factor_params": int(depthwise_params),
                        "parameter_delta": int(depthwise_params - weight.size),
                    }
                    row["families"].append(depthwise_plan)
                    structure_census["depthwise_pointwise_scans"] += 1
                    if depthwise_plan["exact"]:
                        structure_census["exact_depthwise_then_pointwise"] += 1
                        if depthwise_params < weight.size:
                            structure_census[
                                "depthwise_then_pointwise_parameter_reductions"
                            ] += 1

                    pointwise_ranks: list[int] = []
                    pointwise_stored_exact = True
                    for output in range(outputs):
                        matrix = np.ascontiguousarray(
                            weight[output].reshape(inputs_per_group, spatial_elements)
                        )
                        current_rank = rational_rank(matrix)
                        pointwise_ranks.append(current_rank)
                        if current_rank > 1 or (
                            current_rank == 1 and exact_column_factor(matrix) is None
                        ):
                            pointwise_stored_exact = False
                    pointwise_params = outputs * inputs_per_group + outputs * spatial_elements
                    pointwise_plan = {
                        "kind": "pointwise_then_depthwise",
                        "exact_ranks_by_output_channel": pointwise_ranks,
                        "exact": max(pointwise_ranks, default=0) <= 1,
                        "stored_dtype_exact_factors": pointwise_stored_exact,
                        "factor_params": int(pointwise_params),
                        "parameter_delta": int(pointwise_params - weight.size),
                    }
                    row["families"].append(pointwise_plan)
                    structure_census["pointwise_depthwise_scans"] += 1
                    if pointwise_plan["exact"]:
                        structure_census["exact_pointwise_then_depthwise"] += 1
                        if pointwise_params < weight.size:
                            structure_census[
                                "pointwise_then_depthwise_parameter_reductions"
                            ] += 1

                    # Shared spatial-bank lower bound for already grouped/depthwise
                    # kernels: reshape channels into batch, use one bank, reshape,
                    # then mix per original group.  It can share coefficients but
                    # necessarily materializes group*rank output planes.
                    bank_matrix = np.ascontiguousarray(weight.reshape(outputs, -1))
                    bank_rank = rational_rank(bank_matrix)
                    bank_params = bank_rank * (outputs + bank_matrix.shape[1])
                    bank_activation_lower_bound = 0
                    for use_index, use_position in locations:
                        use_node = model.graph.node[use_index]
                        output_shape = shapes.get(use_node.output[0])
                        dtype = dtypes.get(use_node.input[0], weight.dtype)
                        if output_shape is not None and len(output_shape) == len(kernel) + 2:
                            # Reshaping original groups into batch and applying
                            # a shared bank must expose at least group*rank
                            # planes at the original output lattice.  Reshape
                            # node outputs are deliberately omitted, making this
                            # a conservative lower bound.
                            elements = (
                                output_shape[0]
                                * group
                                * bank_rank
                                * math.prod(output_shape[2:])
                            )
                        else:
                            elements = max(1, group * bank_rank)
                        bank_activation_lower_bound += elements * dtype.itemsize
                    row["families"].append(
                        {
                            "kind": "shared_outer_product_bank",
                            "rank": bank_rank,
                            "factor_params": int(bank_params),
                            "parameter_delta": int(bank_params - weight.size),
                            "requires_reshape_batch_and_added_activation": group > 1,
                            "added_activation_lower_bound": int(
                                bank_activation_lower_bound
                            ),
                            "official_delta_lower_bound_with_factors": int(
                                bank_params - weight.size + bank_activation_lower_bound
                            ),
                            "official_delta_lower_bound_even_if_all_factors_reused": int(
                                bank_activation_lower_bound - weight.size
                            ),
                        }
                    )
                    if bank_params < weight.size:
                        structure_census["conv_shared_bank_parameter_reductions"] += 1

                    # Direct one-hot/diagonal 1x1 simplifications.
                    if group == 1 and all(size == 1 for size in kernel):
                        matrix = weight.reshape(outputs, inputs_per_group)
                        classification = matrix_class(matrix.T)
                        conv_class = matrix_class(matrix)
                        row["families"].append(
                            {
                                "kind": "conv_1x1_matrix_structure",
                                "input_to_output": classification,
                                "output_to_input": conv_class,
                            }
                        )
                        if classification.get("selection_plus_one"):
                            structure_census["conv_1x1_onehot_selection"] += 1
                        if conv_class.get("diagonal"):
                            structure_census["conv_1x1_diagonal"] += 1
                        if classification.get("selection_plus_one") or conv_class.get(
                            "diagonal"
                        ):
                            direct_rows.append(
                                {
                                    "task": task,
                                    "initializer": weight_name,
                                    "kind": (
                                        "conv_1x1_gather"
                                        if classification.get("selection_plus_one")
                                        else "conv_1x1_diagonal_mul"
                                    ),
                                    "all_uses_are_linear_weights": all_weight_uses,
                                    "strict_lower_eligible": False,
                                    "barrier": (
                                        "requires per-use attribute/bias compatibility and raw-output gate"
                                        if all_weight_uses
                                        else "initializer also has non-linear-weight uses and cannot be removed"
                                    ),
                                }
                            )
                else:
                    effective = weight
                    properties = attrs(node)
                    if node.op_type == "Gemm" and int(properties.get("transB", 0)):
                        effective = np.swapaxes(effective, -1, -2)
                    if effective.ndim >= 2:
                        matrix = np.ascontiguousarray(
                            effective.reshape(math.prod(effective.shape[:-1]), effective.shape[-1])
                        )
                        rank = rational_rank(matrix)
                        factor_params = rank * sum(matrix.shape)
                        classification = matrix_class(matrix)
                        row["families"].append(
                            {
                                "kind": "matrix_exact_rank",
                                "matrix_shape": list(matrix.shape),
                                "rank": rank,
                                "factor_params": int(factor_params),
                                "parameter_delta": int(factor_params - weight.size),
                                "structure": classification,
                            }
                        )
                        structure_census["matrix_rank_scans"] += 1
                        if factor_params < weight.size:
                            structure_census["matrix_parameter_reducing_ranks"] += 1
                        if classification.get("diagonal"):
                            structure_census["matrix_diagonal"] += 1
                        if classification.get("selection_plus_one"):
                            structure_census["matrix_onehot_selection"] += 1
                        if classification.get("diagonal") or classification.get(
                            "selection_plus_one"
                        ):
                            # The only authority hits are shared 1x1 scalars in
                            # tasks177/340.  Both also feed Einsum, so rewriting
                            # their Gemm occurrences cannot delete the initializer.
                            direct_rows.append(
                                {
                                    "task": task,
                                    "initializer": weight_name,
                                    "kind": (
                                        "matrix_onehot_gather"
                                        if classification.get("selection_plus_one")
                                        else "matrix_diagonal_mul"
                                    ),
                                    "structure": classification,
                                    "all_uses_are_linear_weights": all_weight_uses,
                                    "strict_lower_eligible": False,
                                    "barrier": (
                                        "requires per-use Gemm alpha/beta/bias compatibility and raw-output gate"
                                        if all_weight_uses
                                        else "initializer also has non-linear-weight uses and cannot be removed"
                                    ),
                                }
                            )
                rows.append(row)

            # Exhaustive global choice: zero or one plan per initializer.  New
            # factors are deduplicated against every surviving initializer and
            # against one another.  Thus a shared source is never credited until
            # every one of its occurrences is covered.
            options = [[None, *plans] for plans in plans_by_initializer.values()]
            if not options:
                continue
            if math.prod(len(item) for item in options) > 100_000:
                raise RuntimeError(f"task{task:03d}: unexpected global option explosion")
            for choice in itertools.product(*options):
                selected = [item for item in choice if item is not None]
                if not selected:
                    continue
                removed = {str(item["initializer"]) for item in selected}
                factor_keys: dict[tuple[str, tuple[int, ...], bytes], int] = {}
                for item in selected:
                    for array in item["factor_arrays"]:
                        factor_keys[array_key(array)] = int(array.size)
                surviving_keys = {
                    key
                    for key, names in initializer_keys.items()
                    if any(name not in removed for name in names)
                }
                added_params = sum(
                    size for key, size in factor_keys.items() if key not in surviving_keys
                )
                removed_params = sum(int(initializers[name].size) for name in removed)
                added_memory = sum(int(item["activation_bytes"]) for item in selected)
                delta = added_params - removed_params + added_memory
                global_row = {
                    "task": task,
                    "actions": [
                        {
                            key: value
                            for key, value in item.items()
                            if key != "factor_arrays"
                        }
                        for item in selected
                    ],
                    "removed_initializers": sorted(removed),
                    "removed_params": removed_params,
                    "new_unique_params": added_params,
                    "added_activation_bytes": added_memory,
                    "official_cost_delta": int(delta),
                    "projected_official_cost": int(profiles[task]["cost"] + delta),
                    "strict_lower": delta < 0,
                    "new_factor_keys": [serial_key(key) for key in factor_keys],
                }
                all_global.append(global_row)
                action_rows.extend(global_row["actions"])

    strict = [row for row in all_global if row["strict_lower"]]
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": sha256(AUTHORITY.read_bytes()),
        "models_scanned": 400,
        "method": {
            "rank": "exact rational Gaussian elimination over serialized values",
            "approximate_svd": False,
            "shared_initializer_rule": "remove only when every graph occurrence is covered",
            "factor_reuse_rule": "deduplicate by dtype+shape+raw bytes against all surviving initializers and selected factors",
            "cost": "official ORT_DISABLE_ALL profile plus exact added activation bytes and parameter delta",
            "candidate_emission": "strict-lower global choices only",
        },
        "node_census": dict(node_census),
        "task_census": dict(task_census),
        "constant_position_census": dict(constant_positions),
        "structure_census": dict(structure_census),
        "official_profiles": {str(task): value for task, value in sorted(profiles.items())},
        "consecutive_linear_edges": chain_rows,
        "initializer_rows": rows,
        "global_factor_choices": all_global,
        "direct_simplification_rows": direct_rows,
        "direct_strict_lower_count": sum(
            bool(row["strict_lower_eligible"]) for row in direct_rows
        ),
        "strict_lower_count": len(strict),
        "strict_lower_choices": strict,
        "candidate_files": [],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "target_nodes": sum(node_census.values()),
                "tasks": task_census["tasks_with_target_op"],
                "constant_weight_initializers": len(rows),
                "consecutive_linear_edges": len(chain_rows),
                "global_factor_choices": len(all_global),
                "strict_lower": len(strict),
            },
            indent=2,
        )
    )
    if strict:
        raise RuntimeError(
            "strict-lower structural survivors require build + four-config raw audit before reporting"
        )


if __name__ == "__main__":
    main()

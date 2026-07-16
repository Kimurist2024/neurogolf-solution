#!/usr/bin/env python3
"""Exact small-initializer CP/Kronecker factor scan for authority Einsums.

The scan is deliberately algebraic.  It never uses SVD or an approximate
rank tolerance.  Serialized finite initializer values are treated as exact
rationals, every proposed factorization is rebuilt in the stored dtype, and
the rebuilt byte string must equal the source byte string.  Candidates are
kept only when full ONNX checking and strict data-propagating shape inference
pass, counted memory does not increase, and official-compatible cost falls.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import string
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
CANDIDATES = HERE / "candidates"
MAX_ELEMENTS = 4096
EXCLUDED_TASKS = {310}
LABELS = string.ascii_letters

sys.path.insert(0, str(ROOT / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_bytes(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def array_key(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return array.dtype.str, tuple(int(x) for x in array.shape), array_bytes(array)


def equation_attr(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def equation(node: onnx.NodeProto) -> str:
    return equation_attr(node).s.decode("ascii")


def split_equation(node: onnx.NodeProto) -> tuple[list[str], str]:
    text = equation(node)
    if "->" not in text or "..." in text:
        raise ValueError(f"unsupported Einsum equation {text!r}")
    lhs, rhs = text.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(node.input):
        raise ValueError("Einsum input/equation arity mismatch")
    return terms, rhs


def set_equation(node: onnx.NodeProto, terms: list[str], rhs: str) -> None:
    equation_attr(node).s = (",".join(terms) + "->" + rhs).encode("ascii")


def exact_fraction(value: Any) -> Fraction:
    if isinstance(value, (np.integer, int)):
        return Fraction(int(value), 1)
    return Fraction.from_float(float(value))


def finite_numeric(array: np.ndarray) -> bool:
    return array.dtype.kind in "fiu" and bool(np.all(np.isfinite(array)))


def rebuild_separable(factors: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    dtype = factors[0].dtype
    result = factors[0].reshape((shape[0],) + (1,) * (len(shape) - 1))
    for axis in range(1, len(shape)):
        view = [1] * len(shape)
        view[axis] = shape[axis]
        result = np.asarray(result * factors[axis].reshape(view), dtype=dtype)
    return np.ascontiguousarray(result)


def exact_separable(array: np.ndarray) -> list[np.ndarray] | None:
    """Return fully axis-separable factors with byte-identical reconstruction."""
    if array.ndim < 2 or not np.any(array):
        return None
    if array.size <= sum(array.shape):
        return None
    coordinates = np.argwhere(array != 0)
    for coordinate_array in coordinates[: min(512, len(coordinates))]:
        coordinate = tuple(int(x) for x in coordinate_array)
        pivot = array[coordinate]
        for raw_axis in range(array.ndim):
            factors: list[np.ndarray] = []
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                for axis in range(array.ndim):
                    selection: list[int | slice] = list(coordinate)
                    selection[axis] = slice(None)
                    factor = np.asarray(array[tuple(selection)], dtype=array.dtype)
                    if axis != raw_axis:
                        factor = np.asarray(factor / pivot, dtype=array.dtype)
                    factors.append(np.ascontiguousarray(factor))
            rebuilt = rebuild_separable(factors, tuple(array.shape))
            if array_bytes(rebuilt) == array_bytes(array):
                return factors
    return None


def exact_column_factor(
    matrix: np.ndarray, max_rank: int
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Exact rational column elimination followed by a stored-dtype byte gate."""
    rows, cols = (int(matrix.shape[0]), int(matrix.shape[1]))
    echelon: list[tuple[int, list[Fraction], list[Fraction]]] = []
    basis_columns: list[int] = []
    coefficients: list[list[Fraction]] = []

    for col_index in range(cols):
        vector = [exact_fraction(matrix[row, col_index]) for row in range(rows)]
        combination = [Fraction(0) for _ in basis_columns]
        for pivot, reduced, representation in echelon:
            factor = vector[pivot]
            if not factor:
                continue
            vector = [left - factor * right for left, right in zip(vector, reduced)]
            for index, value in enumerate(representation):
                combination[index] += factor * value
        pivot = next((index for index, value in enumerate(vector) if value), None)
        if pivot is None:
            coefficients.append(combination)
            continue
        if len(basis_columns) >= max_rank:
            return None
        pivot_value = vector[pivot]
        for old_index, (old_pivot, old_vector, old_rep) in enumerate(echelon):
            echelon[old_index] = (old_pivot, old_vector, old_rep + [Fraction(0)])
        for old in coefficients:
            old.append(Fraction(0))
        new_representation = [(-value) / pivot_value for value in combination]
        new_representation.append(Fraction(1, 1) / pivot_value)
        reduced = [value / pivot_value for value in vector]
        echelon.append((pivot, reduced, new_representation))
        basis_columns.append(col_index)
        unit = [Fraction(0) for _ in basis_columns]
        unit[-1] = Fraction(1)
        coefficients.append(unit)

    rank = len(basis_columns)
    if rank == 0 or rank > max_rank:
        return None
    left = np.ascontiguousarray(matrix[:, basis_columns])
    right = np.asarray(
        [[float(coefficients[col][row]) for col in range(cols)] for row in range(rank)],
        dtype=matrix.dtype,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        rebuilt = np.asarray(np.einsum("ir,rj->ij", left, right, optimize=False), dtype=matrix.dtype)
    if array_bytes(rebuilt) != array_bytes(matrix):
        return None
    return left, np.ascontiguousarray(right), rank


def partition_plans(name: str, array: np.ndarray) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    axes = tuple(range(array.ndim))
    # Axis 0 is fixed to the left so complementary partitions are not repeated.
    for left_size in range(1, array.ndim):
        for extra in itertools.combinations(axes[1:], left_size - 1):
            left_axes = (0, *extra)
            right_axes = tuple(axis for axis in axes if axis not in left_axes)
            left_shape = tuple(int(array.shape[axis]) for axis in left_axes)
            right_shape = tuple(int(array.shape[axis]) for axis in right_axes)
            left_elements = math.prod(left_shape)
            right_elements = math.prod(right_shape)
            max_rank = min(left_elements, right_elements, (array.size - 1) // (left_elements + right_elements))
            if max_rank < 1:
                continue
            transposed = np.transpose(array, left_axes + right_axes)
            matrix = np.ascontiguousarray(transposed.reshape(left_elements, right_elements))
            factored = exact_column_factor(matrix, max_rank)
            if factored is None:
                continue
            left, right, rank = factored
            if rank == 1:
                left_factor = np.ascontiguousarray(left[:, 0].reshape(left_shape))
                right_factor = np.ascontiguousarray(right[0, :].reshape(right_shape))
            else:
                left_factor = np.ascontiguousarray(left.reshape(left_shape + (rank,)))
                right_factor = np.ascontiguousarray(right.reshape((rank,) + right_shape))
            factor_params = int(left_factor.size + right_factor.size)
            if factor_params >= array.size:
                continue
            result.append(
                {
                    "kind": "kronecker_rank" if rank == 1 else "axis_partition_rank",
                    "initializer": name,
                    "shape": list(array.shape),
                    "rank": rank,
                    "left_axes": list(left_axes),
                    "right_axes": list(right_axes),
                    "original_params": int(array.size),
                    "factor_params": factor_params,
                    "parameter_saving": int(array.size - factor_params),
                    "arrays": [left_factor, right_factor],
                }
            )
    return result


def walsh_plan(name: str, array: np.ndarray) -> dict[str, Any] | None:
    """Exact sparse Walsh CP for binary-axis tensors, with factor deduplication."""
    if array.ndim < 3 or any(int(size) != 2 for size in array.shape):
        return None
    order = array.ndim
    values = [exact_fraction(value) for value in array.reshape(-1)]
    coefficients: list[tuple[int, Fraction]] = []
    for mask in range(1 << order):
        total = Fraction(0)
        for flat, value in enumerate(values):
            parity = (mask & flat).bit_count() & 1
            total += -value if parity else value
        coefficient = total / (1 << order)
        if coefficient:
            coefficients.append((mask, coefficient))
    rank = len(coefficients)
    if not rank:
        return None
    weights = np.asarray([float(value) for _, value in coefficients], dtype=array.dtype)
    axis_arrays: list[np.ndarray] = []
    for axis in range(order):
        bit = 1 << (order - 1 - axis)
        factor = np.ones((rank, 2), dtype=array.dtype)
        for row, (mask, _) in enumerate(coefficients):
            factor[row, 1] = -1 if mask & bit else 1
        axis_arrays.append(factor)
    unique: list[np.ndarray] = []
    axis_factor_indices: list[int] = []
    keys: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for factor in axis_arrays:
        key = array_key(factor)
        if key not in keys:
            keys[key] = len(unique)
            unique.append(np.ascontiguousarray(factor))
        axis_factor_indices.append(keys[key])
    arrays = unique + [np.ascontiguousarray(weights)]
    symbols = LABELS[:order]
    expression = ",".join(["k" + symbols[index] for index in range(order)] + ["k"])
    rebuilt = np.asarray(
        np.einsum(expression + "->" + symbols, *axis_arrays, weights, optimize=False),
        dtype=array.dtype,
    )
    if array_bytes(rebuilt) != array_bytes(array):
        return None
    factor_params = int(sum(item.size for item in arrays))
    if factor_params >= array.size:
        return None
    return {
        "kind": "walsh_cp",
        "initializer": name,
        "shape": list(array.shape),
        "rank": rank,
        "axis_factor_indices": axis_factor_indices,
        "weight_factor_index": len(arrays) - 1,
        "original_params": int(array.size),
        "factor_params": factor_params,
        "parameter_saving": int(array.size - factor_params),
        "arrays": arrays,
    }


def onehot_cp_plan(name: str, array: np.ndarray) -> dict[str, Any] | None:
    """Exact nonzero-coordinate CP with bit-identical factors and deduplication."""
    coordinates = np.argwhere(array != 0)
    rank = int(len(coordinates))
    if rank == 0 or rank >= array.size:
        return None
    axis_arrays: list[np.ndarray] = []
    for axis, size in enumerate(array.shape):
        factor = np.zeros((rank, int(size)), dtype=array.dtype)
        factor[np.arange(rank), coordinates[:, axis]] = 1
        axis_arrays.append(factor)
    weights = np.asarray([array[tuple(int(x) for x in coordinate)] for coordinate in coordinates], dtype=array.dtype)
    unique: list[np.ndarray] = []
    indices: list[int] = []
    keys: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for factor in axis_arrays:
        key = array_key(factor)
        if key not in keys:
            keys[key] = len(unique)
            unique.append(np.ascontiguousarray(factor))
        indices.append(keys[key])
    arrays = unique + [np.ascontiguousarray(weights)]
    factor_params = int(sum(item.size for item in arrays))
    if factor_params >= array.size:
        return None
    # Each coordinate has exactly one nonzero CP component, so rebuilding is
    # assignment-exact and cannot change floating accumulation order.
    rebuilt = np.zeros_like(array)
    for row, coordinate in enumerate(coordinates):
        rebuilt[tuple(int(x) for x in coordinate)] = weights[row]
    if array_bytes(rebuilt) != array_bytes(array):
        return None
    return {
        "kind": "onehot_cp",
        "initializer": name,
        "shape": list(array.shape),
        "rank": rank,
        "axis_factor_indices": indices,
        "weight_factor_index": len(arrays) - 1,
        "original_params": int(array.size),
        "factor_params": factor_params,
        "parameter_saving": int(array.size - factor_params),
        "arrays": arrays,
    }


def diagonal_plan(
    name: str,
    array: np.ndarray,
    locations: list[tuple[int, int]],
    model: onnx.ModelProto,
) -> dict[str, Any] | None:
    """Compress repeated-label Einsum operands to their exactly used diagonal."""
    by_term: dict[str, np.ndarray] = {}
    occurrence_terms: list[str] = []
    for node_index, input_index in locations:
        terms, _ = split_equation(model.graph.node[node_index])
        term = terms[input_index]
        unique = "".join(dict.fromkeys(term))
        if len(unique) == len(term):
            return None
        try:
            compressed = np.asarray(np.einsum(f"{term}->{unique}", array, optimize=False), dtype=array.dtype)
        except Exception:
            return None
        by_term.setdefault(term, np.ascontiguousarray(compressed))
        occurrence_terms.append(term)
    unique_arrays: list[np.ndarray] = []
    term_indices: dict[str, int] = {}
    keys: dict[tuple[str, tuple[int, ...], bytes], int] = {}
    for term, compressed in by_term.items():
        key = array_key(compressed)
        if key not in keys:
            keys[key] = len(unique_arrays)
            unique_arrays.append(compressed)
        term_indices[term] = keys[key]
    factor_params = int(sum(item.size for item in unique_arrays))
    if factor_params >= array.size:
        return None
    return {
        "kind": "diagonal_projection",
        "initializer": name,
        "shape": list(array.shape),
        "term_factor_indices": term_indices,
        "original_params": int(array.size),
        "factor_params": factor_params,
        "parameter_saving": int(array.size - factor_params),
        "arrays": unique_arrays,
    }


def uses(model: onnx.ModelProto) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                result[name].append((node_index, input_index))
    return result


def eligible_initializer(
    model: onnx.ModelProto,
    name: str,
    array: np.ndarray,
    locations: list[tuple[int, int]],
) -> bool:
    if not locations or not 2 <= array.ndim <= 8 or array.size > MAX_ELEMENTS:
        return False
    if not finite_numeric(array) or array.dtype.kind != "f":
        return False
    for node_index, input_index in locations:
        node = model.graph.node[node_index]
        if node.op_type != "Einsum":
            return False
        try:
            terms, _ = split_equation(node)
        except Exception:
            return False
        if input_index >= len(terms) or len(terms[input_index]) != array.ndim:
            return False
    return True


def plans(model: onnx.ModelProto) -> tuple[list[dict[str, Any]], dict[str, int]]:
    locations = uses(model)
    result: list[dict[str, Any]] = []
    census = Counter()
    for initializer in model.graph.initializer:
        array = np.asarray(numpy_helper.to_array(initializer))
        locs = locations.get(initializer.name, [])
        if not eligible_initializer(model, initializer.name, array, locs):
            continue
        census["eligible_initializers"] += 1
        candidates: list[dict[str, Any]] = []
        factors = exact_separable(array)
        if factors is not None:
            factor_params = int(sum(item.size for item in factors))
            candidates.append(
                {
                    "kind": "fully_separable_rank1",
                    "initializer": initializer.name,
                    "shape": list(array.shape),
                    "original_params": int(array.size),
                    "factor_params": factor_params,
                    "parameter_saving": int(array.size - factor_params),
                    "arrays": factors,
                }
            )
        candidates.extend(partition_plans(initializer.name, array))
        walsh = walsh_plan(initializer.name, array)
        if walsh is not None:
            candidates.append(walsh)
        onehot = onehot_cp_plan(initializer.name, array)
        if onehot is not None:
            candidates.append(onehot)
        diagonal = diagonal_plan(initializer.name, array, locs, model)
        if diagonal is not None:
            candidates.append(diagonal)

        # Keep the best distinct parameter count of each algebraic family.
        selected: dict[tuple[str, int], dict[str, Any]] = {}
        for candidate in candidates:
            if candidate["parameter_saving"] <= 0:
                continue
            key = (str(candidate["kind"]), int(candidate["factor_params"]))
            selected.setdefault(key, candidate)
        for candidate in selected.values():
            candidate["locations"] = locs
            candidate["initializer_sha256"] = sha_bytes(array_bytes(array))
            result.append(candidate)
            census[f"plans_{candidate['kind']}"] += 1
    return result, dict(census)


def free_label(terms: list[str], rhs: str) -> str:
    used = set("".join(terms) + rhs)
    label = next((item for item in LABELS if item not in used), None)
    if label is None:
        raise ValueError("no free Einsum label")
    return label


def replace_position(
    plan: dict[str, Any],
    term: str,
    factor_names: list[str],
    latent: str,
) -> tuple[list[str], list[str]]:
    kind = str(plan["kind"])
    if kind == "fully_separable_rank1":
        return factor_names, list(term)
    if kind in {"kronecker_rank", "axis_partition_rank"}:
        left_axes = [int(axis) for axis in plan["left_axes"]]
        right_axes = [int(axis) for axis in plan["right_axes"]]
        left = "".join(term[axis] for axis in left_axes)
        right = "".join(term[axis] for axis in right_axes)
        if int(plan["rank"]) == 1:
            return factor_names, [left, right]
        return factor_names, [left + latent, latent + right]
    if kind in {"walsh_cp", "onehot_cp"}:
        indices = [int(index) for index in plan["axis_factor_indices"]]
        weight_index = int(plan["weight_factor_index"])
        names = [factor_names[index] for index in indices] + [factor_names[weight_index]]
        terms = [latent + term[axis] for axis in range(len(term))] + [latent]
        return names, terms
    if kind == "diagonal_projection":
        index = int(plan["term_factor_indices"][term])
        return [factor_names[index]], ["".join(dict.fromkeys(term))]
    raise ValueError(f"unknown plan kind {kind}")


def build(source: onnx.ModelProto, plan: dict[str, Any]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = str(plan["initializer"])
    arrays = [np.asarray(item) for item in plan["arrays"]]
    factor_names = [f"{name}__cp239_{index}" for index in range(len(arrays))]
    replacements = 0
    for node in model.graph.node:
        positions = [index for index, value in enumerate(node.input) if value == name]
        if not positions:
            continue
        terms, rhs = split_equation(node)
        inputs = list(node.input)
        for position in reversed(positions):
            latent = free_label(terms, rhs)
            new_inputs, new_terms = replace_position(plan, terms[position], factor_names, latent)
            inputs[position : position + 1] = new_inputs
            terms[position : position + 1] = new_terms
            replacements += 1
        del node.input[:]
        node.input.extend(inputs)
        set_equation(node, terms, rhs)
    if replacements == 0:
        raise RuntimeError("initializer had no replaceable use")
    remaining = Counter(value for node in model.graph.node for value in node.input if value)
    if remaining[name]:
        raise RuntimeError("initializer remains used")
    kept = [item for item in model.graph.initializer if item.name != name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for factor_name, array in zip(factor_names, arrays):
        model.graph.initializer.append(numpy_helper.from_array(array, factor_name))
    return model


def profile(model: onnx.ModelProto, task: int, label: str) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix=f"cp239_{task:03d}_{label}_") as work:
        path = Path(work) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
    return {"memory": int(memory), "params": int(params), "cost": int(cost)}


def serial_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key not in {"arrays", "locations"}}


def main() -> int:
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    authority_sha = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    census = Counter()
    baseline_cache: dict[int, dict[str, int]] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        census["authority_models"] = len(members)
        for member in members:
            task = int(Path(member).stem.removeprefix("task"))
            if task in EXCLUDED_TASKS:
                census["excluded_tasks"] += 1
                continue
            source_bytes = archive.read(member)
            model = onnx.load_model_from_string(source_bytes)
            task_plans, task_census = plans(model)
            census.update(task_census)
            if task_plans:
                census["tasks_with_algebraic_plan"] += 1
            for ordinal, plan in enumerate(task_plans, 1):
                details = serial_plan(plan)
                try:
                    candidate = build(model, plan)
                    onnx.checker.check_model(candidate, full_check=True)
                    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                    if task not in baseline_cache:
                        baseline_cache[task] = profile(model, task, "base")
                    baseline = baseline_cache[task]
                    current = profile(candidate, task, f"p{ordinal}")
                    strict_lower = current["cost"] < baseline["cost"]
                    memory_nonincrease = current["memory"] <= baseline["memory"]
                    row: dict[str, Any] = {
                        "task": task,
                        "ordinal": ordinal,
                        "authority_member_sha256": sha_bytes(source_bytes),
                        **details,
                        "baseline": baseline,
                        "candidate": current,
                        "memory_nonincrease": memory_nonincrease,
                        "strict_lower": strict_lower,
                    }
                    if strict_lower and memory_nonincrease:
                        path = CANDIDATES / f"task{task:03d}_p{ordinal:02d}_{plan['kind']}.onnx"
                        onnx.save(candidate, path)
                        row["path"] = str(path.relative_to(ROOT))
                        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                        row["projected_gain"] = math.log(baseline["cost"] / current["cost"])
                        census["strict_lower_candidates"] += 1
                    rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "task": task,
                            "ordinal": ordinal,
                            **details,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    rows.sort(
        key=lambda row: (
            not bool(row.get("strict_lower") and row.get("memory_nonincrease")),
            -float(row.get("projected_gain", 0.0)),
            int(row["task"]),
            int(row["ordinal"]),
        )
    )
    for key in (
        "plans_fully_separable_rank1",
        "plans_kronecker_rank",
        "plans_axis_partition_rank",
        "plans_walsh_cp",
        "plans_onehot_cp",
        "plans_diagonal_projection",
        "strict_lower_candidates",
    ):
        census.setdefault(key, 0)
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": authority_sha,
        "excluded_tasks": sorted(EXCLUDED_TASKS),
        "method": {
            "max_initializer_elements": MAX_ELEMENTS,
            "approximate_svd": False,
            "serialized_dtype_byte_identity_required": True,
            "families": [
                "fully_separable_rank1",
                "exact_rational_axis_partition_rank",
                "kronecker_rank1",
                "walsh_cp",
                "deduplicated_onehot_cp",
                "repeated_label_diagonal_projection",
            ],
        },
        "census": dict(census),
        "rows": rows,
        "errors": errors,
    }
    (HERE / "scan.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary = {
        "census": dict(census),
        "rows": len(rows),
        "errors": len(errors),
        "strict_lower": [
            {key: row.get(key) for key in ("task", "kind", "path", "sha256", "baseline", "candidate", "projected_gain")}
            for row in rows
            if row.get("strict_lower") and row.get("memory_nonincrease")
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

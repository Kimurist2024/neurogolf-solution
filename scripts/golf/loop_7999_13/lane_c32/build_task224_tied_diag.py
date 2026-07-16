#!/usr/bin/env python3
"""Tie Csum to diagonally-equivalent Cdiag using global Einsum gauge scales."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import sympy as sp
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task224.onnx"
OUTPUT = HERE / "task224_tied_csum_cdiag.onnx"


def typed_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    result = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        result[value.name] = tuple(int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)
    for item in inferred.graph.initializer:
        result[item.name] = tuple(item.dims)
    return result


def solve_gf2(matrix: list[list[int]], rhs: list[int], variables: int) -> list[int]:
    rows = [[value & 1 for value in row] + [target & 1] for row, target in zip(matrix, rhs)]
    pivot_columns = []
    pivot_row = 0
    for column in range(variables):
        found = next((row for row in range(pivot_row, len(rows)) if rows[row][column]), None)
        if found is None:
            continue
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        for row in range(len(rows)):
            if row != pivot_row and rows[row][column]:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
    if any(not any(row[:variables]) and row[variables] for row in rows):
        raise RuntimeError("sign gauge system is inconsistent")
    solution = [0] * variables
    for row, column in enumerate(pivot_columns):
        solution[column] = rows[row][variables]
    return solution


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {item.name: numpy_helper.to_array(item).copy() for item in model.graph.initializer}
    shapes = typed_shapes(model)
    kept = [name for name in arrays if name != "Csum"]
    variables = []
    for name in kept:
        for axis, dimension in enumerate(arrays[name].shape):
            for component in range(dimension):
                variables.append((name, axis, component))
    for pseudo in ("__row_codes__", "__col_codes__"):
        for component in range(3):
            variables.append((pseudo, 1, component))
    variable_index = {variable: index for index, variable in enumerate(variables)}
    magnitude_rows: list[list[int]] = []
    magnitude_rhs: list[int] = []
    sign_rows: list[list[int]] = []
    sign_rhs: list[int] = []

    # Csum = diag([10,1]) @ Cdiag @ diag([1,-0.1]).
    row_exponent = [1, 0]
    col_exponent = [0, -1]
    col_sign = [0, 1]

    for node in model.graph.node:
        equation = next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")
        left, _ = equation.split("->")
        terms = left.split(",")
        label_dims: dict[str, int] = defaultdict(lambda: 1)
        for name, term in zip(node.input, terms):
            for label, dimension in zip(term, shapes[name]):
                label_dims[label] = max(label_dims[label], dimension)

        expressions_mag: dict[str, list[dict[int, int]]] = {
            label: [defaultdict(int) for _ in range(dimension)]
            for label, dimension in label_dims.items()
        }
        expressions_sign: dict[str, list[dict[int, int]]] = {
            label: [defaultdict(int) for _ in range(dimension)]
            for label, dimension in label_dims.items()
        }
        fixed_mag = {label: [0] * dimension for label, dimension in label_dims.items()}
        fixed_sign = {label: [0] * dimension for label, dimension in label_dims.items()}

        for original_name, term in zip(node.input, terms):
            candidate_name = "Cdiag" if original_name == "Csum" else original_name
            if candidate_name in arrays and candidate_name != "Csum":
                shape = arrays[candidate_name].shape
                for axis, (label, axis_dimension) in enumerate(zip(term, shape)):
                    for component in range(label_dims[label]):
                        local = 0 if axis_dimension == 1 else component
                        index = variable_index[(candidate_name, axis, local)]
                        expressions_mag[label][component][index] += 1
                        expressions_sign[label][component][index] ^= 1
            if original_name == "Csum":
                row_label, col_label = term
                for component in range(label_dims[row_label]):
                    fixed_mag[row_label][component] -= row_exponent[component]
                for component in range(label_dims[col_label]):
                    fixed_mag[col_label][component] -= col_exponent[component]
                    fixed_sign[col_label][component] ^= col_sign[component]
            if original_name in ("row_codes", "col_codes"):
                pseudo = "__row_codes__" if original_name == "row_codes" else "__col_codes__"
                label = term[1]
                for component in range(label_dims[label]):
                    index = variable_index[(pseudo, 1, component)]
                    expressions_mag[label][component][index] += 1
                    expressions_sign[label][component][index] ^= 1

        # Permit the two upstream code tensors to be diagonally scaled, then
        # account for those exact scales at every consumer in the final node.
        output_name = node.output[0]
        if output_name in ("row_codes", "col_codes"):
            pseudo = "__row_codes__" if output_name == "row_codes" else "__col_codes__"
            output_term = equation.split("->")[1]
            label = output_term[1]
            for component in range(label_dims[label]):
                index = variable_index[(pseudo, 1, component)]
                expressions_mag[label][component][index] -= 1
                expressions_sign[label][component][index] ^= 1

        # Each label's scale ratio must be component-independent.
        for label, dimension in label_dims.items():
            for component in range(1, dimension):
                mag_row = [0] * len(variables)
                sign_row = [0] * len(variables)
                for index, coefficient in expressions_mag[label][component].items():
                    mag_row[index] += coefficient
                for index, coefficient in expressions_mag[label][0].items():
                    mag_row[index] -= coefficient
                for index, coefficient in expressions_sign[label][component].items():
                    sign_row[index] ^= coefficient
                for index, coefficient in expressions_sign[label][0].items():
                    sign_row[index] ^= coefficient
                magnitude_rows.append(mag_row)
                magnitude_rhs.append(-(fixed_mag[label][component] - fixed_mag[label][0]))
                sign_rows.append(sign_row)
                sign_rhs.append(fixed_sign[label][component] ^ fixed_sign[label][0])

        # Product of the per-label constants must be exactly one.
        mag_row = [0] * len(variables)
        sign_row = [0] * len(variables)
        fixed_m = fixed_s = 0
        for label in label_dims:
            for index, coefficient in expressions_mag[label][0].items():
                mag_row[index] += coefficient
            for index, coefficient in expressions_sign[label][0].items():
                sign_row[index] ^= coefficient
            fixed_m += fixed_mag[label][0]
            fixed_s ^= fixed_sign[label][0]
        magnitude_rows.append(mag_row)
        magnitude_rhs.append(-fixed_m)
        sign_rows.append(sign_row)
        sign_rhs.append(fixed_s)

    matrix = sp.Matrix(magnitude_rows)
    target = sp.Matrix(magnitude_rhs)
    solution_set = sp.linsolve((matrix, target))
    if solution_set is sp.EmptySet:
        raise RuntimeError("magnitude gauge system is inconsistent")
    symbolic = next(iter(solution_set))
    free = sorted(set().union(*(value.free_symbols for value in symbolic)), key=str)
    magnitude_solution = [sp.simplify(value.subs({symbol: 0 for symbol in free})) for value in symbolic]
    sign_solution = solve_gf2(sign_rows, sign_rhs, len(variables))

    scales: dict[str, list[np.ndarray]] = {}
    nontrivial = []
    for name in kept:
        scales[name] = [np.ones(dimension, dtype=np.float64) for dimension in arrays[name].shape]
    for variable, exponent, sign in zip(variables, magnitude_solution, sign_solution):
        name, axis, component = variable
        value = ((-1.0) if sign else 1.0) * 10.0 ** float(exponent)
        if name in scales:
            scales[name][axis][component] = value
        if value != 1.0:
            nontrivial.append({"variable": list(variable), "exponent": str(exponent), "sign": sign, "scale": value})

    for name in kept:
        value = arrays[name].astype(np.float64)
        for axis, vector in enumerate(scales[name]):
            reshape = [1] * value.ndim
            reshape[axis] = len(vector)
            value *= vector.reshape(reshape)
        arrays[name] = value.astype(arrays[name].dtype)

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "Csum":
                node.input[index] = "Cdiag"
    del model.graph.initializer[:]
    for name in kept:
        model.graph.initializer.append(numpy_helper.from_array(arrays[name], name=name))
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    (HERE / "task224_tied_gauge.json").write_text(
        json.dumps(
            {
                "variables": len(variables),
                "magnitude_equations": len(magnitude_rows),
                "magnitude_rank": int(matrix.rank()),
                "free_symbols": [str(symbol) for symbol in free],
                "nontrivial_scales": nontrivial,
            },
            indent=2,
        )
        + "\n"
    )
    print(OUTPUT)
    print("nontrivial", len(nontrivial))


if __name__ == "__main__":
    main()

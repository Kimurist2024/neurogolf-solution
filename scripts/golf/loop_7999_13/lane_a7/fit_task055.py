#!/usr/bin/env python3
"""Refit task055's color polynomial at lower degree from generator constraints."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
NORMAL_Q = (0, 1, 2, 3, 4, 6)


def region_target(
    row: int,
    col: int,
    rows: tuple[int, int, int],
    cols: tuple[int, int, int],
) -> int:
    row_breaks = (rows[0], rows[0] + rows[1] + 1)
    col_breaks = (cols[0], cols[0] + cols[1] + 1)
    rr = 0 if row < row_breaks[0] else 1 if row < row_breaks[1] else 2
    cc = 0 if col < col_breaks[0] else 1 if col < col_breaks[1] else 2
    return {(0, 1): 2, (1, 0): 4, (1, 1): 6, (1, 2): 3, (2, 1): 1}.get((rr, cc), 0)


def axis_features(n: int, s0: int, s1: int, other_n: int) -> np.ndarray:
    """Exact three-coordinate feature carried by the incumbent terminal net."""
    k = -float((n - 2) * (other_n - 2))
    hs = 2.0 * k * (s0 + s1 - 1)
    hd = -2.0 * k * (s1 - s0)
    x = np.arange(30, dtype=np.float64) * 2.0
    a = 0.5 * hd
    b = x * k - 0.5 * hs
    return np.stack((np.full(30, 0.5 * a * a), -0.5 * a * b, b * b), axis=1)


def fit(degree: int, shifts: np.ndarray) -> np.ndarray:
    """Fit l,g,o coefficients with normalized signed margin constraints."""
    assert shifts.shape == (degree + 1,)
    constraints: list[np.ndarray] = []
    values = (1, 2, 3, 5, 7, 9)
    triples = [
        x for x in itertools.product(values, repeat=3) if sum(x) + 2 <= 30
    ]
    cases = [
        (triples[i], triples[(i * 37 + 11) % len(triples)])
        for i in range(0, len(triples), 9)
    ]
    for rows, cols in cases:
        height, width = sum(rows) + 2, sum(cols) + 2
        sr = (rows[0], rows[0] + rows[1] + 1)
        sc = (cols[0], cols[0] + cols[1] + 1)
        hf = axis_features(height, *sr, width)
        vf = axis_features(width, *sc, height)
        for row in range(height):
            if row in sr:
                continue
            for col in range(width):
                if col in sc:
                    continue
                target = region_target(row, col, rows, cols)
                geometry = np.outer(hf[row], vf[col]).reshape(9)
                for q in NORMAL_Q:
                    qbasis = (float(q) / 8.0 + shifts) ** degree
                    feature = np.einsum("l,x->lx", qbasis, geometry).reshape(-1)
                    sign = 1.0 if q == target else -1.0
                    feature /= np.max(np.abs(feature))
                    constraints.append(-sign * feature)
    matrix = np.stack(constraints)
    result = linprog(
        np.zeros(matrix.shape[1]),
        A_ub=matrix,
        b_ub=-np.ones(matrix.shape[0]),
        bounds=[(None, None)] * matrix.shape[1],
        method="highs",
        options={
            "dual_feasibility_tolerance": 1e-9,
            "primal_feasibility_tolerance": 1e-9,
        },
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.reshape(degree + 1, 3, 3).astype(np.float32)


def build(degree: int) -> Path:
    shifts = np.linspace(-1.0, 1.0, degree + 1, dtype=np.float64)
    coefficients = fit(degree, shifts)
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    graph = model.graph
    arrays = {
        "Lpoly": np.stack(
            (shifts.astype(np.float32), np.ones(degree + 1, dtype=np.float32))
        ),
        "Acoef": coefficients,
    }
    kept = []
    for initializer in graph.initializer:
        if initializer.name in arrays:
            kept.append(numpy_helper.from_array(arrays[initializer.name], initializer.name))
        else:
            kept.append(initializer)
    del graph.initializer[:]
    graph.initializer.extend(kept)

    final = graph.node[9]
    equation_attr = next(attr for attr in final.attribute if attr.name == "equation")
    equation = onnx.helper.get_attribute_value(equation_attr).decode("ascii")
    old_prefix = "qA,Al,qB,Bl,qC,Cl,qD,Dl,qE,El,lgo,"
    pairs = ",".join(
        item for _ in range(degree) for item in ("qA", "Al")
    ) + ",lgo,"
    assert equation.startswith(old_prefix)
    equation_attr.s = (pairs + equation[len(old_prefix) :]).encode("ascii")
    old_inputs = list(final.input)
    final.input[:] = [
        item for _ in range(degree) for item in ("Qpoly", "Lpoly")
    ] + ["Acoef"] + old_inputs[11:]

    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidates" / f"task055_degree{degree}_refit.onnx"
    onnx.save(model, output)
    print(output, "params", sum(np.prod(x.dims) for x in graph.initializer))
    return output


if __name__ == "__main__":
    for candidate_degree in (3, 4):
        try:
            build(candidate_degree)
        except RuntimeError as error:
            print(f"degree {candidate_degree}: {error}")

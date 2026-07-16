#!/usr/bin/env python3
"""Cutting-plane degree-4 fit using task055's exact incumbent carriers."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASK = 55
ACTIVE_Q = np.asarray([0, 1, 2, 3, 4, 6, 8], dtype=np.int64)


def carrier_session(model: onnx.ModelProto) -> ort.InferenceSession:
    probe = copy.deepcopy(model)
    probe.graph.output.extend(
        [
            helper.make_tensor_value_info("HZ", onnx.TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("VZ", onnx.TensorProto.FLOAT, [3]),
        ]
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(probe.SerializeToString(), options)


def encode(grid: list[list[int]]) -> np.ndarray:
    output = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            output[0, color, row, col] = 1.0
    return output


def geometry(hz: np.ndarray, vz: np.ndarray, x: np.ndarray, pa: np.ndarray) -> np.ndarray:
    # A[h,e] = sum_{r,u} X[r,h] * HZ[u] * PA[r,e,u]
    ah = np.einsum("rh,u,reu->he", x, hz, pa)
    av = np.einsum("aw,m,ajm->wj", x, vz, pa)
    hgo = np.einsum("he,hf,efg->hg", ah, ah, pa)
    voo = np.einsum("wj,wk,jko->wo", av, av, pa)
    return np.einsum("hg,wo->hwgo", hgo, voo)


def rows_for_example(
    example: dict,
    session: ort.InferenceSession,
    arrays: dict[str, np.ndarray],
    degree: int,
    shifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    input_grid = example["input"]
    expected_grid = example["output"]
    height, width = len(expected_grid), len(expected_grid[0])
    if height > 30 or width > 30:
        return None
    encoded = encode(input_grid)
    hz, vz = session.run(["HZ", "VZ"], {"input": encoded})
    geo = geometry(hz.reshape(3), vz.reshape(3), arrays["X"], arrays["PA"])
    qbasis = (ACTIVE_Q[:, None].astype(np.float64) / 8.0 + shifts) ** degree
    colors = np.asarray(input_grid, dtype=np.int64)
    expected = np.asarray(expected_grid, dtype=np.int64)
    color_dot = arrays["Ssel"] @ arrays["Ssel"].T

    all_features: list[np.ndarray] = []
    all_signs: list[np.ndarray] = []
    for q_index, q in enumerate(ACTIVE_Q):
        factor = color_dot[colors, q]
        feature = np.einsum(
            "l,hwgo,hw->hwlgo",
            qbasis[q_index],
            geo[:height, :width],
            factor,
        ).reshape(height * width, -1)
        sign = np.where(expected.reshape(-1) == q, 1.0, -1.0)
        norm = np.max(np.abs(feature), axis=1)
        usable = norm > 0
        feature = feature[usable] / norm[usable, None]
        all_features.append(feature)
        all_signs.append(sign[usable])
    return np.concatenate(all_features), np.concatenate(all_signs)


def fit(features: np.ndarray, signs: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(55007)
    selected = rng.choice(len(features), size=min(5000, len(features)), replace=False)
    selected = np.unique(selected)
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    for iteration in range(20):
        matrix = -signs[selected, None] * features[selected]
        result = linprog(
            np.zeros(features.shape[1]),
            A_ub=matrix,
            b_ub=-np.ones(len(selected)),
            bounds=[(None, None)] * features.shape[1],
            method="highs",
            options={
                "dual_feasibility_tolerance": 1e-9,
                "primal_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            raise RuntimeError(result.message)
        coefficients = result.x
        margins = signs * (features @ coefficients)
        bad = np.flatnonzero(margins < 1.0 - 1e-7)
        print(
            "iteration",
            iteration,
            "pool",
            len(selected),
            "bad",
            len(bad),
            "min_margin",
            float(margins.min()),
        )
        if len(bad) == 0:
            break
        worst = bad[np.argsort(margins[bad])[:5000]]
        selected = np.unique(np.concatenate((selected, worst)))
    else:
        raise RuntimeError("cutting plane did not converge")
    # Enlarge the margin before f32 storage and terminal fp32 contractions.
    return (coefficients * 16.0).astype(np.float32)


def build(model: onnx.ModelProto, coefficients: np.ndarray, degree: int, shifts: np.ndarray) -> Path:
    graph = model.graph
    replacements = {
        "Lpoly": np.stack(
            (shifts.astype(np.float32), np.ones(degree + 1, dtype=np.float32))
        ),
        "Acoef": coefficients.reshape(degree + 1, 3, 3),
    }
    initializers = []
    for initializer in graph.initializer:
        if initializer.name in replacements:
            initializers.append(
                numpy_helper.from_array(replacements[initializer.name], initializer.name)
            )
        else:
            initializers.append(initializer)
    del graph.initializer[:]
    graph.initializer.extend(initializers)

    final = graph.node[9]
    equation_attr = next(attr for attr in final.attribute if attr.name == "equation")
    equation = helper.get_attribute_value(equation_attr).decode("ascii")
    prefix = "qA,Al,qB,Bl,qC,Cl,qD,Dl,qE,El,lgo,"
    assert equation.startswith(prefix)
    new_prefix = ",".join(
        item for _ in range(degree) for item in ("qA", "Al")
    ) + ",lgo,"
    equation_attr.s = (new_prefix + equation[len(prefix) :]).encode("ascii")
    old_inputs = list(final.input)
    final.input[:] = [
        item for _ in range(degree) for item in ("Qpoly", "Lpoly")
    ] + ["Acoef"] + old_inputs[11:]
    onnx.checker.check_model(model, full_check=True)
    output = HERE / "candidates" / f"task055_degree{degree}_exact_feature_fit.onnx"
    onnx.save(model, output)
    print(output)
    return output


def main() -> None:
    degree = 4
    shifts = np.linspace(-1.0, 1.0, degree + 1, dtype=np.float64)
    model = onnx.load(HERE / "baseline" / "task055.onnx")
    arrays = {
        item.name: numpy_helper.to_array(item).astype(np.float64)
        for item in model.graph.initializer
    }
    session = carrier_session(model)
    examples = scoring.load_examples(TASK)
    known = [example for split in ("train", "test", "arc-gen") for example in examples[split]]
    feature_parts = []
    sign_parts = []
    for index, example in enumerate(known):
        encoded_rows = rows_for_example(example, session, arrays, degree, shifts)
        if encoded_rows is None:
            continue
        features, signs = encoded_rows
        feature_parts.append(features)
        sign_parts.append(signs)
        if index % 50 == 0:
            print("encoded", index, "rows", sum(len(item) for item in feature_parts))
    features = np.concatenate(feature_parts)
    signs = np.concatenate(sign_parts)
    print("constraints", features.shape)
    coefficients = fit(features, signs)
    build(model, coefficients, degree, shifts)


if __name__ == "__main__":
    main()

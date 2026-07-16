#!/usr/bin/env python3
"""Search a lower-magnitude task328 CoefAB over all color-orbit states.

The final output is linear in the 16 CoefAB entries.  The generator has only
143 representatives modulo permutation of nonzero colors.  We remove CoefAB
from the terminal Einsum, expose its four labels as output axes, enumerate all
representatives, deduplicate the resulting feature vectors, and solve a linear
program for true logits >= 1 and false logits <= 0.

This is a discovery tool.  Any coefficient hit must still be rebuilt into an
ONNX artifact and pass the independent four-runtime full-support audit.
"""

from __future__ import annotations

import copy
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = (
    ROOT
    / "scripts/golf/loop_7999_13/lane_b26/task328_reuse_j_diagonal.onnx"
)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize_model rejected source")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        clean.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def canonical_cases():
    generator = importlib.import_module("task_d22278a0")
    for size in range(6, 19):
        corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        for count in range(2, 5):
            for selected in itertools.combinations(corners, count):
                rows, cols = zip(*selected)
                colors = tuple(range(1, count + 1))
                yield generator.generate(
                    size=size, rows=rows, cols=cols, colors=colors
                )


def traced_dynamic(model: onnx.ModelProto) -> ort.InferenceSession:
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    traced.graph.output.extend(
        (
            helper.make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1, 10, 2, 2]),
            helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, [5]),
        )
    )
    return session(traced)


def coefficient_basis(
    node: onnx.NodeProto,
    arrays: dict[str, np.ndarray],
    dynamic: dict[str, np.ndarray],
) -> np.ndarray:
    equation = next(
        attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation"
    )
    lhs = equation.split("->")[0].split(",")
    names = list(node.input)
    index = next(i for i, name in enumerate(names) if name == "CoefAB")
    operands = [dynamic.get(name, arrays.get(name)) for name in names]
    basis_equation = (
        ",".join(lhs[:index] + lhs[index + 1 :]) + "->nghabcrs"
    )
    basis = np.einsum(
        basis_equation,
        *(operands[:index] + operands[index + 1 :]),
        optimize="greedy",
    )
    # Move the four coefficient axes to the end: [9000, 16].
    return np.moveaxis(basis, (1, 2, 3, 4), (4, 5, 6, 7)).reshape(-1, 16)


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    final = model.graph.node[-1]
    trace = traced_dynamic(model)
    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for index, example in enumerate(canonical_cases(), start=1):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError("canonical case was not convertible")
        c_value, z_value = trace.run(
            None, {trace.get_inputs()[0].name: benchmark["input"]}
        )
        feature_rows.append(
            coefficient_basis(
                final,
                arrays,
                {"C": c_value, "z": z_value, "input": benchmark["input"]},
            )
        )
        labels.append(benchmark["output"].astype(bool).reshape(-1))
        if index % 25 == 0:
            print(f"features {index}/143", flush=True)

    features = np.concatenate(feature_rows).astype(np.float64)
    truth = np.concatenate(labels)
    joined = np.concatenate((features, truth[:, None]), axis=1)
    unique = np.unique(joined, axis=0)
    unique_features = unique[:, :16]
    unique_truth = unique[:, 16].astype(bool)

    # Detect an impossible collision where one feature vector has both labels.
    vectors, inverse = np.unique(unique_features, axis=0, return_inverse=True)
    positive_group = np.zeros(vectors.shape[0], dtype=bool)
    negative_group = np.zeros(vectors.shape[0], dtype=bool)
    np.logical_or.at(positive_group, inverse, unique_truth)
    np.logical_or.at(negative_group, inverse, ~unique_truth)
    conflicts = int(np.count_nonzero(positive_group & negative_group))
    zero = np.all(vectors == 0, axis=1)
    zero_positive = int(np.count_nonzero(zero & positive_group))

    positive = vectors[positive_group]
    negative = vectors[negative_group & ~zero]
    # Variables are coefficient[16] and absolute-value envelope[16].
    zeros = np.zeros((positive.shape[0] + negative.shape[0], 16))
    a_margin = np.concatenate((-positive, zeros[: positive.shape[0]]), axis=1)
    a_false = np.concatenate((negative, zeros[: negative.shape[0]]), axis=1)
    eye = np.eye(16)
    a_abs_pos = np.concatenate((eye, -eye), axis=1)
    a_abs_neg = np.concatenate((-eye, -eye), axis=1)
    a_ub = np.concatenate((a_margin, a_false, a_abs_pos, a_abs_neg), axis=0)
    b_ub = np.concatenate(
        (
            -np.ones(positive.shape[0]),
            np.zeros(negative.shape[0]),
            np.zeros(32),
        )
    )
    objective = np.concatenate((np.zeros(16), np.ones(16)))
    print(
        "LP",
        "positive", positive.shape[0],
        "negative", negative.shape[0],
        "constraints", a_ub.shape[0],
        flush=True,
    )
    solved = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * 16 + [(0, None)] * 16,
        method="highs",
        options={"presolve": True},
    )
    coefficients = solved.x[:16] if solved.success else None
    validation = None
    if coefficients is not None:
        all_logits = vectors @ coefficients
        validation = {
            "minimum_positive_logit": float(all_logits[positive_group].min()),
            "maximum_negative_logit": float(all_logits[negative_group].max()),
            "maximum_abs_coefficient": float(np.abs(coefficients).max()),
            "l1_coefficient": float(np.abs(coefficients).sum()),
        }
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "orbit_representatives": 143,
        "raw_rows": int(features.shape[0]),
        "unique_feature_label_rows": int(unique.shape[0]),
        "unique_feature_vectors": int(vectors.shape[0]),
        "positive_vectors": int(positive_group.sum()),
        "negative_nonzero_vectors": int((negative_group & ~zero).sum()),
        "conflicting_label_vectors": conflicts,
        "zero_positive_vectors": zero_positive,
        "lp_success": bool(solved.success),
        "lp_status": int(solved.status),
        "lp_message": solved.message,
        "coefficients": coefficients.reshape(2, 2, 2, 2).tolist()
        if coefficients is not None
        else None,
        "validation": validation,
    }
    (HERE / "stable_coef_search.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

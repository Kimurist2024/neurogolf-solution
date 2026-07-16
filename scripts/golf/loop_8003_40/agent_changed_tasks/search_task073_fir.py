#!/usr/bin/env python3
"""Solve the exhaustive task073 FIR inequalities after removing the NaN mask tap."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from scipy.optimize import linprog


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "base_models" / "task073.onnx"
OUT = HERE / "task073_fir"


def cases() -> list[tuple[np.ndarray, np.ndarray, tuple[int, ...]]]:
    placements = [(column,) for column in range(5)]
    placements += [pair for pair in itertools.combinations(range(5), 2) if pair[1] - pair[0] > 1]
    result = []
    for columns in placements:
        source = np.zeros((5, 5), dtype=np.int64)
        target = np.zeros((5, 5), dtype=np.int64)
        source[4, :] = target[4, :] = 5
        for column in columns:
            source[3, column] = target[3, column] = 5
            source[2, column] = 1
            target[4, column] = 1
        result.append((source, target, columns))
    return result


def one_hot(grid: np.ndarray) -> np.ndarray:
    encoded = np.zeros((10, 30, 30), dtype=np.float64)
    rows, cols = grid.shape
    for row in range(rows):
        for col in range(cols):
            encoded[grid[row, col], row, col] = 1.0
    return encoded


def normalized_input(grid: np.ndarray) -> np.ndarray:
    encoded = one_hot(grid)
    mean = encoded.mean()
    variance = ((encoded - mean) ** 2).mean()
    scale = np.float32(0.05272629112005234)
    bias = np.float32(-0.9871222376823425)
    epsilon = np.float32(1e-5)
    return ((encoded - mean) / np.sqrt(variance + epsilon)) * scale + bias


def feature_matrix(length: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for source, target, _ in cases():
        g = normalized_input(source)
        expected = one_hot(target).astype(bool)
        for row in range(30):
            taps = []
            for tap in range(length):
                if row >= tap:
                    taps.append(g[:, row - tap, :])
                else:
                    taps.append(np.zeros((10, 30), dtype=np.float64))
            stacked = np.stack(taps, axis=-1)
            features.append(stacked.reshape(-1, length))
            labels.append(expected[:, row, :].reshape(-1))
    return np.concatenate(features), np.concatenate(labels)


def solve(length: int) -> dict[str, object]:
    x, positive = feature_matrix(length)
    signed = np.where(positive[:, None], -x, x)
    # Force a unit raw margin; the system is homogeneous, so this is exactly a
    # strict-separability test and not a fitted threshold.
    result = linprog(
        np.zeros(length),
        A_ub=signed,
        b_ub=-np.ones(len(signed)),
        bounds=[(None, None)] * length,
        method="highs",
        options={"presolve": True},
    )
    row: dict[str, object] = {
        "length": length,
        "cases": len(cases()),
        "constraints": int(len(signed)),
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
    }
    if not result.success:
        return row
    coeff = np.asarray(result.x, dtype=np.float32)
    raw = x @ coeff.astype(np.float64)
    row.update({
        "coefficients": coeff.tolist(),
        "minimum_positive": float(np.min(raw[positive])),
        "maximum_nonpositive": float(np.max(raw[~positive])),
    })
    model = onnx.load(SOURCE)
    replacement = numpy_helper.from_array(coeff.reshape(1, 1, length, 1), name="x")
    for index, init in enumerate(model.graph.initializer):
        if init.name == "x":
            model.graph.initializer[index].CopyFrom(replacement)
            break
    conv = next(node for node in model.graph.node if node.op_type == "ConvTranspose")
    for attr in conv.attribute:
        if attr.name == "pads":
            attr.ints[:] = [0, 0, length - 1, 0]
            break
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"task073_fir{length}.onnx"
    onnx.save(model, destination)
    row.update({
        "candidate": str(destination),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "checker": "PASS",
        "strict_shape_inference": "PASS",
    })
    return row


def main() -> None:
    rows = [solve(length) for length in range(1, 6)]
    path = HERE / "task073_fir_search.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

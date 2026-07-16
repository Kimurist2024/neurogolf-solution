#!/usr/bin/env python3
"""Find parameter-saving exact low-rank initializers consumed by Einsum nodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT.parent / "loop_8000_46" / "latest_8002_63_models"
OUTPUT = ROOT / "einsum_low_rank_findings.json"


def equation(node: onnx.NodeProto) -> str:
    for attribute in node.attribute:
        if attribute.name == "equation":
            return attribute.s.decode("utf-8")
    return ""


def simple_rank_factor(array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    matrix = np.asarray(array, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    rows, columns = matrix.shape
    if rank == 0 or rank * (rows + columns) >= rows * columns:
        return None

    _, _, pivots = __import__("scipy.linalg").linalg.qr(
        matrix, mode="economic", pivoting=True
    )
    basis = matrix[:, pivots[:rank]]
    coefficients = np.linalg.lstsq(basis, matrix, rcond=None)[0]
    rounded = np.round(coefficients)
    coefficients = np.where(
        np.isclose(coefficients, rounded, rtol=0.0, atol=1e-10),
        rounded,
        coefficients,
    )
    if not np.allclose(basis @ coefficients, matrix, rtol=1e-7, atol=1e-7):
        return None
    return basis.astype(array.dtype), coefficients.astype(array.dtype)


def main() -> None:
    findings: list[dict[str, object]] = []
    for path in sorted(MODEL_DIR.glob("task*.onnx")):
        task = int(path.stem.removeprefix("task"))
        model = onnx.load(path, load_external_data=False)
        initializers = {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in model.graph.initializer
        }
        for node_index, node in enumerate(model.graph.node):
            if node.op_type != "Einsum":
                continue
            subscripts = equation(node).split("->", 1)[0].split(",")
            for position, (name, subscript) in enumerate(zip(node.input, subscripts)):
                array = initializers.get(name)
                if array is None or array.ndim != 2 or len(subscript) != 2:
                    continue
                factors = simple_rank_factor(array)
                if factors is None:
                    continue
                left, right = factors
                savings = int(array.size - left.size - right.size)
                findings.append(
                    {
                        "task": task,
                        "node_index": node_index,
                        "initializer": name,
                        "position": position,
                        "subscript": subscript,
                        "shape": list(array.shape),
                        "rank": int(left.shape[1]),
                        "left_shape": list(left.shape),
                        "right_shape": list(right.shape),
                        "parameter_savings": savings,
                        "left": left.tolist(),
                        "right": right.tolist(),
                    }
                )
    findings.sort(key=lambda item: (-int(item["parameter_savings"]), int(item["task"])))
    OUTPUT.write_text(json.dumps({"findings": findings}, indent=2) + "\n")
    print(json.dumps({"count": len(findings), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()

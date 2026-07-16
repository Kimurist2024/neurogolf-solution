#!/usr/bin/env python3
"""Find byte-exact aliases, scalar ties, and diagonal matrix equivalences."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def proportional(left: np.ndarray, right: np.ndarray) -> float | None:
    if not np.array_equal(left == 0, right == 0):
        return None
    mask = right != 0
    if not mask.any():
        return 1.0
    ratios = left[mask].astype(np.float64) / right[mask].astype(np.float64)
    if np.all(ratios == ratios[0]) and np.array_equal(
        (right.astype(np.float64) * ratios[0]).astype(left.dtype), left
    ):
        return float(ratios[0])
    return None


def diagonal_equivalence(left: np.ndarray, right: np.ndarray) -> dict[str, object] | None:
    """Return row/column scales for left=diag(r)*right*diag(c), if exact."""
    if left.ndim != 2 or left.shape != right.shape or not np.array_equal(left == 0, right == 0):
        return None
    rows, cols = left.shape
    row_scale = np.full(rows, np.nan, dtype=np.float64)
    col_scale = np.full(cols, np.nan, dtype=np.float64)
    for start_row in range(rows):
        if not np.isnan(row_scale[start_row]) or not np.any(right[start_row] != 0):
            continue
        row_scale[start_row] = 1.0
        changed = True
        while changed:
            changed = False
            for row, col in zip(*np.nonzero(right)):
                ratio = float(left[row, col]) / float(right[row, col])
                if not np.isnan(row_scale[row]) and np.isnan(col_scale[col]):
                    col_scale[col] = ratio / row_scale[row]
                    changed = True
                elif not np.isnan(col_scale[col]) and np.isnan(row_scale[row]):
                    row_scale[row] = ratio / col_scale[col]
                    changed = True
    row_scale[np.isnan(row_scale)] = 1.0
    col_scale[np.isnan(col_scale)] = 1.0
    rebuilt = (right.astype(np.float64) * row_scale[:, None] * col_scale[None, :]).astype(left.dtype)
    if not np.array_equal(rebuilt, left):
        return None
    return {"row_scale": row_scale.tolist(), "col_scale": col_scale.tolist()}


def main() -> None:
    result = {}
    for path in sorted((HERE / "baseline").glob("*.onnx")):
        model = onnx.load(path)
        arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        aliases = []
        diagonals = []
        repeated_rows = []
        for first, second in itertools.combinations(arrays, 2):
            left, right = arrays[first], arrays[second]
            if left.ndim == right.ndim:
                for permutation in itertools.permutations(range(right.ndim)):
                    moved = np.transpose(right, permutation)
                    if moved.shape != left.shape:
                        continue
                    ratio = proportional(left, moved)
                    if ratio is not None:
                        aliases.append(
                            {
                                "left": first,
                                "right": second,
                                "permutation": list(permutation),
                                "ratio": ratio,
                                "signed": ratio in (-1.0, 1.0),
                                "power2": ratio != 0 and np.log2(abs(ratio)).is_integer(),
                            }
                        )
            relation = diagonal_equivalence(left, right)
            if relation is not None:
                diagonals.append({"left": first, "right": second, **relation})
            if left.ndim == right.ndim == 2 and left.shape[1] == right.shape[1]:
                for i, j in itertools.product(range(left.shape[0]), range(right.shape[0])):
                    if np.array_equal(left[i], right[j]):
                        repeated_rows.append(
                            {"left": first, "left_row": i, "right": second, "right_row": j}
                        )
        result[path.stem] = {
            "exact_or_scalar_aliases": aliases,
            "diagonal_equivalences": diagonals,
            "cross_initializer_repeated_rows": repeated_rows,
        }
    (HERE / "exact_relations.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

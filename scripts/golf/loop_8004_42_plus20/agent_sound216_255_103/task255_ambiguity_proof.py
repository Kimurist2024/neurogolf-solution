#!/usr/bin/env python3
"""Construct equal legal task255 inputs with different expected outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SIZE = 30
GREEN = 3


def draw(grid: np.ndarray, row: int, col: int, color: int) -> None:
    if 0 <= row < SIZE and 0 <= col < SIZE:
        grid[row, col] = color


def render(low_right_tall: int) -> tuple[np.ndarray, np.ndarray]:
    # Legal generator values: artery row=4,col=10,wide=6,tall=size+2 and
    # low-right vein row=row0+23,col=col0+wide0-2,wide=size,tall in {3,4}.
    rows = [4, 27]
    cols = [10, 14]
    wides = [6, SIZE]
    talls = [SIZE + 2, low_right_tall]
    bitmap = np.zeros((SIZE, SIZE), dtype=np.int64)
    for row, col, wide, tall in zip(rows, cols, wides, talls):
        for r in range(row, row + tall):
            for c in range(col, col + wide):
                draw(bitmap, r, c, 0)
    for row, col, wide, tall in zip(rows, cols, wides, talls):
        for r in range(row + 1, row + tall - 1):
            for c in range(col + 1, col + wide - 1):
                draw(bitmap, r, c, GREEN)
    output = bitmap.copy()
    input_grid = bitmap.copy()
    input_grid[input_grid == GREEN] = 0
    return input_grid, output


def main() -> int:
    input3, output3 = render(3)
    input4, output4 = render(4)
    result = {
        "same_input": bool(np.array_equal(input3, input4)),
        "same_output": bool(np.array_equal(output3, output4)),
        "output_diff_cells": int(np.count_nonzero(output3 != output4)),
        "legal_parameter_difference": {"low_right_tall": [3, 4]},
        "conclusion": "no deterministic input-only ONNX can be exact on both",
    }
    (HERE / "task255_ambiguity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["same_input"] and not result["same_output"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

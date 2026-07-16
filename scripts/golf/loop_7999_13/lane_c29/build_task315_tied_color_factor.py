#!/usr/bin/env python3
"""Tie task315's first color factor across the two spatial routing modes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task315.onnx"
OUTPUT = HERE / "task315_tied_color_factor.onnx"

# These float32 factors were solved against the complete 3x3 color truth table.
# For every gate color g, source color s, and output color c in {0,1,2},
# a0(h,w)*K0[g,c] + a1(h,w)*K1[s,c] has the required sign for all 81
# in-grid locations.  Its minimum margin after division by |a0|+|a1| is 4.22.
# Columns 3..9 remain exactly zero so padded channels are never activated.
COLOR_FEATURE = np.asarray(
    [
        [4.38404, 0.08437491, -3.584194],
        [-5.8431463, -6.8093424, -1.8962231],
    ],
    dtype=np.float32,
)
COMMON_FACTOR = np.asarray(
    [[0.9752296, 0.77568525], [-3.066095, -0.7634821]],
    dtype=np.float32,
)
MODE_FACTOR = np.asarray(
    [
        [[0.15122497, -3.8797243], [-0.29262805, 7.352904]],
        [[-1.337327, -2.097078], [0.1307806, -0.10142411]],
    ],
    dtype=np.float32,
)


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise KeyError(name)


def main() -> None:
    model = onnx.load(SOURCE)
    color = np.zeros((2, 10), dtype=np.float32)
    color[:, :3] = COLOR_FEATURE
    replace_initializer(model, "colorF", color)
    replace_initializer(model, "L1", COMMON_FACTOR)
    replace_initializer(model, "L2", MODE_FACTOR)

    node = model.graph.node[0]
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    decoded = equation.s.decode("ascii")
    if decoded.count("tAB") != 1:
        raise RuntimeError(f"unexpected equation: {decoded}")
    equation.s = decoded.replace("tAB", "AB", 1).encode("ascii")

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

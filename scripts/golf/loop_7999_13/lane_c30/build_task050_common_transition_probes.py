#!/usr/bin/env python3
"""Build mode-independent K probes for task050; screening only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline" / "task050.onnx"


def main() -> None:
    source = onnx.load(SOURCE)
    original = numpy_helper.to_array(next(x for x in source.graph.initializer if x.name == "K"))
    variants = {
        "slice0": original[0],
        "slice1": original[1],
        "sum": original[0] + original[1],
        "hadamard": np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32),
    }
    substitutions = {
        "hAB": "AB", "hCD": "CD", "hEF": "EF", "hGH": "GH",
        "lIJ": "IJ", "lKL": "KL", "lMN": "MN", "lOP": "OP",
    }
    for label, matrix in variants.items():
        model = onnx.load(SOURCE)
        for index, item in enumerate(model.graph.initializer):
            if item.name == "K":
                model.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(matrix.astype(np.float32), name="K")
                )
        equation = next(attr for attr in model.graph.node[0].attribute if attr.name == "equation")
        text = equation.s.decode("ascii")
        for old, new in substitutions.items():
            if text.count(old) != 1:
                raise RuntimeError((old, text))
            text = text.replace(old, new)
        equation.s = text.encode("ascii")
        onnx.checker.check_model(model, full_check=True)
        path = HERE / f"task050_common_k_{label}.onnx"
        onnx.save(model, path)
        print(path)


if __name__ == "__main__":
    main()

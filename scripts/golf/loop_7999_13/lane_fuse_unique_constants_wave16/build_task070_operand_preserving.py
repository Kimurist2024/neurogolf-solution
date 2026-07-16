#!/usr/bin/env python3
"""Probe task070 precontraction while retaining the original operand slot."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "submission_7999.13_wave16_candidate_meta.zip"
OUT = HERE / "task070_operand_preserving.onnx"


def main() -> None:
    with zipfile.ZipFile(BASE) as archive:
        model = onnx.load_model_from_string(archive.read("task070.onnx"))
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    product = np.einsum("as,kade->skde", arrays["U"], arrays["D"], optimize=False)
    product = np.asarray(product, dtype=np.float32)
    kept = [item for item in model.graph.initializer if item.name not in {"U", "D"}]
    kept.extend(
        [
            numpy_helper.from_array(np.ones((1,), dtype=np.float32), "U_dummy"),
            numpy_helper.from_array(product, "UD"),
        ]
    )
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    node = model.graph.node[0]
    node.input[2] = "U_dummy"
    node.input[16] = "UD"
    for attr in node.attribute:
        if attr.name == "equation":
            terms, output = attr.s.decode().split("->", 1)
            operands = terms.split(",")
            operands[2] = "a"
            operands[16] = "skde"
            attr.s = (",".join(operands) + "->" + output).encode()
            break
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()

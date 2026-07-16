#!/usr/bin/env python3
"""Fold task048's unit singleton-axis operand into existing coefficient shapes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "baseline" / "task048.onnx"
OUTPUT = HERE / "task048_fold_singleton_axes.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if arrays["einsum_cw"].shape != (10,) or arrays["einsum_ww"].shape != (30,):
        raise RuntimeError("unexpected coefficient shapes")
    if arrays["einsum_nm"].shape != (1, 1) or not np.array_equal(
        arrays["einsum_nm"], np.ones((1, 1), dtype=np.float32)
    ):
        raise RuntimeError("unexpected singleton operand")

    for init in model.graph.initializer:
        if init.name == "einsum_cw":
            init.CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(arrays["einsum_cw"].reshape(1, 10)), init.name)
            )
        elif init.name == "einsum_ww":
            init.CopyFrom(
                numpy_helper.from_array(np.ascontiguousarray(arrays["einsum_ww"].reshape(30, 1)), init.name)
            )
    kept = [item for item in model.graph.initializer if item.name != "einsum_nm"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)

    node = model.graph.node[0]
    if node.op_type != "Einsum" or list(node.input) != [
        "input",
        "einsum_cw",
        "einsum_ww",
        "einsum_nm",
    ]:
        raise RuntimeError("unexpected first Einsum")
    del node.input[:]
    node.input.extend(["input", "einsum_cw", "einsum_ww"])
    equation = next(attr for attr in node.attribute if attr.name == "equation")
    equation.CopyFrom(helper.make_attribute("equation", "bchw,nc,wm->bnhm"))

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    payload = {
        "candidate": str(OUTPUT.relative_to(ROOT)),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "proof": {
            "before": "einsum(input[bchw], cw[c], ww[w], one[nm]) -> bnhm",
            "after": "einsum(input[bchw], cw[nc], ww[wm]) -> bnhm",
            "coefficient_values_unchanged": True,
            "coefficient_element_counts_unchanged": True,
            "removed": "einsum_nm=float32[1,1]=1",
        },
        "parameter_reduction": 1,
        "node_delta": 0,
    }
    (HERE / "task048_fold_singleton_build.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

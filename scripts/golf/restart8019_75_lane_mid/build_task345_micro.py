#!/usr/bin/env python3
"""Build non-mutating task345 micro-variants from the strict cost-369 winner."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "candidates/task345_POLICY90_cost369_1b6b180284a6.onnx"
OUT = HERE / "task345_micro"


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name))
            return
    raise KeyError(name)


def build(*, mul: bool, float32: bool) -> onnx.ModelProto:
    model = copy.deepcopy(onnx.load(SOURCE))
    graph = model.graph
    prelu = graph.node[0]
    castlike = graph.node[1]
    assert prelu.op_type == "PRelu"
    assert castlike.op_type == "CastLike"

    if mul:
        prelu.op_type = "Mul"

    if float32:
        for name in ("Wpack__pb_axis3_Wpack", "Wpack__ps_axis3_Wpack"):
            initializer = next(item for item in graph.initializer if item.name == name)
            replace_initializer(model, name, numpy_helper.to_array(initializer).astype(np.float32))
        produced = prelu.output[0]
        cast_output = castlike.output[0]
        for node in graph.node[2:]:
            for index, value in enumerate(node.input):
                if value == cast_output:
                    node.input[index] = produced
        del graph.node[1]

    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for mul in (False, True):
        for float32 in (False, True):
            if not mul and not float32:
                continue
            name = f"task345_{'mul' if mul else 'prelu'}_{'f32_nocast' if float32 else 'f16_cast'}.onnx"
            onnx.save(build(mul=mul, float32=float32), OUT / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

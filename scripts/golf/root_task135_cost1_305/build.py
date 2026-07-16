#!/usr/bin/env python3
"""Build a finite one-parameter task135 crop using ConvTranspose output padding."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "task135_cost1.onnx"


def main() -> None:
    # Treat the dynamic benchmark tensor as ConvTranspose weights.  A scalar
    # activation exposes that 30x30 kernel once.  Begin/end pads crop the
    # wanted [rows 0:3, cols 6:9] window; output_padding restores the required
    # fixed 30x30 output with finite zeros below and to the right.
    x = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "x")
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 30, 30]
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "input"],
        ["output"],
        kernel_shape=[30, 30],
        strides=[28, 28],
        pads=[0, 6, 27, 21],
        output_padding=[27, 27],
    )
    graph = helper.make_graph([node], "task135_cost1", [inp], [out], [x])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=8,
        producer_name="codex-task135-cost1",
    )
    onnx.checker.check_model(model, full_check=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()

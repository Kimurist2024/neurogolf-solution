#!/usr/bin/env python3
"""Remove task200's redundant Conv background bias exactly at decode level."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "current/task200.onnx"
OUTPUT = HERE / "rejected/task200_zero_background_cost344.onnx"


def main() -> int:
    model = onnx.load(SOURCE)
    # The first Conv channel used +1 for background and -2 impulses for the
    # colored stripes.  Argmax's class-0 tie break lets background be zero, so
    # the same decoded rule is represented by zero bias and -1 impulses.
    for init in model.graph.initializer:
        if init.name == "W_kernel":
            array = numpy_helper.to_array(init)
            array = array.copy()
            array[0] /= 2
            init.CopyFrom(numpy_helper.from_array(array, init.name))

    for node in model.graph.node:
        if node.op_type == "Conv":
            # Bias is optional and is no longer needed by either channel.
            del node.input[2:]
        elif node.op_type == "Einsum" and list(node.output) == ["output"]:
            # Bbase's row sums are exactly [1, 0], the former conv_bias value.
            # Reusing it as an rc operand retains the width gate without a new
            # initializer: sum_c Bbase[r,c] == [1, 0].
            node.input[-1] = "Bbase"
            equation = next(attr for attr in node.attribute if attr.name == "equation")
            equation.s = b"tc,th,btw,rw,rk->bchw"

    del model.graph.initializer[
        next(
            index
            for index, init in enumerate(model.graph.initializer)
            if init.name == "conv_bias"
        )
    ]
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe the exact broadcast factor in task112's repeated sign tensor."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base/task112.onnx"


def set_shape(model: onnx.ModelProto, name: str, shape: list[int]) -> None:
    value = next(item for item in model.graph.value_info if item.name == name)
    del value.type.tensor_type.shape.dim[:]
    for size in shape:
        value.type.tensor_type.shape.dim.add().dim_value = size


def build(output: Path, truthful: bool) -> None:
    model = onnx.load(SOURCE)
    sign = next(init for init in model.graph.initializer if init.name == "sign4_ch_i8")
    array = numpy_helper.to_array(sign)
    assert array.shape == (3, 2, 1, 1)
    assert np.array_equal(array[0], array[1]) and np.array_equal(array[0], array[2])
    sign.CopyFrom(numpy_helper.from_array(array[:1].copy(), name=sign.name))

    if truthful:
        # The incumbent hides the runtime Slice shape as [1,1,4,4].  Expose
        # the true [3,2,4,4] shape so strict inference proves broadcasting.
        set_shape(model, "motif4", [3, 2, 4, 4])
    else:
        # Minimal metadata repair for the declared (cloaked) path.  This is
        # expected to be runtime-risky and is retained only as a rejection
        # probe, never as an automatic winner.
        set_shape(model, "updates", [1, 2, 4, 4])

    model.producer_name = "task112-broadcast-sign-probe"
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(model, output)


def main() -> None:
    build(HERE / "task112_broadcast_sign_cloaked.onnx", truthful=False)
    build(HERE / "task112_broadcast_sign_truthful.onnx", truthful=True)


if __name__ == "__main__":
    main()

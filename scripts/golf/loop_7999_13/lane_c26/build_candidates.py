#!/usr/bin/env python3
"""Build exact, parameter-only candidates for the C26 lane."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent


def replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name == name:
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(value.astype(np.float32), name=name)
            )
            return
    raise KeyError(name)


def build_task328_reuse_ninv() -> Path:
    """Reuse ``ninvB`` as z[0] and compensate in CoreI exactly.

    The incumbent concatenates ``one = [1]`` into z and every use of z is
    paired elementwise with CoreI on the same feature index.  Replacing z[0]
    by the already-present float32 value -1/3 and CoreI[:, 0] by -3 times its
    old value preserves their product.  In float32, (-1/3) * (-3) rounds to
    exactly 1.0.  No runtime node, tensor shape, or Einsum operand is added.
    """

    source = HERE / "base" / "task328.onnx"
    model = onnx.load(source)
    candidate = copy.deepcopy(model)

    initializers = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in candidate.graph.initializer
    }
    one = initializers["one"]
    ninv = initializers["ninvB"]
    core_i = initializers["CoreI"]
    assert one.shape == ninv.shape == (1,)
    assert one[0] == np.float32(1.0)
    assert ninv[0] == np.float32(-1.0 / 3.0)
    assert np.float32(ninv[0] * np.float32(-3.0)) == np.float32(1.0)

    concat = candidate.graph.node[6]
    assert concat.op_type == "Concat"
    assert list(concat.input) == ["one", "S", "NS", "SP", "SPN"]
    concat.input[0] = "ninvB"

    core_i[:, 0] *= np.float32(-3.0)
    replace_initializer(candidate, "CoreI", core_i)

    kept = [x for x in candidate.graph.initializer if x.name != "one"]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)

    onnx.checker.check_model(candidate, full_check=True)
    output = HERE / "task328_r01_reuse_ninv.onnx"
    onnx.save(candidate, output)
    return output


def build_task328_reuse_ninv_coreb() -> Path:
    """The same exact rescaling, applied to CoreB's feature-0 slice."""

    source = HERE / "base" / "task328.onnx"
    candidate = copy.deepcopy(onnx.load(source))
    initializers = {
        initializer.name: numpy_helper.to_array(initializer).copy()
        for initializer in candidate.graph.initializer
    }
    ninv = initializers["ninvB"]
    core_b = initializers["CoreB"]
    assert np.float32(ninv[0] * np.float32(-3.0)) == np.float32(1.0)

    concat = candidate.graph.node[6]
    assert concat.op_type == "Concat"
    concat.input[0] = "ninvB"
    core_b[:, :, 0] *= np.float32(-3.0)
    replace_initializer(candidate, "CoreB", core_b)

    kept = [x for x in candidate.graph.initializer if x.name != "one"]
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend(kept)
    onnx.checker.check_model(candidate, full_check=True)
    output = HERE / "task328_r02_reuse_ninv_coreb.onnx"
    onnx.save(candidate, output)
    return output


def main() -> None:
    for path in (build_task328_reuse_ninv(), build_task328_reuse_ninv_coreb()):
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replace task182's runtime shape-constant chain with honest initializers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline/task182.onnx"
OUTPUT = HERE / "task182_static_shapes.onnx"
REMOVED_OUTPUTS = {
    "sh5",
    "sh19",
    *(f"sh{size}" for size in range(20, 31)),
}


def axes_of(node: onnx.NodeProto) -> tuple[int, ...]:
    for attribute in node.attribute:
        if attribute.name == "axes":
            return tuple(attribute.ints)
    return ()


def replacement(name: str, axes: tuple[int, ...]) -> str:
    if name == "sh5":
        return "s5x2" if axes == (2, 3) else "s5"
    if name == "sh19":
        return "s19x2" if axes == (2, 3) else "s19"
    size = int(name.removeprefix("sh"))
    if axes == (2, 3):
        return f"s{size}x2"
    if len(axes) == 1:
        return f"s{size}"
    raise ValueError(f"unexpected use of {name} with axes={axes}")


def main() -> None:
    model = onnx.load(SOURCE)
    kept = [
        node
        for node in model.graph.node
        if not any(output in REMOVED_OUTPUTS for output in node.output)
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    existing = {initializer.name for initializer in model.graph.initializer}
    needed: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        axes = axes_of(node) if node.op_type == "CenterCropPad" else ()
        for index, name in enumerate(node.input):
            if name not in REMOVED_OUTPUTS:
                continue
            new_name = replacement(name, axes)
            node.input[index] = new_name
            if new_name in existing:
                continue
            size = int(new_name[1:].removesuffix("x2"))
            width = 2 if new_name.endswith("x2") else 1
            needed[new_name] = np.full((width,), size, dtype=np.int64)

    model.graph.initializer.extend(
        numpy_helper.from_array(value, name)
        for name, value in sorted(needed.items())
    )
    # The baseline carries intentionally false [1,1,1,1] intermediate
    # value_info entries.  They are not needed for execution and conflict with
    # the now-static honest shapes, so remove them rather than replacing one
    # shape cloak with another.
    del model.graph.value_info[:]

    output_shape = model.graph.output[0].type.tensor_type.shape
    for dimension, size in zip(output_shape.dim, (1, 10, 30, 30), strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = size

    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    print("removed_nodes", 13 - len([node for node in kept if any(o in REMOVED_OUTPUTS for o in node.output)]))
    print("added_initializers", {name: value.tolist() for name, value in sorted(needed.items())})


if __name__ == "__main__":
    main()

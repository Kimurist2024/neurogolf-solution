#!/usr/bin/env python3
"""Remove cost-shaping metadata and fix output declarations on C9 baselines."""

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
TASKS = (330, 354, 361, 364)


def set_shape(value: onnx.ValueInfoProto, shape: tuple[int, ...]) -> None:
    del value.type.tensor_type.shape.dim[:]
    for size in shape:
        value.type.tensor_type.shape.dim.add().dim_value = size


def main() -> None:
    for task in TASKS:
        model = onnx.load(HERE / "base" / f"task{task:03d}.onnx")
        del model.graph.value_info[:]
        set_shape(model.graph.output[0], (1, 10, 30, 30))
        path = HERE / f"task{task:03d}_decloaked.onnx"
        onnx.save(model, path)
        print(path)


if __name__ == "__main__":
    main()

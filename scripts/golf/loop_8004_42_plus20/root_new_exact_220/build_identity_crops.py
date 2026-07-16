#!/usr/bin/env python3
"""Remove three provably identity CenterCropPad operations from task341."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "base" / "task341.onnx"
OUTPUT = ROOT / "candidates" / "task341_identity_crops.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.initializer}

    expected = {
        "e186": np.asarray([1, 8, 6], dtype=np.int64),
        "e199": np.asarray([1, 9, 9], dtype=np.int64),
        "target_shape_src_i64": np.asarray([2, 10, 10], dtype=np.int64),
    }
    for name, value in expected.items():
        if not np.array_equal(initializers[name], value):
            raise RuntimeError(f"unexpected initializer {name}: {initializers[name]!r}")

    # Shape(e186, start=0, end=1) is the constant vector [3].  Each of the
    # three following CenterCropPad nodes therefore receives an already
    # length-three tensor and returns it unchanged.  Redirect every consumer
    # to the initializer and delete the four now-dead nodes.
    replacements = {
        "e186d": "e186",
        "e199d": "e199",
        "idxshape3_i64": "target_shape_src_i64",
    }
    removed_outputs = {"c3_i64", *replacements}
    kept = []
    for node in graph.node:
        if any(output in removed_outputs for output in node.output):
            continue
        for index, name in enumerate(node.input):
            if name in replacements:
                node.input[index] = replacements[name]
        kept.append(node)

    if len(graph.node) - len(kept) != 4:
        raise RuntimeError("expected to remove Shape plus three CenterCropPad nodes")
    del graph.node[:]
    graph.node.extend(kept)

    # The authority carries deliberately underspecified intermediate shapes.
    # Removing the indirection exposes constants to shape inference, so discard
    # stale value_info and make the graph output declaration runtime-truthful.
    del graph.value_info[:]
    output_shape = graph.output[0].type.tensor_type.shape
    del output_shape.dim[:]
    for value in (1, 10, 30, 30):
        output_shape.dim.add().dim_value = value
    onnx.checker.check_model(model, full_check=True)
    truthful_shapes = {
        "g": (onnx.TensorProto.FLOAT, [1, 10, 30, 30]),
        "x_i8": (onnx.TensorProto.INT8, [1, 10, 30, 30]),
        "g01": (onnx.TensorProto.FLOAT, [1, 10, 30, 30]),
        "x01_i8": (onnx.TensorProto.INT8, [1, 10, 30, 30]),
        "rm": (onnx.TensorProto.INT8, [1, 1, 6, 2]),
        "row_empty6": (onnx.TensorProto.INT8, [1, 1, 6, 1]),
        "cm": (onnx.TensorProto.INT8, [1, 1, 2, 6]),
        "col_empty6": (onnx.TensorProto.INT8, [1, 1, 1, 6]),
        "tb_i8": (onnx.TensorProto.INT8, [1, 1, 2, 8]),
        "lr_i8": (onnx.TensorProto.INT8, [1, 1, 8, 2]),
        "tb_bg_or6": (onnx.TensorProto.INT8, [1, 1, 1, 6]),
        "lr_bg_or6": (onnx.TensorProto.INT8, [1, 1, 6, 1]),
        "row_sig": (onnx.TensorProto.INT8, [1, 1, 6, 1]),
        "col_sig": (onnx.TensorProto.INT8, [1, 1, 1, 6]),
        "row_delta6": (onnx.TensorProto.INT8, [1, 2, 6, 1]),
        "core6": (onnx.TensorProto.INT8, [1, 2, 6, 6]),
        "upd10": (onnx.TensorProto.INT8, [1, 2, 10, 10]),
        "idx_full_i32": (onnx.TensorProto.INT32, [1, 10, 30, 30]),
        "idx86_i32": (onnx.TensorProto.INT32, [1, 1, 8, 6]),
        "idx10": (onnx.TensorProto.INT32, [1, 2, 10, 10]),
    }
    graph.value_info.extend(
        onnx.helper.make_tensor_value_info(name, dtype, shape)
        for name, (dtype, shape) in truthful_shapes.items()
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

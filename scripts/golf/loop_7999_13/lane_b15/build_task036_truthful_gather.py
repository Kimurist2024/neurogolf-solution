#!/usr/bin/env python3
"""Build a truthful-static task036 rule model using a fixed 5x5 GatherND.

Unlike the compact incumbents, this graph has no dynamic Slice and no false
value_info.  Every intermediate shape is statically honest.  Positions beyond
the tight bbox use feature value 1, the ConvInteger input zero point, so they
decode to zero-hot while background/target cells decode to channels 0/target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
OUT = HERE / "candidate_task036_truthful_gather.onnx"


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build() -> onnx.ModelProto:
    f = np.zeros((3, 30), dtype=np.float32)
    coordinates = np.arange(30, dtype=np.float32)
    f[0] = np.exp(3.0 * (9.0 - coordinates))
    f[1] = np.exp(3.0 * (coordinates - 20.0))
    f[2] = 1.0

    selector = np.zeros((3, 3, 1, 1), dtype=np.float32)
    selector[0, 1, 0, 0] = 1.0
    selector[2, 2, 0, 0] = -float(np.exp(-20.75))

    endpoint_projection = np.zeros((2, 3), dtype=np.float32)
    endpoint_projection[0, 0] = float(np.exp(60.0))
    endpoint_projection[1, 1] = float(np.exp(60.0))

    gather_indices = np.zeros((1, 1, 5, 5, 4), dtype=np.int64)
    for row in range(5):
        for col in range(5):
            gather_indices[0, 0, row, col] = (0, 0, row, col)

    initializers = [
        init("F", f),
        init("K", selector),
        init("depth10", np.array(10, dtype=np.int64)),
        init("ohvals", np.array([0.0, 1.0], dtype=np.float32)),
        init("div3", np.array(3, dtype=np.uint8)),
        init("twenty9", np.array([29], dtype=np.uint8)),
        init("one8", np.array([1], dtype=np.uint8)),
        init("zero8", np.array([0], dtype=np.uint8)),
        init("three8", np.array([3], dtype=np.uint8)),
        init("bgvec", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8).reshape(10, 1, 1, 1)),
        init("negone", np.array([[[[-1]]]], dtype=np.int8)),
        init("P2", endpoint_projection),
        init("gather_base", gather_indices),
        init("row5", np.arange(5, dtype=np.uint8).reshape(1, 1, 5, 1)),
        init("col5", np.arange(5, dtype=np.uint8).reshape(1, 1, 1, 5)),
    ]

    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "input", "F", "F", "K"],
            ["score"],
            equation="bchw,bcij,kw,lj,klxy->bcxy",
        ),
        helper.make_node("ArgMin", ["score"], ["target_idx"], axis=1, keepdims=1),
        helper.make_node("OneHot", ["target_idx", "depth10", "ohvals"], ["target_f"], axis=-1),
        helper.make_node(
            "Einsum",
            ["input", "target_f", "P2", "F"],
            ["cfeat"],
            equation="bchw,bdxyc,mk,kw->m",
        ),
        helper.make_node(
            "Einsum",
            ["input", "target_f", "P2", "F"],
            ["rfeat"],
            equation="bchw,bdxyc,mk,kh->m",
        ),
        helper.make_node("Log", ["cfeat"], ["clog"]),
        helper.make_node("Log", ["rfeat"], ["rlog"]),
        helper.make_node("Cast", ["clog"], ["cu8"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["rlog"], ["ru8"], to=TensorProto.UINT8),
        helper.make_node("Div", ["cu8", "div3"], ["cenc"]),
        helper.make_node("Div", ["ru8", "div3"], ["renc"]),
        helper.make_node("Split", ["renc"], ["rfirst", "rlast"], axis=0, num_outputs=2),
        helper.make_node("Split", ["cenc"], ["cfirst", "clast"], axis=0, num_outputs=2),
        helper.make_node("Sub", ["twenty9", "rfirst"], ["rmin"]),
        helper.make_node("Sub", ["twenty9", "cfirst"], ["cmin"]),
        helper.make_node("Add", ["rlast", "one8"], ["rend"]),
        helper.make_node("Add", ["clast", "one8"], ["cend"]),
        # Fixed-shape crop indices: [batch=0, channel=0, row, col] + bbox top-left.
        helper.make_node("Concat", ["zero8", "zero8", "rmin", "cmin"], ["offset8"], axis=0),
        helper.make_node("Cast", ["offset8"], ["offset64"], to=TensorProto.INT64),
        helper.make_node("Add", ["gather_base", "offset64"], ["gather_idx"]),
        helper.make_node("GatherND", ["input", "gather_idx"], ["bg_float"]),
        helper.make_node("Cast", ["bg_float"], ["bg_bool"], to=TensorProto.BOOL),
        helper.make_node("Where", ["bg_bool", "three8", "zero8"], ["inside_code"]),
        # Runtime bbox dimensions decide which of the fixed 5x5 cells are in-grid.
        helper.make_node("Sub", ["rend", "rmin"], ["height8"]),
        helper.make_node("Sub", ["cend", "cmin"], ["width8"]),
        helper.make_node("Less", ["row5", "height8"], ["valid_rows"]),
        helper.make_node("Less", ["col5", "width8"], ["valid_cols"]),
        helper.make_node("And", ["valid_rows", "valid_cols"], ["valid"]),
        helper.make_node("Where", ["valid", "inside_code", "one8"], ["feature"]),
        helper.make_node("ScatterElements", ["bgvec", "target_idx", "negone"], ["weights"], axis=0),
        helper.make_node(
            "ConvInteger",
            ["feature", "weights", "one8"],
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 25, 25],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task036_truthful_static_gather",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.INT32, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)

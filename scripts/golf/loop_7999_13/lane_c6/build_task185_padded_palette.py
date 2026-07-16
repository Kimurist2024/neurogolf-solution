#!/usr/bin/env python3
"""Rebuild task185's terminal colour decoder with a compact signed palette.

The exact baseline already extracts the three non-background colour channels
and eight cell codes correctly.  This rebuild preserves those learned token
tables, but maps channel ``c`` to signed label ``2*c-9``.  The padded label
``-9`` is therefore background.  Ten two-feature affine classifiers on
``(label, label**2)`` separate labels -9,-7,...,9 exactly, replacing ten bias
parameters with one output zero point and avoiding a four-element Reshape
shape tensor.  All scale cancellation is exact because the same positive
per-example token norm is used as input and output scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "base/task185.onnx"
OUTPUT = HERE / "task185_padded_palette.onnx"


def init(name: str, value: object, dtype: np.dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def main() -> None:
    source = onnx.load(SOURCE)
    source_nodes = list(source.graph.node)
    tfidf_nodes = [node for node in source_nodes if node.op_type == "TfIdfVectorizer"]
    assert len(tfidf_nodes) == 8

    moment = numpy_helper.to_array(
        next(item for item in source.graph.initializer if item.name == "mom")
    ).reshape(30)
    initializers = [
        init("mom", moment, np.float32),
        init("k3", [3], np.int64),
        init("bg_cell_4d", [[[[-9]]]], np.int8),
        init("zero_i8", 0, np.int8),
        init("one_f", 1.0, np.float32),
        init("map_w", [[[[2]]]], np.int8),
        init("map_zero_i8", -9, np.int8),
        init("palette_axes", [0, 1, 2], np.int64),
        init(
            "classifier_w",
            [
                [[[-5]], [[0]]],
                [[[-13]], [[-1]]],
                [[[-19]], [[-2]]],
                [[[-23]], [[-3]]],
                [[[-55]], [[-14]]],
                [[[55]], [[-14]]],
                [[[23]], [[-3]]],
                [[[19]], [[-2]]],
                [[[13]], [[-1]]],
                [[[5]], [[0]]],
            ],
            np.int8,
        ),
        init("output_zero_i8", -40, np.int8),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Einsum",
            ["input", "mom", "mom", "mom", "mom", "mom"],
            ["moment_all_f"],
            # Keep the exact baseline contraction order.  The TfIdf token
            # tables intentionally key off these float32 results.
            equation="bchw,h,h,h,w,w->c",
        ),
        helper.make_node(
            "TopK", ["moment_all_f", "k3"], ["mscore_f", "mch"],
            axis=0, largest=1, sorted=1,
        ),
        helper.make_node(
            "ReduceL2", ["mscore_f"], ["token_f"], keepdims=1,
        ),
        helper.make_node("Cast", ["token_f"], ["token_i32"], to=TensorProto.INT32),
        helper.make_node("Concat", ["token_i32", "token_i32"], ["tokseq"], axis=0),
        helper.make_node("Cast", ["mch"], ["mch_i8_1d"], to=TensorProto.INT8),
        helper.make_node("Unsqueeze", ["mch_i8_1d", "palette_axes"], ["mch_i8"]),
        helper.make_node(
            "QLinearConv",
            [
                "mch_i8", "token_f", "zero_i8", "map_w", "one_f", "zero_i8",
                "token_f", "map_zero_i8",
            ],
            ["palette"],
            pads=[0, 1, 0, 0],
        ),
    ]

    cell_names: list[str] = []
    for index, old in enumerate(tfidf_nodes):
        tfidf = onnx.NodeProto()
        tfidf.CopyFrom(old)
        del tfidf.input[:]
        tfidf.input.extend(["tokseq"])
        del tfidf.output[:]
        tfidf.output.extend([f"code{index}_f"])
        tfidf.name = ""
        nodes.extend(
            [
                tfidf,
                helper.make_node(
                    "Cast", [f"code{index}_f"], [f"code{index}_i32"],
                    to=TensorProto.INT32,
                ),
                helper.make_node(
                    "Gather", ["palette", f"code{index}_i32"],
                    [f"cell{index}_label"], axis=3,
                ),
            ]
        )
        cell_names.append(f"cell{index}_label")

    nodes.extend(
        [
            helper.make_node("Concat", cell_names[0:3], ["row0"], axis=3),
            helper.make_node(
                "Concat", [cell_names[3], "bg_cell_4d", cell_names[4]], ["row1"], axis=3,
            ),
            helper.make_node("Concat", cell_names[5:8], ["row2"], axis=3),
            helper.make_node("Concat", ["row0", "row1", "row2"], ["label"], axis=2),
            helper.make_node("Mul", ["label", "label"], ["square"]),
            helper.make_node("Concat", ["label", "square"], ["features"], axis=1),
            helper.make_node(
                "QLinearConv",
                [
                    "features", "token_f", "zero_i8", "classifier_w", "one_f",
                    "zero_i8", "token_f", "output_zero_i8",
                ],
                ["output"],
                pads=[0, 0, 27, 27],
                strides=[1, 1],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task185_padded_palette",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT8, [1, 10, 30, 30])],
        initializers,
    )
    model = helper.make_model(
        graph,
        # Opset 17 keeps ReduceL2's compact attribute-form axes.  Every copied
        # TfIdfVectorizer attribute and both QLinearConv nodes are compatible.
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="task185-padded-palette-from-exact-baseline",
    )
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build small constant-reuse/folding candidates without adding shape cloaks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent


def task182() -> Path:
    model = onnx.load(HERE / "baseline/task182.onnx")
    sh5 = next(node for node in model.graph.node if list(node.output) == ["sh5"])
    if sh5.op_type != "Identity" or list(sh5.input) != ["s5"]:
        raise RuntimeError("unexpected task182 sh5 producer")
    sh5.op_type = "Add"
    del sh5.input[:]
    sh5.input.extend(["s2", "s3"])

    sh19 = next(node for node in model.graph.node if list(node.output) == ["sh19"])
    sh20 = next(node for node in model.graph.node if list(node.output) == ["sh20"])
    if sh19.op_type != "Identity" or sh20.op_type != "Add":
        raise RuntimeError("unexpected task182 sh19/sh20 producers")
    sh19.op_type = "Mul"
    del sh19.input[:]
    sh19.input.extend(["s4", "sh5"])
    del sh19.output[:]
    sh19.output.append("sh20")
    sh20.op_type = "Sub"
    del sh20.input[:]
    sh20.input.extend(["sh20", "s1"])
    del sh20.output[:]
    sh20.output.append("sh19")

    neg_bias = next(node for node in model.graph.node if list(node.output) == ["detect_bias_scalar"])
    if neg_bias.op_type != "Mul" or list(neg_bias.input) != ["count_i32", "i32_neg1"]:
        raise RuntimeError("unexpected task182 bias producer")
    neg_bias.op_type = "Neg"
    del neg_bias.input[:]
    neg_bias.input.append("count_i32")

    cnt6 = next(node for node in model.graph.node if list(node.output) == ["cnt6"])
    if cnt6.op_type != "Mul" or list(cnt6.input) != ["col_count", "f6"]:
        raise RuntimeError("unexpected task182 cnt6 producer")
    cnt6.op_type = "Selu"
    del cnt6.input[:]
    cnt6.input.append("col_count")
    del cnt6.attribute[:]
    cnt6.attribute.extend(
        [helper.make_attribute("alpha", 1.0), helper.make_attribute("gamma", 6.0)]
    )

    removed = {"s5", "s19", "i32_neg1", "f6"}
    retained = [item for item in model.graph.initializer if item.name not in removed]
    if len(retained) + len(removed) != len(model.graph.initializer):
        raise RuntimeError("task182 initializer removal mismatch")
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output = HERE / "task182_reuse_constants.onnx"
    onnx.save(model, output)
    return output


def task238() -> Path:
    model = onnx.load(HERE / "baseline/task238.onnx")
    producer = next(node for node in model.graph.node if list(node.output) == ["one_i64"])
    if producer.op_type != "Div" or list(producer.input) != ["route8_target_i64", "route8_target_i64"]:
        raise RuntimeError("unexpected task238 one_i64 producer")
    retained = [node for node in model.graph.node if node is not producer]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one_i64")
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    output = HERE / "task238_fold_one_i64.onnx"
    onnx.save(model, output)
    return output


def main() -> None:
    print(task182())


if __name__ == "__main__":
    main()

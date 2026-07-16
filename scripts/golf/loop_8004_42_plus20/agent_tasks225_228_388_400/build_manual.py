#!/usr/bin/env python3
"""Build support-exact local candidates without touching authority/staging."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx
from onnx import helper

HERE = Path(__file__).resolve().parent


def drop_unused_initializers(model: onnx.ModelProto) -> None:
    uses = Counter(name for node in model.graph.node for name in node.input if name)
    keep = [item for item in model.graph.initializer if uses[item.name]]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def task388_q_bool_greater() -> None:
    model = onnx.load(HERE / "base" / "task388.onnx")
    node = model.graph.node[4]
    if node.op_type != "CastLike" or list(node.input) != ["q", "bool_ref"]:
        raise RuntimeError("latest authority task388 node4 drift")
    node.CopyFrom(helper.make_node("Greater", ["q", "eight_u8"], ["qb"]))
    drop_unused_initializers(model)
    out = HERE / "candidates" / "task388_q_gt8.onnx"
    onnx.save(model, out)


def task388_reduce_u8_then_greater() -> None:
    model = onnx.load(HERE / "base" / "task388.onnx")
    nodes = list(model.graph.node)
    if nodes[4].op_type != "CastLike" or nodes[9].op_type != "Slice":
        raise RuntimeError("latest authority task388 structure drift")
    # q is exactly {0,248} on the complete generator support.  Reducing the
    # cropped background channel first is equivalent to casting every cell to
    # bool first, while moving the bool-producing comparison to the <=6-cell
    # reduced vector.
    nodes[9].input[0] = "q"
    nodes[10].output[0] = "all_bg_u8"
    for value in model.graph.value_info:
        if value.name == "background":
            value.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    gt = helper.make_node("Greater", ["all_bg_u8", "eight_u8"], ["all_bg"])
    del model.graph.node[:]
    model.graph.node.extend(nodes[:4] + nodes[5:11] + [gt] + nodes[11:])
    drop_unused_initializers(model)
    out = HERE / "candidates" / "task388_reduce_u8_gt8.onnx"
    onnx.save(model, out)


if __name__ == "__main__":
    task388_q_bool_greater()
    task388_reduce_u8_then_greater()

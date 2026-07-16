#!/usr/bin/env python3
"""Repair the cheapest shape-truthful task096 lead for both ORT modes.

The historical cost-1128 graph passes ORT_DISABLE_ALL but feeds one-element
shape tensors to CenterCropPad nodes whose ``axes`` attribute has two axes.
ORT's default optimizer rejects that invalid contract.  This builder makes
each dynamic square size an explicit two-element vector, turns the constant
11 shape into ``[11, 11]``, and removes the now-unnecessary Identity carrier.

It deliberately does not touch the historical cost-1111 graph: that model's
apparent saving comes from false runtime shapes around dynamically constructed
QLinearConv weights.  Its bias already has ten elements; it is not a genuine
"10 output channels, length-1 bias" candidate.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "others/71402/task096_further_improved.onnx"
OUTPUT = HERE / "candidates/task096_default_safe.onnx"


def pair_node(value: str) -> onnx.NodeProto:
    return helper.make_node("Concat", [value, value], [f"{value}_pair"], axis=0)


def main() -> int:
    model = onnx.load(SOURCE)
    graph = model.graph

    shape_init = next(item for item in graph.initializer if item.name == "shape11_const1")
    shape_init.CopyFrom(
        numpy_helper.from_array(np.asarray([11, 11], dtype=np.int64), "shape11_const1")
    )

    repaired_nodes: list[onnx.NodeProto] = []
    for original in graph.node:
        node = copy.deepcopy(original)
        if node.op_type == "Identity" and list(node.output) == ["shape11_dyn"]:
            continue

        for index, name in enumerate(node.input):
            if name == "shape11_dyn":
                node.input[index] = "shape11_const1"

        if node.op_type == "CenterCropPad" and len(node.input) >= 2:
            if node.input[1] in {"qtarget", "qinter", "qplus3"}:
                node.input[1] = f"{node.input[1]}_pair"

        repaired_nodes.append(node)
        produced = set(node.output)
        for value in ("qtarget", "qinter", "qplus3"):
            if value in produced:
                repaired_nodes.append(pair_node(value))

    del graph.node[:]
    graph.node.extend(repaired_nodes)

    # Preserve scoreability for a diagnostic re-measure by replacing the
    # historical scalar/undersized declarations with the actual shapes on the
    # first known case.  These shapes vary with the generated radius, which is
    # precisely why this graph still cannot satisfy the campaign's truthful
    # static-shape gate; the diagnostic is not an adoption candidate.
    observed = {
        "featcrop": [1, 2, 7, 7],
        "featinter": [1, 2, 8, 8],
        "featp1": [1, 2, 9, 9],
        "featp2": [1, 2, 10, 10],
        "featcen": [1, 2, 11, 11],
    }
    for value in graph.value_info:
        if value.name not in observed:
            continue
        del value.type.tensor_type.shape.dim[:]
        for size in observed[value.name]:
            value.type.tensor_type.shape.dim.add().dim_value = size
    for value in graph.output:
        if value.name == "output":
            del value.type.tensor_type.shape.dim[:]
            for size in (1, 10, 30, 30):
                value.type.tensor_type.shape.dim.add().dim_value = size

    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

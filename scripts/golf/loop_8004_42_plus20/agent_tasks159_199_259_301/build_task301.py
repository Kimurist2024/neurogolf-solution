#!/usr/bin/env python3
"""Build a generator-support-exact task301 reduction.

For task beb8660c the cyan bar is always generated last and has length N,
while every other bar has length 1..N-1.  Therefore the per-channel maximum
``B`` is exactly the cyan count selected by the incumbent's Gather ``n``.
The incumbent expression

    B / n + (-n/2) - 1/2

is consequently exactly ``(B - 1) * -1/2`` on the complete generator
support.  The rewrite removes the Gather/index, Div, and one multiply/sum
chain without changing the final Einsum.
"""

from __future__ import annotations

import copy
from pathlib import Path

import onnx
from onnx import TensorProto, helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "current" / "task301.onnx"
OUTPUT = HERE / "rejected" / "task301_cyan_max_exact_REJECT.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph

    remove_outputs = {"n", "bn", "nh", "gp"}
    kept = [node for node in graph.node if not set(node.output) & remove_outputs]

    # Replace every use of the gathered cyan count by the already-computed
    # per-channel maximum.  This is generator-support exact by construction.
    for node in kept:
        for index, name in enumerate(node.input):
            if name == "n":
                node.input[index] = "B"

    b_node_index = next(
        index for index, node in enumerate(kept) if "B" in node.output
    )
    replacement = [
        helper.make_node(
            "Sum",
            ["B", "neg_half_f16", "neg_half_f16"],
            ["B_minus_one"],
            name="B_minus_one",
        ),
        helper.make_node(
            "Mul",
            ["B_minus_one", "neg_half_f16"],
            ["gp"],
            name="gp",
        ),
    ]
    kept[b_node_index + 1 : b_node_index + 1] = replacement

    del graph.node[:]
    graph.node.extend(kept)

    # ``axis_neg2`` is still shared by the three Pad nodes as their axes input,
    # so it must remain even though the Gather was removed.

    retained_vi = [
        value
        for value in graph.value_info
        if value.name not in {"n", "bn", "nh"}
    ]
    del graph.value_info[:]
    graph.value_info.extend(retained_vi)
    if not any(value.name == "B_minus_one" for value in graph.value_info):
        graph.value_info.append(
            helper.make_tensor_value_info(
                "B_minus_one", TensorProto.FLOAT16, [1, 1]
            )
        )

    model = onnx.shape_inference.infer_shapes(
        copy.deepcopy(model), strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(model, full_check=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

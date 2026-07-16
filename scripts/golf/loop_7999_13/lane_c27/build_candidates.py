#!/usr/bin/env python3
"""Build exact local probes for tasks 184 and 377.

These probes are deliberately written only under lane_c27.  They are not
promotion artifacts: the subsequent structural audit must still reject any
model that inherits the incumbents' false static shapes.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent


def build_task184_shrink() -> None:
    """Replace Add([1], [22]) by the exactly equivalent Shrink attribute."""

    source = HERE / "candidates" / "task184_sound422_r06.onnx"
    model = onnx.load(source)
    node = model.graph.node[3]
    assert node.op_type == "Add" and list(node.input) == ["sh0", "C23_i64"]
    replacement = helper.make_node(
        "Shrink",
        ["sh0"],
        list(node.output),
        name=node.name,
        bias=-22.0,
        lambd=0.0,
    )
    model.graph.node[3].CopyFrom(replacement)
    kept = [item for item in model.graph.initializer if item.name != "C23_i64"]
    assert len(kept) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, HERE / "candidates" / "task184_r06_shrink421.onnx")


def build_task377_diff5_witness() -> None:
    """Use sum(diff5)==1 as the singleton fp16 Einsum/type witness.

    Reshaping diff5 from [5,5] to [1,5,5] costs no parameters.  The leading
    singleton supplies the old k dimension, while summing its two matrix axes
    yields exactly one.  This deletes one_vec_f16 (one parameter) without
    changing any mathematical output.
    """

    model = onnx.load(HERE / "baseline" / "task377.onnx")
    inits = {item.name: item for item in model.graph.initializer}
    diff = numpy_helper.to_array(inits["diff5"])
    assert diff.shape == (5, 5) and float(diff.sum()) == 1.0
    inits["diff5"].CopyFrom(numpy_helper.from_array(diff.reshape(1, 5, 5), "diff5"))

    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "one_vec_f16":
                node.input[index] = "diff5"

    equations = {
        "min_area": "nc,nc,kij->nk",
        "min_moment": "nchw,h,nc,kij->nk",
        "line_f": "nchw,nh,nh,c,kij->nwk",
        "output": "njc,njc,ajk,kr,kr,kq,kq->ncrq",
    }
    for node in model.graph.node:
        if node.op_type != "Einsum" or not node.output:
            continue
        equation = equations.get(node.output[0])
        if equation is None:
            continue
        attr = next(item for item in node.attribute if item.name == "equation")
        attr.s = equation.encode("ascii")

    kept = [item for item in model.graph.initializer if item.name != "one_vec_f16"]
    assert len(kept) + 1 == len(model.graph.initializer)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, HERE / "candidates" / "task377_diff5_witness408.onnx")


def main() -> None:
    build_task184_shrink()
    build_task377_diff5_witness()


if __name__ == "__main__":
    main()

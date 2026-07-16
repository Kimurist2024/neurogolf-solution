#!/usr/bin/env python3
"""Compute task200's column directly from the one-hot input."""

from __future__ import annotations

from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "focus/candidates/task200_POLICY90_cost342_c659ae401e4c.onnx"
OUTPUT = HERE / "task200_direct_seed.onnx"


def main() -> int:
    model = onnx.load(SOURCE)
    graph = model.graph
    seed_node = graph.node[1]
    div_node = graph.node[2]
    if seed_node.op_type != "Einsum" or div_node.op_type != "Div":
        raise RuntimeError("unexpected task200 seed prefix")
    if list(seed_node.input) != ["input", "color_w", "width_w"]:
        raise RuntimeError(f"unexpected seed inputs: {list(seed_node.input)}")
    del seed_node.input[:]
    seed_node.input.extend(["input", "width_w"])
    equation = next(attr for attr in seed_node.attribute if attr.name == "equation")
    equation.s = b"bchw,wi->bi"
    seed_node.output[0] = "seed_f"
    del graph.node[2]
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

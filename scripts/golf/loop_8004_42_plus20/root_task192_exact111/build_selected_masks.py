#!/usr/bin/env python3
"""Exact task192 polynomial with the relation tensor factored dynamically.

For each relation row r the original contraction

    sum(d,a) input[d] * relation[r,d,a] * selected[a]

is either the all-color in-grid mask (r=0) or the selected-color mask (r=1).
Build those two rows directly as ``concat(ones, selected)``.  This removes the
200-element relation initializer at the cost of one 20-element intermediate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "task192_selected_masks.onnx"
EQUATION = "bchw,rc,bdhq,rd,qw,bepw,re,ph,ru,uo->bohw"


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def build(path: Path) -> None:
    adjacency = np.zeros((30, 30), dtype=np.float32)
    for i in range(30):
        adjacency[i, max(0, i - 1) : min(30, i + 2)] = 1.0

    color_masks = np.asarray(([1] * 10, [0] + [1] * 9), dtype=np.float32)
    hist_select = np.asarray((0, 1), dtype=np.float32)
    route = np.asarray(((1, 0), (-9, 1)), dtype=np.float32)
    background = np.asarray(([1] + [0] * 9,), dtype=np.float32)
    all_colors = np.ones((1, 10), dtype=np.float32)

    initializers = [
        tensor("adj", adjacency),
        tensor("color_masks", color_masks),
        tensor("hist_select", hist_select),
        tensor("route", route),
        tensor("background", background),
        tensor("all_colors", all_colors),
        tensor("depth", np.asarray(10, dtype=np.int64)),
        tensor("onehot_values", np.asarray((0.0, 1.0), dtype=np.float32)),
    ]
    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "color_masks", "hist_select"],
            ["hist"],
            equation="bchw,rc,r->c",
            name="nonzero_histogram",
        ),
        helper.make_node(
            "ArgMax", ["hist"], ["selected_i64"], axis=0, keepdims=1,
            name="lowest_color_tie_argmax",
        ),
        helper.make_node(
            "OneHot", ["selected_i64", "depth", "onehot_values"], ["selected"],
            axis=-1, name="selected_color_onehot",
        ),
        helper.make_node(
            "Concat", ["all_colors", "selected"], ["selected_masks"],
            axis=0, name="all_and_selected_color_masks",
        ),
        helper.make_node(
            "Concat", ["background", "selected"], ["output_basis"],
            axis=0, name="background_and_selected_output_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "color_masks",
                "input", "selected_masks", "adj",
                "input", "selected_masks", "adj",
                "route", "output_basis",
            ],
            ["output"],
            equation=EQUATION,
            name="exact_local_polynomial_and_output_route",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task192_selected_masks",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-selected-masks",
        ir_version=10,
    )
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    params = sum(int(np.prod(item.dims)) for item in model.graph.initializer)
    print(f"wrote {path} params={params} bytes={path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

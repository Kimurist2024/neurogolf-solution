#!/usr/bin/env python3
"""Build an all-input exact, one-output-Einsum task192 model.

Let A be the most frequent nonzero color and define, at each cell,

  P = NZ(center) * count_A(horizontal radius 1) * count_A(vertical radius 1)

All factors are nonnegative integers, so P>0 is exactly the decoded rule.
For the background channel use B-9P, where B is the product of the in-grid
horizontal/vertical counts. Since 1 <= B <= 9 and P>=1 when active, this is
nonpositive exactly when the selected-color output must be positive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "candidates" / "task192_exact_poly.onnx"
EQUATION = "bchw,rc,bdhq,rda,za,qw,bepw,ref,zf,ph,ru,uo->bohw"


def tensor(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name)


def build(path: Path) -> None:
    # Shared center-inclusive radius-one adjacency for rows and columns.
    adjacency = np.zeros((30, 30), dtype=np.float32)
    for i in range(30):
        adjacency[i, max(0, i - 1) : min(30, i + 2)] = 1.0

    # Relation 0 accepts every in-grid color; relation 1 accepts exactly A.
    # The relation is shared by the horizontal and vertical counts.
    relation = np.empty((2, 10, 10), dtype=np.float32)
    relation[0] = 1.0
    relation[1] = np.eye(10, dtype=np.float32)

    # Relation 0 keeps every in-grid center; relation 1 keeps nonzero centers.
    color_masks = np.asarray(
        ([1] * 10, [0] + [1] * 9), dtype=np.float32
    )
    hist_select = np.asarray((0, 1), dtype=np.float32)
    # r0 contributes +B to background.  r1 contributes -9P to background
    # and +P to the selected-color channel.
    route = np.asarray(((1, 0), (-9, 1)), dtype=np.float32)
    background = np.asarray(([1] + [0] * 9,), dtype=np.float32)

    initializers = [
        tensor("adj", adjacency),
        tensor("relation", relation),
        tensor("color_masks", color_masks),
        tensor("hist_select", hist_select),
        tensor("route", route),
        tensor("background", background),
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
            "Concat", ["background", "selected"], ["output_basis"],
            axis=0, name="background_and_selected_output_basis",
        ),
        helper.make_node(
            "Einsum",
            [
                "input", "color_masks",
                "input", "relation", "selected", "adj",
                "input", "relation", "selected", "adj",
                "route", "output_basis",
            ],
            ["output"],
            equation=EQUATION,
            name="exact_local_polynomial_and_output_route",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task192_exact_poly",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        producer_name="codex-task192-exact-poly",
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

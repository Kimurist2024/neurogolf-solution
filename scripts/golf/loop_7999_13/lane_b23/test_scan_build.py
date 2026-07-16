#!/usr/bin/env python3
"""Synthetic proof test for the B23 singleton-axis equation rewrite."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from scan_build import rewrite


def build_fixture() -> onnx.ModelProto:
    source = np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
    target = source.squeeze().transpose(1, 0).reshape(1, 3, 2, 1)
    node = helper.make_node(
        "Einsum",
        ["X", "target"],
        ["Y"],
        equation="ij,aijb->a",
    )
    graph = helper.make_graph(
        [node],
        "singleton_alias_fixture",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])],
        [
            numpy_helper.from_array(source, "source"),
            numpy_helper.from_array(target, "target"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def run(model: onnx.ModelProto, values: np.ndarray, disabled: bool) -> np.ndarray:
    options = ort.SessionOptions()
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session.run(None, {"X": values})[0]


def main() -> int:
    base = build_fixture()
    candidate, change = rewrite(base, "target", "source", (1, 0))
    initializers = {tensor.name for tensor in candidate.graph.initializer}
    assert initializers == {"source"}
    node = candidate.graph.node[0]
    assert list(node.input) == ["X", "source"]
    equation = next(attribute.s for attribute in node.attribute if attribute.name == "equation")
    assert equation == b"ij,aji->a"
    assert change["removed_parameters"] == 6
    assert change["nodes_added"] == 0
    assert change["runtime_tensors_added"] == 0
    assert change["einsum_operands_added"] == 0

    rng = np.random.default_rng(20260714)
    for disabled in (True, False):
        for _ in range(20):
            values = rng.normal(size=(3, 2)).astype(np.float32)
            np.testing.assert_array_equal(run(base, values, disabled), run(candidate, values, disabled))
    print("synthetic singleton-axis rewrite: PASS (40/40 dual-mode comparisons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

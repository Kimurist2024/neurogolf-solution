#!/usr/bin/env python3
"""Probe exact one-dimensional CenterCropPad shift compositions."""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


def run(targets: tuple[int, ...]) -> list[float]:
    nodes = []
    initializers = []
    current = "input"
    for index, target in enumerate(targets):
        target_name = f"target_{index}"
        output = f"v_{index}"
        initializers.append(numpy_helper.from_array(np.asarray([target], dtype=np.int64), target_name))
        nodes.append(helper.make_node("CenterCropPad", [current, target_name], [output], axes=[3]))
        current = output
    graph = helper.make_graph(
        nodes,
        "probe",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 6])],
        [helper.make_tensor_value_info(current, TensorProto.FLOAT, [1, 1, 1, targets[-1]])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    value = np.arange(1, 7, dtype=np.float32).reshape(1, 1, 1, 6)
    return session.run(None, {"input": value})[0].reshape(-1).tolist()


for targets in ((5, 7, 6), (7, 5, 6), (5, 6), (7, 6), (4, 6), (8, 6)):
    print(targets, run(targets))

for wanted in ((5, 7, 6), (7, 5, 6)):
    expected = run(wanted)
    matches = []
    for middle in range(1, 13):
        candidate = (middle, 6)
        if run(candidate) == expected:
            matches.append(candidate)
    print("two-op matches", wanted, matches)

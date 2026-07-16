#!/usr/bin/env python3
"""Replace task238's runtime-computed int64 one-vector by one initializer."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
source = HERE / "base/task238.onnx"
output = HERE / "task238_const_one_i64.onnx"
model = onnx.load(source)

removed = [node for node in model.graph.node if list(node.output) == ["one_i64"]]
assert len(removed) == 1 and removed[0].op_type == "Div", removed
kept = [node for node in model.graph.node if node is not removed[0]]
del model.graph.node[:]
model.graph.node.extend(kept)
model.graph.initializer.append(
    numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="one_i64")
)

# Making the axis an initializer lets shape inference see the true four side
# colours; the incumbent intentionally hides these two carrier shapes behind
# a runtime-computed axis.  Keep this probe checker-clean by declaring truth.
for value in model.graph.value_info:
    if value.name in {"side_indices_i8", "side_indices"}:
        dims = value.type.tensor_type.shape.dim
        assert len(dims) == 2
        dims[0].dim_value = 4
        dims[1].dim_value = 1

onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, output)
print(output)

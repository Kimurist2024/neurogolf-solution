#!/usr/bin/env python3
"""Compare task366 declared intermediate shapes with one real runtime trace."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def shape_of(value: onnx.ValueInfoProto):
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


original = onnx.load(HERE / "baseline_task366.onnx")
inferred = onnx.shape_inference.infer_shapes(
    copy.deepcopy(original), strict_mode=False, data_prop=True
)
typed = {
    value.name: value
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
}
declared = {
    value.name: shape_of(value)
    for value in list(original.graph.value_info) + list(original.graph.output)
}
model = copy.deepcopy(original)
del model.graph.output[:]
names = []
for node in model.graph.node:
    for name in node.output:
        if name and name in typed and name not in names:
            model.graph.output.append(copy.deepcopy(typed[name]))
            names.append(name)

options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.log_severity_level = 4
session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
example = scoring.convert_to_numpy(scoring.load_examples(366)["train"][0])
assert example is not None
outputs = session.run(names, {"input": example["input"]})
actual_arrays = {name: np.asarray(value) for name, value in zip(names, outputs)}
actual = {name: list(value.shape) for name, value in actual_arrays.items()}
mismatches = [
    {
        "tensor": name,
        "producer": next((node.op_type for node in original.graph.node if name in node.output), None),
        "declared": shape,
        "actual": actual[name],
        "actual_bytes": int(actual_arrays[name].nbytes),
    }
    for name, shape in declared.items()
    if name in actual and shape != actual[name]
]
report = {
    "runtime_tensor_count": len(actual),
    "declared_tensor_count": len(declared),
    "mismatch_count": len(mismatches),
    "mismatches": mismatches,
    "actual_intermediate_bytes_sum": int(
        sum(value.nbytes for name, value in actual_arrays.items() if name != "output")
    ),
}
(HERE / "runtime_shape_trace.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("runtime_tensor_count", "declared_tensor_count", "mismatch_count", "actual_intermediate_bytes_sum")}))

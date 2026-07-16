#!/usr/bin/env python3
"""Truthfully trace every task013 candidate node output without sanitize rewrites."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task013_r001.onnx"
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def main() -> None:
    model = onnx.load(MODEL)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    names = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(scoring.load_examples(13)["train"][0])
    assert benchmark is not None
    outputs = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = []
    nonfinite = []
    for name, output in zip(names, outputs):
        array = np.asarray(output)
        if dims(typed[name]) != list(array.shape):
            mismatches.append(
                {"name": name, "declared": dims(typed[name]), "actual": list(array.shape)}
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            nonfinite.append(name)
    payload = {
        "model": str(MODEL.relative_to(ROOT)),
        "node_outputs_traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_output_count": len(nonfinite),
        "nonfinite_outputs": nonfinite,
        "truthful": not mismatches and not nonfinite,
    }
    (HERE / "task013_runtime_shapes.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

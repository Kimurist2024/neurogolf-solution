#!/usr/bin/env python3
"""Audit declared versus actual intermediate shapes on one real task366 case."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

MODELS = {
    "authority_7987": HERE / "baseline_task366.onnx",
    "historical_7985": ROOT / "others/2/1203/task366_improved.onnx",
    "historical_7916": ROOT / "others/2/1203/task366_further_improved.onnx",
    "historical_7646": ROOT / "others/2/1201/7120/task366_further_improved.onnx",
}


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def audit(path: Path, benchmark: np.ndarray) -> dict[str, object]:
    original = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(original), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in [
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        ]
    }
    declared = {
        value.name: shape_of(value)
        for value in [*original.graph.value_info, *original.graph.output]
    }
    model = copy.deepcopy(original)
    del model.graph.output[:]
    names: list[str] = []
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
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    values = session.run(names, {"input": benchmark})
    actual = {name: list(np.asarray(value).shape) for name, value in zip(names, values)}
    mismatches = [
        {"tensor": name, "declared": shape, "actual": actual[name]}
        for name, shape in declared.items()
        if name in actual and shape != actual[name]
    ]
    return {
        "path": str(path.relative_to(ROOT)),
        "node_count": len(original.graph.node),
        "center_crop_pad_count": sum(
            node.op_type == "CenterCropPad" for node in original.graph.node
        ),
        "declared_tensor_count": len(declared),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def main() -> None:
    example = scoring.convert_to_numpy(scoring.load_examples(366)["train"][0])
    assert example is not None
    report = {
        label: audit(path, example["input"]) for label, path in MODELS.items()
    }
    (HERE / "historical_shape_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

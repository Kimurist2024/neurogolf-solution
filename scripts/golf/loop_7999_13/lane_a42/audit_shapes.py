#!/usr/bin/env python3
"""Compare declared and runtime shapes for task196 authority/history."""

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
    "authority": HERE / "baseline_task196.onnx",
    "historical_968": ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task196_r07_static296.onnx",
    "truthful_authority": HERE / "truthful_authority.onnx",
    "truthful_historical_968": HERE / "truthful_historical_968.onnx",
}


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def audit(path: Path, benchmark: np.ndarray) -> tuple[dict[str, object], onnx.ModelProto]:
    original = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(original), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    declared = {
        value.name: shape(value)
        for value in [*original.graph.value_info, *original.graph.output]
    }
    instrumented = copy.deepcopy(original)
    del instrumented.graph.output[:]
    names: list[str] = []
    for node in instrumented.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                instrumented.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        instrumented.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    outputs = session.run(names, {"input": benchmark})
    arrays = {name: np.asarray(value) for name, value in zip(names, outputs)}
    actual = {name: list(value.shape) for name, value in arrays.items()}
    mismatches = [
        {
            "tensor": name,
            "producer": next(
                (node.op_type for node in original.graph.node if name in node.output), None
            ),
            "declared": dims,
            "actual": actual[name],
            "actual_bytes": int(arrays[name].nbytes),
        }
        for name, dims in declared.items()
        if name in actual and dims != actual[name]
    ]
    truthful = copy.deepcopy(original)
    value_map = {
        value.name: value
        for value in [*truthful.graph.value_info, *truthful.graph.output]
    }
    for name, dims in actual.items():
        value = value_map.get(name)
        if value is None:
            continue
        del value.type.tensor_type.shape.dim[:]
        for dimension in dims:
            value.type.tensor_type.shape.dim.add().dim_value = dimension
    onnx.checker.check_model(truthful, full_check=True)
    onnx.shape_inference.infer_shapes(truthful, strict_mode=True, data_prop=True)
    return (
        {
            "path": str(path.relative_to(ROOT)),
            "declared_count": len(declared),
            "runtime_count": len(actual),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "actual_intermediate_bytes": int(
                sum(value.nbytes for name, value in arrays.items() if name != "output")
            ),
        },
        truthful,
    )


def main() -> None:
    example = scoring.convert_to_numpy(scoring.load_examples(196)["train"][0])
    assert example is not None
    reports = {}
    for label, path in MODELS.items():
        reports[label], truthful = audit(path, example["input"])
        if not label.startswith("truthful_"):
            onnx.save(truthful, HERE / f"truthful_{label}.onnx")
    (HERE / "shape_audit.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Record runtime tensor shapes and compare them with all declared shapes."""

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


TASKS = (14, 36, 75, 159, 218, 225, 245)


def declared_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def trace(task: int) -> dict[str, object]:
    path = HERE / "base" / f"task{task:03d}.onnx"
    original = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(original), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    declared = {
        value.name: declared_shape(value)
        for value in list(original.graph.output) + list(original.graph.value_info)
    }
    model = copy.deepcopy(original)
    del model.graph.output[:]
    output_names: list[str] = []
    for node in model.graph.node:
        for name in node.output:
            if name and name in typed and name not in output_names:
                model.graph.output.append(copy.deepcopy(typed[name]))
                output_names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    example = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert example is not None
    outputs = session.run(output_names, {"input": example["input"]})
    actual_arrays = {name: np.asarray(value) for name, value in zip(output_names, outputs)}
    actual = {name: list(value.shape) for name, value in actual_arrays.items()}
    original_outputs = {value.name for value in original.graph.output}
    actual_tensor_bytes = {
        name: int(value.nbytes) for name, value in actual_arrays.items()
    }
    mismatches = [
        {"tensor": name, "declared": shape, "actual": actual.get(name)}
        for name, shape in declared.items()
        if name in actual and shape != actual[name]
    ]
    undeclared = [
        {"tensor": name, "actual": actual[name]}
        for name in output_names
        if name not in declared
    ]
    return {
        "task": task,
        "declared_count": len(declared),
        "runtime_tensor_count": len(actual),
        "mismatches": mismatches,
        "undeclared_intermediates": undeclared,
        "all_actual_shapes": actual,
        "actual_tensor_bytes": actual_tensor_bytes,
        "single_example_truthful_intermediate_bytes": sum(
            nbytes
            for name, nbytes in actual_tensor_bytes.items()
            if name not in original_outputs
        ),
    }


def main() -> None:
    results: dict[str, object] = {}
    for task in TASKS:
        try:
            result = trace(task)
        except Exception as exc:  # noqa: BLE001
            result = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
        results[str(task)] = result
        print(
            task,
            "mismatches", len(result.get("mismatches", [])),
            "undeclared", len(result.get("undeclared_intermediates", [])),
            "error", result.get("error"),
            flush=True,
        )
    (HERE / "runtime_shape_trace.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

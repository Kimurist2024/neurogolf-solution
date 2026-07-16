#!/usr/bin/env python3
"""Trace actual shapes for the lane baselines on a known example."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (23, 187, 209, 367)


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value)
        if dim.HasField("dim_value")
        else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def trace(task: int) -> dict[str, object]:
    source = onnx.load(HERE / f"baseline_task{task:03d}.onnx")
    inferred = onnx.shape_inference.infer_shapes(
        copy.deepcopy(source), strict_mode=False, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    }
    declared = {
        value.name: shape_of(value)
        for value in list(source.graph.output) + list(source.graph.value_info)
    }
    probe = copy.deepcopy(source)
    del probe.graph.output[:]
    output_names = []
    for node in probe.graph.node:
        for name in node.output:
            if name and name in typed and name not in output_names:
                probe.graph.output.append(copy.deepcopy(typed[name]))
                output_names.append(name)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        probe.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    case = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    outputs = session.run(output_names, {"input": case["input"]})
    runtime = {
        name: {"shape": list(np.asarray(value).shape), "bytes": np.asarray(value).nbytes}
        for name, value in zip(output_names, outputs)
    }
    original_outputs = {item.name for item in source.graph.output}
    return {
        "task": task,
        "truthful_memory": sum(
            value["bytes"]
            for name, value in runtime.items()
            if name not in original_outputs
        ),
        "mismatches": [
            {"name": name, "declared": declared[name], "actual": value["shape"]}
            for name, value in runtime.items()
            if name in declared and declared[name] != value["shape"]
        ],
        "runtime": runtime,
    }


def main() -> None:
    report = {}
    for task in TASKS:
        row = trace(task)
        report[str(task)] = row
        print(
            f"task{task:03d}: truthful_memory={row['truthful_memory']} "
            f"mismatches={len(row['mismatches'])}"
        )
        for mismatch in row["mismatches"]:
            print(" ", mismatch)
    (HERE / "baseline_runtime_trace.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

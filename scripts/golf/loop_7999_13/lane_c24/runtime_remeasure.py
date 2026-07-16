#!/usr/bin/env python3
"""Remeasure exact-model intermediate bytes on maximum-size legal fixtures."""

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
sys.path.insert(0, str(ROOT))

from scripts.lib import scoring  # noqa: E402
from scripts.golf.loop_7999_13.lane_c24.fresh_exact_audit import (  # noqa: E402
    suppress_native_stderr,
)


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def trace(task: int, subset: str, index: int) -> dict[str, object]:
    original = onnx.load(HERE / "base" / f"task{task}.onnx")
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
        value.name: dims(value)
        for value in list(original.graph.output) + list(original.graph.value_info)
    }
    traced = copy.deepcopy(original)
    original_outputs = {value.name for value in traced.graph.output}
    del traced.graph.output[:]
    names: list[str] = []
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                traced.graph.output.append(copy.deepcopy(typed[name]))
                names.append(name)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    example = scoring.load_examples(task)[subset][index]
    benchmark = scoring.convert_to_numpy(example)
    assert benchmark is not None
    with suppress_native_stderr():
        session = ort.InferenceSession(
            traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        values = session.run(names, {traced.graph.input[0].name: benchmark["input"]})
    arrays = {name: np.asarray(value) for name, value in zip(names, values)}
    shapes = {name: list(value.shape) for name, value in arrays.items()}
    mismatches = [
        {"tensor": name, "declared": shape, "actual": shapes[name]}
        for name, shape in declared.items()
        if name in shapes and shape != shapes[name]
    ]
    intermediate_bytes = sum(
        int(value.nbytes)
        for name, value in arrays.items()
        if name not in original_outputs
    )
    params = scoring.calculate_params(original)
    return {
        "task": task,
        "fixture": {"subset": subset, "index": index},
        "grid_size": [len(example["input"]), len(example["input"][0])],
        "declared_cost": scoring.score_and_verify(
            copy.deepcopy(original), task, "/tmp", f"c24_remeasure_{task}",
            require_correct=False,
        )["cost"],
        "runtime_intermediate_bytes": intermediate_bytes,
        "params": params,
        "runtime_probe_cost": intermediate_bytes + params,
        "declared_actual_mismatches": mismatches,
    }


def main() -> None:
    # task363 is fixed at 10x10; task388 train[1] is its generator maximum 6x6.
    result = {
        "task363": trace(363, "train", 0),
        "task388": trace(388, "train", 1),
    }
    (HERE / "runtime_remeasure.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for task, record in result.items():
        print(
            task,
            "declared_cost",
            record["declared_cost"],
            "runtime_probe_cost",
            record["runtime_probe_cost"],
            "shape_mismatches",
            len(record["declared_actual_mismatches"]),
        )


if __name__ == "__main__":
    main()

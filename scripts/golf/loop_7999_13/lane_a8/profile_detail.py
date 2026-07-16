#!/usr/bin/env python3
"""Record per-output profiler shapes for exact A8 baseline members."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def run(task: int) -> None:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    model = onnx.load(path)
    for node in model.graph.node:
        node.name = node.output[0]
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        value.name: value.type.tensor_type
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    examples = scoring.load_examples(task)
    example = next(
        scoring.convert_to_numpy(item)
        for key in ("train", "test", "arc-gen")
        for item in examples[key]
        if scoring.convert_to_numpy(item) is not None
    )
    payload: dict[str, object] = {"task": task, "modes": {}}
    for mode, level in (
        ("disable_all", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("default", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
    ):
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(HERE / f"profile_task{task:03d}_{mode}")
        options.graph_optimization_level = level
        session = ort.InferenceSession(model.SerializeToString(), options)
        session.run(["output"], {"input": example["input"]})
        trace_path = Path(session.end_profiling())
        trace = json.loads(trace_path.read_text())
        trace_path.unlink(missing_ok=True)
        rows: dict[str, dict[str, object]] = {}
        for event in trace:
            if event.get("cat") != "Node" or "output_type_shape" not in event.get("args", {}):
                continue
            node_name = str(event.get("name", "")).removesuffix("_kernel_time")
            node = next((node for node in model.graph.node if node.name == node_name), None)
            if node is None:
                continue
            for index, shape_dict in enumerate(event["args"]["output_type_shape"]):
                if index >= len(node.output) or not node.output[index]:
                    continue
                name = node.output[index]
                tensor_type = types.get(name)
                if tensor_type is None:
                    continue
                itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)).itemsize
                shapes = [dims for dims in shape_dict.values()]
                memory = itemsize * sum(math.prod(dims) for dims in shapes)
                rows[name] = {
                    "op": node.op_type,
                    "shapes": shapes,
                    "bytes": memory,
                    "dtype": helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name,
                }
        payload["modes"][mode] = {
            "runtime_error_count": 0,
            "total_intermediate_bytes": sum(
                int(row["bytes"]) for name, row in rows.items() if name != "output"
            ),
            "outputs": rows,
        }
    (HERE / f"profile_task{task:03d}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["modes"], indent=2))


if __name__ == "__main__":
    run(int(sys.argv[1]))

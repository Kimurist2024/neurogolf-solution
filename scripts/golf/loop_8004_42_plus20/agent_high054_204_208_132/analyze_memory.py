#!/usr/bin/env python3
"""Attribute official memory to individual tensors for lane 132 baselines."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def analyze(task: int) -> dict[str, object]:
    source = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
    model = scoring.sanitize_model(copy.deepcopy(source))
    assert model is not None
    original_name = {
        sanitized_output: original_output
        for original_node, sanitized_node in zip(source.graph.node, model.graph.node)
        for original_output, sanitized_output in zip(original_node.output, sanitized_node.output)
    }
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    value_map = {
        item.name: item
        for item in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    output_to_op = {
        output: node.op_type
        for node in inferred.graph.node
        for output in node.output
        if output
    }
    static, dtype_name, static_shape = {}, {}, {}
    for name, value in value_map.items():
        if name in {"input", "output"}:
            continue
        tensor = value.type.tensor_type
        shape = [int(dim.dim_value) for dim in tensor.shape.dim]
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        static[name] = int(math.prod(shape) * dtype.itemsize)
        dtype_name[name] = dtype.name
        static_shape[name] = shape
    runtime: dict[str, int] = defaultdict(int)
    runtime_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    node_outputs = {node.name: list(node.output) for node in inferred.graph.node}
    with tempfile.TemporaryDirectory(prefix=f"lane132_mem_{task}_", dir="/tmp") as workdir:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.profile_file_prefix = os.path.join(workdir, f"task{task:03d}")
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        errors = 0
        examples = scoring.load_examples(task)
        for example in examples["train"] + examples["test"] + examples["arc-gen"]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                session.run(["output"], {"input": benchmark["input"]})
            except Exception:  # noqa: BLE001
                errors += 1
        trace = json.loads(Path(session.end_profiling()).read_text())
        for event in trace:
            if event.get("cat") != "Node" or "args" not in event:
                continue
            shapes = event["args"].get("output_type_shape")
            if not shapes:
                continue
            outputs = node_outputs.get(event.get("name", "").removesuffix("_kernel_time"), [])
            for index, shape_dict in enumerate(shapes):
                if index >= len(outputs) or outputs[index] not in static:
                    continue
                name = outputs[index]
                shape = tuple(int(dim) for dims in shape_dict.values() for dim in dims)
                runtime_shapes[name].add(shape)
                runtime[name] = max(
                    runtime[name],
                    int(math.prod(shape) * np.dtype(dtype_name[name]).itemsize),
                )
    charged = {name: max(size, runtime.get(name, 0)) for name, size in static.items()}
    by_op: dict[str, int] = defaultdict(int)
    for name, size in charged.items():
        by_op[output_to_op.get(name, "value_info_only")] += size
    top = sorted(charged, key=charged.get, reverse=True)
    return {
        "task": task,
        "known_run_errors": errors,
        "memory": sum(charged.values()),
        "params": scoring.calculate_params(model),
        "memory_by_op": dict(sorted(by_op.items(), key=lambda item: item[1], reverse=True)),
        "top_intermediates": [
            {
                "name": name,
                "original_name": original_name.get(name, name),
                "op": output_to_op.get(name, "value_info_only"),
                "dtype": dtype_name[name],
                "static_shape": static_shape[name],
                "runtime_shapes": [list(shape) for shape in sorted(runtime_shapes.get(name, set()))],
                "charged_bytes": charged[name],
            }
            for name in top
        ],
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {str(task): analyze(task) for task in (54, 204, 208)}
    (HERE / "audit/memory_anatomy.json").write_text(json.dumps(result, indent=2) + "\n")
    for task, row in result.items():
        print(task, row["memory"], row["params"], row["memory_by_op"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce official costs and attribute C8 memory/params to graph objects."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (54, 209, 367)


def tensor_meta(value: onnx.ValueInfoProto) -> tuple[list[int], str, int]:
    tensor_type = value.type.tensor_type
    shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
    return shape, np.dtype(dtype).name, int(math.prod(shape) * np.dtype(dtype).itemsize)


def analyze(task: int) -> dict[str, object]:
    source = onnx.load(HERE / "base" / f"task{task:03d}.onnx")
    model = scoring.sanitize_model(copy.deepcopy(source))
    assert model is not None
    onnx.checker.check_model(model, full_check=True)
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
    static: dict[str, int] = {}
    dtype_name: dict[str, str] = {}
    static_shape: dict[str, list[int]] = {}
    for name, value in value_map.items():
        if name in {"input", "output"}:
            continue
        shape, dtype, nbytes = tensor_meta(value)
        static[name] = nbytes
        dtype_name[name] = dtype
        static_shape[name] = shape

    runtime: dict[str, int] = defaultdict(int)
    runtime_shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    node_outputs = {node.name: list(node.output) for node in inferred.graph.node}
    with tempfile.TemporaryDirectory(prefix="c8_profile_", dir="/tmp") as workdir:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.profile_file_prefix = os.path.join(workdir, f"task{task:03d}")
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        run_errors = 0
        examples = scoring.load_examples(task)
        for example in examples["train"] + examples["test"] + examples["arc-gen"]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                session.run(["output"], {"input": benchmark["input"]})
            except Exception:  # noqa: BLE001
                run_errors += 1
        trace_path = Path(session.end_profiling())
        trace = json.loads(trace_path.read_text())
        for event in trace:
            if event.get("cat") != "Node" or "args" not in event:
                continue
            shapes = event["args"].get("output_type_shape")
            if not shapes:
                continue
            node_name = event.get("name", "").removesuffix("_kernel_time")
            for index, shape_dict in enumerate(shapes):
                outputs = node_outputs.get(node_name, [])
                if index >= len(outputs):
                    continue
                name = outputs[index]
                if name not in static:
                    continue
                shape = tuple(
                    int(dim)
                    for dims in shape_dict.values()
                    for dim in dims
                )
                runtime_shapes[name].add(shape)
                itemsize = np.dtype(dtype_name[name]).itemsize
                runtime[name] = max(runtime[name], int(math.prod(shape) * itemsize))

    charged = {name: max(nbytes, runtime.get(name, 0)) for name, nbytes in static.items()}
    by_op: dict[str, int] = defaultdict(int)
    by_dtype: dict[str, int] = defaultdict(int)
    for name, nbytes in charged.items():
        by_op[output_to_op.get(name, "value_info_only")] += nbytes
        by_dtype[dtype_name[name]] += nbytes
    top = sorted(charged, key=lambda name: charged[name], reverse=True)
    params = scoring.calculate_params(model)
    assert params is not None
    init_rows = []
    for initializer in model.graph.initializer:
        count = int(math.prod(initializer.dims)) if initializer.dims else 1
        init_rows.append(
            {
                "name": initializer.name,
                "count": count,
                "dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
                "shape": list(initializer.dims),
            }
        )
    init_rows.sort(key=lambda row: int(row["count"]), reverse=True)
    banned = [
        node.op_type
        for node in model.graph.node
        if node.op_type.upper()
        in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
        or "Sequence" in node.op_type
    ]
    nested = sum(
        1
        for node in model.graph.node
        for attr in node.attribute
        if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
    )
    return {
        "task": task,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "opset": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "op_histogram": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "known_profile_run_errors": run_errors,
        "memory": sum(charged.values()),
        "params": params,
        "cost": sum(charged.values()) + params,
        "memory_by_op": dict(sorted(by_op.items(), key=lambda item: item[1], reverse=True)),
        "memory_by_dtype": dict(sorted(by_dtype.items(), key=lambda item: item[1], reverse=True)),
        "top_intermediates": [
            {
                "name": name,
                "op": output_to_op.get(name, "value_info_only"),
                "dtype": dtype_name[name],
                "static_shape": static_shape[name],
                "runtime_shapes": [list(shape) for shape in sorted(runtime_shapes.get(name, set()))],
                "static_bytes": static[name],
                "runtime_max_bytes": runtime.get(name, 0),
                "charged_bytes": charged[name],
            }
            for name in top[:25]
        ],
        "top_initializers": init_rows[:20],
        "full_check": True,
        "strict_shape_inference": True,
        "banned_ops": banned,
        "nested_graph_attributes": nested,
        "nonstandard_domains": [
            item.domain for item in model.opset_import if item.domain not in {"", "ai.onnx"}
        ],
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {str(task): analyze(task) for task in TASKS}
    (HERE / "baseline_anatomy.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for task, row in result.items():
        print(task, row["memory"], row["params"], row["cost"], row["memory_by_op"])


if __name__ == "__main__":
    main()

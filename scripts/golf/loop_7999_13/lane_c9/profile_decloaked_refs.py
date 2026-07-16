#!/usr/bin/env python3
"""Measure actual runtime tensor bytes after removing C9 shape metadata."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


TASKS = (330, 354, 361, 364)
ITEMSIZE = {
    "bool": 1,
    "float": 4,
    "float16": 2,
    "double": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}


def profile(task: int) -> dict[str, object]:
    source = onnx.load(HERE / f"task{task:03d}_decloaked.onnx")
    model = scoring.sanitize_model(copy.deepcopy(source))
    assert model is not None
    node_outputs = {node.name: list(node.output) for node in model.graph.node}
    with tempfile.TemporaryDirectory(prefix=f"c9_decloak_{task}_", dir="/tmp") as tmp:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        options.profile_file_prefix = os.path.join(tmp, "trace")
        session = ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        right = wrong = errors = 0
        examples = scoring.load_examples(task)
        for example in examples["train"] + examples["test"] + examples["arc-gen"]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                actual = session.run(["output"], {"input": benchmark["input"]})[0] > 0
                if (actual == (benchmark["output"] > 0)).all():
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
        trace = json.loads(Path(session.end_profiling()).read_text())

    charged: dict[str, int] = {}
    shapes: dict[str, list[dict[str, list[int]]]] = {}
    for event in trace:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        output_shapes = event["args"].get("output_type_shape")
        if not output_shapes:
            continue
        node_name = event.get("name", "").removesuffix("_kernel_time")
        outputs = node_outputs.get(node_name, [])
        for index, shape_dict in enumerate(output_shapes):
            if index >= len(outputs):
                continue
            name = outputs[index]
            if not name or name == "output":
                continue
            nbytes = 0
            for dtype, dims in shape_dict.items():
                nbytes += math.prod(dims) * ITEMSIZE[dtype]
            charged[name] = max(charged.get(name, 0), nbytes)
            shapes.setdefault(name, []).append(shape_dict)
    params = scoring.calculate_params(model)
    assert params is not None
    expected_outputs = {
        output
        for node in model.graph.node
        for output in node.output
        if output and output != "output"
    }
    missing = sorted(expected_outputs - charged.keys())
    return {
        "task": task,
        "known": {"right": right, "wrong": wrong, "errors": errors},
        "warning": "Profiler output_type_shape is absent for some operators; this is a partial lower bound, not an official score.",
        "profiled_intermediates_seen": len(charged),
        "outputs_missing_profile_shapes": len(missing),
        "missing_output_names": missing,
        "profile_reported_memory_lower_bound": sum(charged.values()),
        "params": params,
        "profile_reported_cost_lower_bound": sum(charged.values()) + params,
        "top_runtime_tensors": [
            {"name": name, "bytes": nbytes, "shapes": shapes[name][:3]}
            for name, nbytes in sorted(charged.items(), key=lambda item: item[1], reverse=True)[:20]
        ],
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    results: dict[str, object] = {}
    for task in TASKS:
        results[str(task)] = profile(task)
        print(task, json.dumps(results[str(task)], sort_keys=True), flush=True)
    (HERE / "decloaked_runtime_audit.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

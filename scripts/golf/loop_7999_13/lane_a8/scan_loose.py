#!/usr/bin/env python3
"""Static screen of loose historical ONNX files for lane A8 targets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (12, 19, 34, 35, 46, 117, 125)
BASE_COST = {12: 710, 19: 536, 34: 511, 35: 545, 46: 631, 117: 606, 125: 1050}


def static_cost(model: onnx.ModelProto) -> tuple[int, int, int]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    init_names = {item.name for item in graph.initializer}
    io_names = {item.name for item in graph.input} | {item.name for item in graph.output}
    values = {
        item.name: item
        for item in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    names = {name for node in graph.node for name in node.output if name}
    names.update(values)
    memory = 0
    for name in names:
        if name in init_names or name in io_names:
            continue
        item = values.get(name)
        if item is None or not item.type.HasField("tensor_type"):
            raise ValueError(f"missing tensor metadata: {name}")
        tensor_type = item.type.tensor_type
        dims = tensor_type.shape.dim
        if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims):
            raise ValueError(f"non-static shape: {name}")
        dtype = helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        memory += math.prod(int(dim.dim_value) for dim in dims) * np.dtype(dtype).itemsize
    params = sum(math.prod(item.dims) for item in graph.initializer)
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return memory + params, memory, params


def main() -> None:
    all_paths = list(ROOT.rglob("*.onnx"))
    rows: dict[int, list[dict[str, object]]] = {task: [] for task in TARGETS}
    seen: dict[int, set[str]] = {task: set() for task in TARGETS}
    counts = {task: {"paths": 0, "unique": 0, "checked": 0, "cheaper": 0} for task in TARGETS}
    for path in all_paths:
        rel = str(path.relative_to(ROOT))
        components = set(path.relative_to(ROOT).parts[:-1])
        filename = path.name.lower()
        task = next(
            (
                task
                for task in TARGETS
                if f"task{task:03d}" in components or filename.startswith(f"task{task:03d}")
            ),
            None,
        )
        if task is None:
            continue
        counts[task]["paths"] += 1
        try:
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if sha in seen[task]:
                continue
            seen[task].add(sha)
            counts[task]["unique"] += 1
            model = onnx.load_model_from_string(data)
            onnx.checker.check_model(model, full_check=True)
            cost, memory, params = static_cost(model)
            counts[task]["checked"] += 1
        except Exception:
            continue
        if cost >= BASE_COST[task]:
            continue
        counts[task]["cheaper"] += 1
        rows[task].append(
            {
                "task": task,
                "path": rel,
                "sha256": sha,
                "static_cost": cost,
                "static_memory": memory,
                "params": params,
                "nodes": len(model.graph.node),
                "value_info": len(model.graph.value_info),
            }
        )
    for task in TARGETS:
        rows[task].sort(key=lambda row: (int(row["static_cost"]), str(row["path"])))
    payload = {"baseline_cost": BASE_COST, "counts": counts, "cheaper": rows}
    (HERE / "loose_static_screen.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"counts": counts, "cheaper": {k: len(v) for k, v in rows.items()}}, indent=2))


if __name__ == "__main__":
    main()

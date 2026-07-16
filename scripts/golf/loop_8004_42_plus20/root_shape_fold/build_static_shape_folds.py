#!/usr/bin/env python3
"""Fold Shape nodes whose source has a strict, static inferred shape.

This is narrower than arbitrary constant folding: the input contract is fixed
and every selected source shape must be fully known after strict data-propagating
shape inference. Runtime/gold checks remain mandatory because some historical
incumbents carry stale shape annotations that affect ORT allocation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402

BASE_ZIP = ROOT / "submission_base_8004.50.zip"
TASKS = (233, 131, 107, 137, 234, 278, 308, 397, 81)


def dims(value: onnx.ValueInfoProto) -> list[int] | None:
    if not value.type.HasField("tensor_type"):
        return None
    result: list[int] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        result.append(int(dim.dim_value))
    return result


def attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    return next((int(attr.i) for attr in node.attribute if attr.name == name), default)


def fold(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    values = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    candidate = copy.deepcopy(model)
    kept: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    removed_value_info: set[str] = set()
    changes: list[dict[str, object]] = []
    graph_outputs = {value.name for value in candidate.graph.output}
    initializer_names = {value.name for value in candidate.graph.initializer}
    for index, node in enumerate(candidate.graph.node):
        if node.op_type != "Shape" or len(node.input) != 1 or len(node.output) != 1:
            kept.append(node)
            continue
        output = node.output[0]
        if not output or output in graph_outputs or output in initializer_names:
            kept.append(node)
            continue
        source_shape = dims(values.get(node.input[0])) if values.get(node.input[0]) is not None else None
        if source_shape is None:
            kept.append(node)
            continue
        rank = len(source_shape)
        start = attr_int(node, "start", 0)
        end = attr_int(node, "end", rank)
        if start < 0:
            start += rank
        if end < 0:
            end += rank
        start = min(max(start, 0), rank)
        end = min(max(end, 0), rank)
        value = np.asarray(source_shape[start:end], dtype=np.int64)
        new_initializers.append(numpy_helper.from_array(value, name=output))
        removed_value_info.add(output)
        changes.append({
            "node_index": index,
            "source": node.input[0],
            "source_shape": source_shape,
            "output": output,
            "value": value.tolist(),
        })
    if not changes:
        return candidate, []
    del candidate.graph.node[:]
    candidate.graph.node.extend(kept)
    candidate.graph.initializer.extend(new_initializers)
    value_info = [value for value in candidate.graph.value_info if value.name not in removed_value_info]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(value_info)
    onnx.checker.check_model(candidate, full_check=True)
    shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    return candidate, changes


def measure(model: onnx.ModelProto, task: int, label: str) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"shape_fold_{task:03d}_{label}_") as tmp:
        path = Path(tmp) / "model.onnx"
        onnx.save(model, path)
        memory, params, total = cost_of(str(path))
        return int(memory), int(params), int(total)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            try:
                base = onnx.load_from_string(archive.read(f"task{task:03d}.onnx"))
                candidate, changes = fold(base)
                if not changes:
                    rows.append({"task": task, "status": "no_foldable_shape"})
                    continue
                base_memory, base_params, base_cost = measure(base, task, "base")
                memory, params, total = measure(candidate, task, "candidate")
                row: dict[str, object] = {
                    "task": task,
                    "status": "not_cheaper",
                    "base": {"memory": base_memory, "params": base_params, "cost": base_cost},
                    "candidate": {"memory": memory, "params": params, "cost": total},
                    "changes": changes,
                    "gain": math.log(base_cost / total) if 0 < total < base_cost else 0.0,
                }
                if 0 < total < base_cost:
                    path = HERE / f"task{task:03d}_static_shape_fold.onnx"
                    onnx.save(candidate, path)
                    row.update({
                        "status": "cost_winner_pending_correctness",
                        "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
                rows.append(row)
            except Exception as exc:  # noqa: BLE001
                rows.append({"task": task, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            (HERE / "build_manifest.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    document = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": hashlib.sha256(BASE_ZIP.read_bytes()).hexdigest(),
        "rows": rows,
        "winners": sum(row.get("status") == "cost_winner_pending_correctness" for row in rows),
        "nominal_gain": sum(float(row.get("gain", 0.0)) for row in rows),
    }
    (HERE / "build_manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))


if __name__ == "__main__":
    main()

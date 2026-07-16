#!/usr/bin/env python3
"""Inventory loose historical task121/task315 models for lane C29."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
TARGETS = {121: 125, 315: 128}


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [int(d.dim_value) if d.HasField("dim_value") else None for d in value.type.tensor_type.shape.dim]


def static_cost(model: onnx.ModelProto) -> tuple[int, int, int] | None:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        init_names = {item.name for item in inferred.graph.initializer}
        params = sum(int(np.prod(item.dims, dtype=np.int64)) for item in inferred.graph.initializer)
        memory = 0
        seen: set[str] = set()
        for value in list(inferred.graph.value_info) + list(inferred.graph.output):
            if value.name in seen or value.name in init_names or value.name in {"input", "output"}:
                continue
            seen.add(value.name)
            tensor = value.type.tensor_type
            shape = dims(value)
            if any(d is None or d <= 0 for d in shape):
                return None
            memory += int(np.prod(shape, dtype=np.int64)) * np.dtype(
                helper.tensor_dtype_to_np_dtype(tensor.elem_type)
            ).itemsize
        return memory + params, memory, params
    except Exception:
        return None


def main() -> None:
    rows: dict[str, object] = {"tasks": {}}
    for task, base_cost in TARGETS.items():
        files: set[Path] = set()
        needle = f"task{task:03d}"
        for root in (ROOT / "others", ROOT / "artifacts", ROOT / "inputs", ROOT / "scripts" / "golf"):
            if root.exists():
                files.update(path for path in root.rglob("*.onnx") if needle in path.name.lower())
        unique: dict[str, dict[str, object]] = {}
        errors = 0
        for path in sorted(files):
            try:
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if digest in unique:
                    unique[digest]["copies"] = int(unique[digest]["copies"]) + 1
                    continue
                model = onnx.load_model_from_string(payload)
                try:
                    onnx.checker.check_model(model, full_check=True)
                    checker = True
                except Exception:
                    checker = False
                try:
                    inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
                    strict = True
                    inferred_output = dims(inferred.graph.output[0])
                    all_static = all(
                        d is not None and d > 0
                        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
                        for d in dims(value)
                    )
                except Exception:
                    strict = False
                    inferred_output = []
                    all_static = False
                einsums = []
                for node in model.graph.node:
                    if node.op_type != "Einsum":
                        continue
                    equation = next((attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation"), "")
                    einsums.append({"operands": len(node.input), "equation_length": len(equation)})
                cost = static_cost(model)
                unique[digest] = {
                    "path": str(path.relative_to(ROOT)),
                    "copies": 1,
                    "bytes": len(payload),
                    "checker": checker,
                    "strict_inference": strict,
                    "all_static": all_static,
                    "declared_input": dims(model.graph.input[0]),
                    "declared_output": dims(model.graph.output[0]),
                    "inferred_output": inferred_output,
                    "nodes": len(model.graph.node),
                    "ops": dict(Counter(node.op_type for node in model.graph.node)),
                    "einsums": einsums,
                    "cost": None if cost is None else cost[0],
                    "memory": None if cost is None else cost[1],
                    "params": None if cost is None else cost[2],
                }
            except Exception:
                errors += 1
        values = list(unique.values())
        values.sort(key=lambda row: (row["cost"] is None, row["cost"] or 10**9, row["path"]))
        rows["tasks"][str(task)] = {
            "baseline_cost": base_cost,
            "files_seen": len(files),
            "unique_hashes": len(values),
            "errors": errors,
            "cheaper_count": sum(row["cost"] is not None and row["cost"] < base_cost for row in values),
            "models": values,
        }
    (HERE / "history_inventory.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    for task, report in rows["tasks"].items():
        print(task, {key: value for key, value in report.items() if key != "models"})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Deduplicate and structurally screen historical task068/task383 models."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, shape_inference


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("history_manifest.json")
TARGETS = {68: "task068", 383: "task383"}
GIANT_OPERANDS = 12
GIANT_EQUATION = 60


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    result: list[int | None] = []
    for dim in value.type.tensor_type.shape.dim:
        result.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
    return result


def static_cost(model: onnx.ModelProto) -> int | None:
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception:
        return None
    initializers = {item.name for item in inferred.graph.initializer}
    params = sum(int(np.prod(item.dims, dtype=np.int64)) for item in inferred.graph.initializer)
    memory = 0
    seen: set[str] = set()
    for value in list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.name in seen or value.name in initializers or value.name in {"input", "output"}:
            continue
        seen.add(value.name)
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            return None
        elements = 1
        for dim in tensor.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return None
            elements *= int(dim.dim_value)
        try:
            itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.elem_type)).itemsize
        except Exception:
            return None
        memory += elements * itemsize
    return int(params + memory)


def paths_for(stem: str) -> list[Path]:
    roots = [ROOT / "scripts" / "golf", ROOT / "artifacts", ROOT / "others"]
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.onnx"):
            if stem in path.name.lower():
                paths.add(path)
    return sorted(paths)


def main() -> None:
    report: dict[str, object] = {"incumbent_cost": 172, "tasks": {}}
    for task, stem in TARGETS.items():
        by_hash: dict[str, dict[str, object]] = {}
        parse_errors = 0
        all_paths = paths_for(stem)
        for path in all_paths:
            try:
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if digest in by_hash:
                    by_hash[digest]["source_count"] = int(by_hash[digest]["source_count"]) + 1
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
                        dim is not None and dim > 0
                        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
                        for dim in dims(value)
                    )
                except Exception:
                    strict = False
                    inferred_output = []
                    all_static = False
                ops = Counter(node.op_type for node in model.graph.node)
                giant = any(
                    node.op_type == "Einsum"
                    and (
                        len(node.input) >= GIANT_OPERANDS
                        or any(attr.name == "equation" and len(attr.s) >= GIANT_EQUATION for attr in node.attribute)
                    )
                    for node in model.graph.node
                )
                output_shape = dims(model.graph.output[0])
                truthful = output_shape == [1, 10, 30, 30] and inferred_output == [1, 10, 30, 30]
                strict_safe = bool(
                    checker
                    and strict
                    and all_static
                    and truthful
                    and not giant
                    and ops["CenterCropPad"] == 0
                )
                by_hash[digest] = {
                    "path": str(path.relative_to(ROOT)),
                    "source_count": 1,
                    "checker": checker,
                    "strict_inference": strict,
                    "all_static": all_static,
                    "output_shape": output_shape,
                    "inferred_output_shape": inferred_output,
                    "center_crop_pad": ops["CenterCropPad"],
                    "giant_einsum": giant,
                    "nodes": len(model.graph.node),
                    "params": sum(int(np.prod(item.dims, dtype=np.int64)) for item in model.graph.initializer),
                    "static_cost": static_cost(model),
                    "strict_safe": strict_safe,
                }
            except Exception:
                parse_errors += 1
        rows = list(by_hash.values())
        safe = [row for row in rows if row["strict_safe"]]
        cheaper_safe = [row for row in safe if row["static_cost"] is not None and int(row["static_cost"]) < 172]
        report["tasks"][str(task)] = {
            "files_seen": len(all_paths),
            "unique_hashes": len(rows),
            "parse_errors": parse_errors,
            "truthful_no_cloak_no_giant": len(safe),
            "truthful_no_cloak_no_giant_static_below_172": len(cheaper_safe),
            "best_truthful_no_cloak_no_giant_static": min(
                (int(row["static_cost"]) for row in safe if row["static_cost"] is not None),
                default=None,
            ),
            "cheaper_safe_candidates": cheaper_safe,
            "unique_models": rows,
        }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for task, row in report["tasks"].items():
        print(task, {key: value for key, value in row.items() if key not in {"unique_models", "cheaper_safe_candidates"}})


if __name__ == "__main__":
    main()

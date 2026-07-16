#!/usr/bin/env python3
"""Extract and structurally inventory exact task023/task036 baselines."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_7999.13.zip"


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    out: list[int | str] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            out.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            out.append(dimension.dim_param)
        else:
            out.append("?")
    return out


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(task: int, path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    equations: list[str] = []
    for node in model.graph.node:
        if node.op_type != "Einsum":
            continue
        for attribute in node.attribute:
            if attribute.name == "equation":
                value = helper.get_attribute_value(attribute)
                equations.append(value.decode() if isinstance(value, bytes) else str(value))
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(path.read_bytes()),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "node_list": [
            {
                "index": index,
                "op": node.op_type,
                "domain": node.domain,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
            for index, node in enumerate(model.graph.node)
        ],
        "initializers_list": [
            {
                "name": item.name,
                "dtype": str(numpy_helper.to_array(item).dtype),
                "shape": list(numpy_helper.to_array(item).shape),
                "elements": int(numpy_helper.to_array(item).size),
            }
            for item in model.graph.initializer
        ],
        "declared_value_info": [
            {"name": item.name, "dtype": item.type.tensor_type.elem_type, "shape": dims(item)}
            for item in model.graph.value_info
        ],
        "inferred_value_info": [
            {"name": item.name, "dtype": item.type.tensor_type.elem_type, "shape": dims(item)}
            for item in inferred.graph.value_info
        ],
        "einsum_equations": equations,
    }


def main() -> int:
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in (23, 36):
            name = f"task{task:03d}.onnx"
            (baseline / name).write_bytes(archive.read(name))
    report = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": digest(BASE_ZIP.read_bytes()),
        "models": {
            str(task): describe(task, baseline / f"task{task:03d}.onnx")
            for task in (23, 36)
        },
    }
    (HERE / "baseline_structure.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({task: report["models"][str(task)]["sha256"] for task in (23, 36)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

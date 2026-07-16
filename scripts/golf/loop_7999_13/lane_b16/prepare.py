#!/usr/bin/env python3
"""Extract and inventory exact B16 baselines without modifying the submission."""

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
TASKS = (157, 319)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value)
        if dim.HasField("dim_value")
        else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def describe(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    equations: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        equation = ""
        for attribute in node.attribute:
            if attribute.name == "equation":
                value = helper.get_attribute_value(attribute)
                equation = value.decode() if isinstance(value, bytes) else str(value)
        equations.append(
            {"node": index, "inputs": len(node.input), "equation": equation}
        )
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "opsets": [
            {"domain": item.domain, "version": item.version}
            for item in model.opset_import
        ],
        "io": {
            "input": [
                {"name": value.name, "shape": shape(value)}
                for value in model.graph.input
            ],
            "output": [
                {"name": value.name, "shape": shape(value)}
                for value in model.graph.output
            ],
        },
        "nodes_list": [
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
        "inferred_values": [
            {"name": value.name, "shape": shape(value)}
            for value in inferred.graph.value_info
        ],
        "einsum": equations,
    }


def main() -> int:
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            (baseline / name).write_bytes(archive.read(name))
    report = {
        "base_zip": str(BASE_ZIP.relative_to(ROOT)),
        "base_zip_sha256": sha256(BASE_ZIP),
        "models": {
            str(task): describe(baseline / f"task{task:03d}.onnx")
            for task in TASKS
        },
    }
    (HERE / "baseline_structure.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                task: report["models"][str(task)]["sha256"]
                for task in TASKS
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

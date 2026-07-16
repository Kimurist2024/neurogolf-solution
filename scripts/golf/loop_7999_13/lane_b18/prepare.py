#!/usr/bin/env python3
"""Extract the exact task089/task255 models from the 7999.13 baseline."""

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
TASKS = (89, 255)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def describe(path: Path) -> dict[str, object]:
    model = onnx.load(path)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
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
                "equation": next(
                    (
                        value.decode() if isinstance(value, bytes) else str(value)
                        for attribute in node.attribute
                        if attribute.name == "equation"
                        for value in [helper.get_attribute_value(attribute)]
                    ),
                    None,
                ),
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

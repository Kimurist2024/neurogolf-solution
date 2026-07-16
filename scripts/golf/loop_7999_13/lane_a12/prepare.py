#!/usr/bin/env python3
"""Extract exact A12 baselines and record structural model summaries."""

from __future__ import annotations

import collections
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_7999.13.zip"
TASKS = (198, 200, 201, 219, 302, 343)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str]:
    out: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            out.append(dim.dim_param)
        else:
            out.append("")
    return out


def summary(path: Path, kind: str, task: int) -> dict[str, object]:
    model = onnx.load(path)
    max_einsum = max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0)
    initializers = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        initializers.append(
            {
                "name": item.name,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "elements": int(array.size),
                "unique": int(np.unique(array).size),
            }
        )
    return {
        "task": task,
        "kind": kind,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "ops": dict(collections.Counter(node.op_type for node in model.graph.node)),
        "params": sum(int(numpy_helper.to_array(x).size) for x in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "max_einsum_inputs": max_einsum,
        "input_shape": dims(model.graph.input[0]),
        "output_shape": dims(model.graph.output[0]),
        "initializers": initializers,
        "opset": [[op.domain, int(op.version)] for op in model.opset_import],
    }


def main() -> None:
    (HERE / "baseline").mkdir(parents=True, exist_ok=True)
    models: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(archive.read(f"task{task:03d}.onnx"))
            models.append(summary(path, "exact_baseline", task))
    inventory = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json").read_text()
    )
    for task in TASKS:
        for item in inventory["retained"].get(str(task), []):
            path = ROOT / item["path"]
            models.append(summary(path, "retained_candidate", task))
    report = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": sha(BASE_ZIP),
        "models": models,
    }
    (HERE / "model_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

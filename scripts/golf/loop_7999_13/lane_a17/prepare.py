#!/usr/bin/env python3
"""Extract exact A17 baselines and summarize retained models."""

from __future__ import annotations

import collections
import hashlib
import json
import zipfile
from pathlib import Path

import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
TASKS = (29, 51, 195, 397, 400)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(path: Path, task: int, kind: str) -> dict[str, object]:
    model = onnx.load(path)
    return {
        "task": task, "kind": kind, "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "bytes": path.stat().st_size, "nodes": len(model.graph.node),
        "ops": dict(collections.Counter(node.op_type for node in model.graph.node)),
        "params": sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "max_einsum_inputs": max((len(node.input) for node in model.graph.node if node.op_type == "Einsum"), default=0),
        "opset": [[item.domain, int(item.version)] for item in model.opset_import],
    }


def main() -> None:
    (HERE / "baseline").mkdir(parents=True, exist_ok=True)
    models: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(archive.read(f"task{task:03d}.onnx"))
            models.append(summary(path, task, "exact_baseline"))
    inventory = json.loads((ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json").read_text())
    for task in TASKS:
        for item in inventory["retained"].get(str(task), []):
            models.append(summary(ROOT / item["path"], task, "retained_candidate"))
    (HERE / "model_manifest.json").write_text(json.dumps({"baseline_zip": str(BASE.relative_to(ROOT)), "baseline_zip_sha256": sha(BASE), "models": models}, indent=2) + "\n")


if __name__ == "__main__":
    main()

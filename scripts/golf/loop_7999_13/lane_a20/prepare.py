#!/usr/bin/env python3
"""Prepare exact A20 baselines and all retained task191/task216 variants."""

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
BASE_ZIP = ROOT / "submission_base_7999.13.zip"
SOURCE = ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep"
INVENTORY = SOURCE / "inventory.json"
TASKS = (191, 216)
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summary(path: Path, task: int, label: str) -> dict[str, object]:
    model = onnx.load(path)
    data = path.read_bytes()
    return {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "params": sum(
            int(numpy_helper.to_array(item).size) for item in model.graph.initializer
        ),
        "value_info": len(model.graph.value_info),
        "ops": dict(collections.Counter(node.op_type for node in model.graph.node)),
        "opset": [[item.domain, int(item.version)] for item in model.opset_import],
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
    }


def main() -> None:
    base_sha = sha(BASE_ZIP.read_bytes())
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"wrong base archive: {base_sha}")
    (HERE / "baseline").mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(parents=True, exist_ok=True)
    models = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(archive.read(f"task{task:03d}.onnx"))
            models.append(summary(path, task, "exact_baseline"))
    inventory = json.loads(INVENTORY.read_text())
    selected: dict[str, object] = {}
    for task in TASKS:
        for index, item in enumerate(inventory["retained"][str(task)], 1):
            label = f"task{task:03d}_r{index:02d}"
            source = ROOT / item["path"]
            path = HERE / "candidates" / f"{label}.onnx"
            path.write_bytes(source.read_bytes())
            if sha(path.read_bytes()) != item["sha256"]:
                raise RuntimeError(f"hash mismatch: {label}")
            models.append(summary(path, task, label))
            selected[label] = item
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "baseline_zip_sha256": base_sha,
                "models": models,
                "inventory_entries": selected,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

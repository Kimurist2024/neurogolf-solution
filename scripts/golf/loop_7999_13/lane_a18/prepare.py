#!/usr/bin/env python3
"""Prepare exact 7999.13 baselines and the six A18 archive candidates."""

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
ARCHIVE = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400"
TASKS = (63, 73, 139, 202)
CANDIDATES = (
    (63, "r01", "task063_r01_static24.onnx"),
    (73, "r01", "task073_r01_static15.onnx"),
    (73, "r02", "task073_r02_static15.onnx"),
    (139, "r04", "task139_r04_static50.onnx"),
    (202, "r02", "task202_r02_static28.onnx"),
    (202, "r03", "task202_r03_static28.onnx"),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize(path: Path, task: int, variant: str, kind: str) -> dict[str, object]:
    model = onnx.load(path)
    data = path.read_bytes()
    return {
        "task": task,
        "variant": variant,
        "kind": kind,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(data),
        "bytes": len(data),
        "nodes": len(model.graph.node),
        "ops": dict(collections.Counter(node.op_type for node in model.graph.node)),
        "params": sum(
            int(numpy_helper.to_array(item).size) for item in model.graph.initializer
        ),
        "value_info": len(model.graph.value_info),
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "max_einsum_inputs": max(
            (
                len(node.input)
                for node in model.graph.node
                if node.op_type == "Einsum"
            ),
            default=0,
        ),
        "opset": [[item.domain, int(item.version)] for item in model.opset_import],
    }


def main() -> None:
    baseline_dir = HERE / "baseline"
    candidate_dir = HERE / "candidates"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            data = archive.read(f"task{task:03d}.onnx")
            path = baseline_dir / f"task{task:03d}.onnx"
            path.write_bytes(data)
            models.append(summarize(path, task, "base", "exact_baseline"))
    for task, variant, filename in CANDIDATES:
        source = ARCHIVE / filename
        destination = candidate_dir / f"task{task:03d}_{variant}.onnx"
        destination.write_bytes(source.read_bytes())
        models.append(summarize(destination, task, variant, "archive_candidate"))
    inventory = json.loads((ARCHIVE / "inventory.json").read_text())
    selected = {
        f"task{task:03d}_{variant}": next(
            item
            for item in inventory["retained"][str(task)]
            if item["sha256"]
            == sha((candidate_dir / f"task{task:03d}_{variant}.onnx").read_bytes())
        )
        for task, variant, _ in CANDIDATES
    }
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
                "baseline_zip_sha256": sha(BASE_ZIP.read_bytes()),
                "models": models,
                "inventory_entries": selected,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

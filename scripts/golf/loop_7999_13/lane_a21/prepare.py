#!/usr/bin/env python3
"""Prepare exact A21 baselines, retained history, and sound reference rebuilds."""

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
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/inventory.json"
TASKS = (285, 286)
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
SOUND = {
    "task285_spec_bitboard": ROOT / "scripts/golf/scratch_codex/task285/agent_hub_algebra/candidate_columnfull_bitboard.onnx",
    "task286_spec_fullrow": ROOT / "scripts/golf/scratch_codex/task286/cand_groundup_fullrow_recheck.onnx",
    "task286_spec_unionfind": ROOT / "scripts/golf/scratch_codex/task286/cand_unionfind_truthful.onnx",
    "task286_spec_literal624": ROOT / "scripts/golf/scratch_codex/task286/cand_cloaked_gather_flood_v7_parity.onnx",
}


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
        "params": sum(int(numpy_helper.to_array(x).size) for x in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "ops": dict(collections.Counter(x.op_type for x in model.graph.node)),
    }


def main() -> None:
    base_sha = sha(BASE_ZIP.read_bytes())
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"wrong base archive: {base_sha}")
    for name in ("baseline", "candidates", "sound"):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    models = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(archive.read(f"task{task:03d}.onnx"))
            models.append(summary(path, task, f"task{task:03d}_base"))
    inventory = json.loads(INVENTORY.read_text())
    selected = {}
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
    sound = {}
    for label, source in SOUND.items():
        task = int(label[4:7])
        path = HERE / "sound" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        models.append(summary(path, task, label))
        sound[label] = {
            "task": task,
            "source": str(source.relative_to(ROOT)),
            "sha256": sha(path.read_bytes()),
        }
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "base_archive": str(BASE_ZIP.relative_to(ROOT)),
                "base_archive_sha256": base_sha,
                "models": models,
                "history_entries": selected,
                "sound_entries": sound,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare exact A22 baselines, retained history, and sound rebuilds."""

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
TASKS = (101, 158)
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
EXPECTED_TASK_SHA256 = {
    101: "4f0de084b752a247766872e8dba29416f9198021d53e1a834a690d722a5a65b9",
    158: "24eb1f1d238ea5087b1372d3fb322865c181026fb4997379cc58fab35c39381b",
}
SOUND = {
    "task101_spec_greedy": ROOT / "scripts/golf/scratch_codex_7994/task101_sound/sound_7264.onnx",
    "task101_spec_topk": ROOT / "scripts/golf/scratch_codex/task101/topk_parity_rebuild.onnx",
    "task158_spec_anchor13": ROOT / "scripts/golf/scratch_codex/task158/incumbent_agent_anchor13.onnx",
    "task158_spec_exact_tail": ROOT / "scripts/golf/scratch_codex/task158/worker_safe_exact_tail.onnx",
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
            data = archive.read(f"task{task:03d}.onnx")
            if sha(data) != EXPECTED_TASK_SHA256[task]:
                raise RuntimeError(f"wrong exact task{task:03d} lineage: {sha(data)}")
            path.write_bytes(data)
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
                "expected_task_sha256": EXPECTED_TASK_SHA256,
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

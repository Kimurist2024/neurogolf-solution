#!/usr/bin/env python3
"""Prepare exact A23 baselines and the complete lower-cost/sound frontier."""

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
C3_SCAN = ROOT / "scripts/golf/loop_7999_13/lane_c3/scan_results.json"
TASKS = (44, 205)
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
EXPECTED_TASK_SHA256 = {
    44: "d4cd7b0883b1ef523a676e438fa8e0af3d90b69e1fef5e0831e92b791abe54d4",
    205: "8a6acdc20a366ccbd32cf761285cbb2f1cbcf7d3d2ef8ea71d0fb5a3ed6f1468",
}
EXTRA_HISTORY = {
    "task205_c3_cost1038": ROOT / "scripts/golf/loop_7999_13/lane_c3/candidates/task205.onnx",
}
RULE_REFERENCES = {
    "task044_rule_autocorr": ROOT / "scripts/golf/scratch_codex/task044/cand_autocorr_groundup.onnx",
    "task205_known_97p8": ROOT / "scripts/golf/scratch_claude/task205/incumbent_backup.onnx",
    "task205_compact_d16b": ROOT / "scripts/golf/scratch_codex/task205/candidate_groundup_d16b.onnx",
    "task205_exact_border_scan": ROOT / "scripts/golf/scratch_codex/task205/candidate_border_scan_v2.onnx",
    "task205_exact_single_scan": ROOT / "scripts/golf/scratch_claude/task205/candidate_single_scan.onnx",
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
    for name in ("baseline", "candidates", "rule_references"):
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
    history = {}
    for index, item in enumerate(inventory["retained"]["205"], 1):
        label = f"task205_r{index:02d}"
        source = ROOT / item["path"]
        path = HERE / "candidates" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        if sha(path.read_bytes()) != item["sha256"]:
            raise RuntimeError(f"hash mismatch: {label}")
        models.append(summary(path, 205, label))
        history[label] = item
    for label, source in EXTRA_HISTORY.items():
        path = HERE / "candidates" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        models.append(summary(path, 205, label))
        history[label] = {
            "task": 205,
            "static_cost": 1038,
            "sha256": sha(path.read_bytes()),
            "path": str(source.relative_to(ROOT)),
            "sources": [
                "others/2/7901/task205_cost1038.onnx",
                "others/2/7902/task205_cost1038.onnx",
            ],
        }

    references = {}
    for label, source in RULE_REFERENCES.items():
        task = int(label[4:7])
        path = HERE / "rule_references" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        models.append(summary(path, task, label))
        references[label] = {
            "task": task,
            "source": str(source.relative_to(ROOT)),
            "sha256": sha(path.read_bytes()),
        }

    c3 = json.loads(C3_SCAN.read_text())
    rows = [row for row in c3["rows"] if int(row["task"]) in TASKS]
    coverage = {
        str(task): {
            "unique_sha": sum(int(row["task"]) == task for row in rows),
            "status_counts": dict(collections.Counter(row["status"] for row in rows if int(row["task"]) == task)),
            "scored_below_base": [row for row in rows if int(row["task"]) == task and row["status"] == "scored"],
        }
        for task in TASKS
    }
    (HERE / "full_history_inventory.json").write_text(
        json.dumps(
            {
                "source": str(C3_SCAN.relative_to(ROOT)),
                "source_base_sha256": c3["base_sha256"],
                "tasks": list(TASKS),
                "coverage": coverage,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "base_archive": str(BASE_ZIP.relative_to(ROOT)),
                "base_archive_sha256": base_sha,
                "expected_task_sha256": EXPECTED_TASK_SHA256,
                "models": models,
                "history_entries": history,
                "rule_references": references,
                "full_history_coverage": coverage,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

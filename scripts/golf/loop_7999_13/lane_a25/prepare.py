#!/usr/bin/env python3
"""Prepare exact task117/task160 A25 baselines and complete relevant history."""

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
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json"
C4_SCAN = ROOT / "scripts/golf/loop_7999_13/lane_c4/scan_results.json"
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
EXPECTED_TASK_SHA256 = {
    117: "e8dee03b3c5f0dc39fe1333b9b6cd53c4903aa2015baa5db15ef4a7897ac6073",
    160: "6300f4550400fc63391ee490cbb8635f468e571296dc84085c24e7aba85b8548",
}
RULE_REFERENCES = {
    "task117_truthful_copy_hist": ROOT / "scripts/golf/scratch_codex/task117/task117_copy_hist_pruned.onnx",
    "task160_truthful_rule_v1": ROOT / "scripts/golf/scratch_codex/task160/task160_v1.onnx",
}
EXTRA_HISTORY = {
    "task117_col_once_clean": ROOT / "scripts/golf/scratch_codex/task117/task117_col_once_clean.onnx",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
        for dim in value.type.tensor_type.shape.dim
    ]


def summary(path: Path, task: int, label: str) -> dict[str, object]:
    model = onnx.load(path, load_external_data=False)
    ops = collections.Counter(node.op_type for node in model.graph.node)
    return {
        "task": task,
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path.read_bytes()),
        "bytes": path.stat().st_size,
        "nodes": len(model.graph.node),
        "params": sum(int(numpy_helper.to_array(item).size) for item in model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "ops": dict(ops),
        "max_einsum_inputs": max(
            (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
            default=0,
        ),
        "declared_inputs": {item.name: shape(item) for item in model.graph.input},
        "declared_outputs": {item.name: shape(item) for item in model.graph.output},
    }


def main() -> None:
    base_sha = sha(BASE_ZIP.read_bytes())
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"wrong base archive: {base_sha}")
    for name in ("baseline", "history", "rule_references"):
        (HERE / name).mkdir(parents=True, exist_ok=True)

    models: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in (117, 160):
            data = archive.read(f"task{task:03d}.onnx")
            if sha(data) != EXPECTED_TASK_SHA256[task]:
                raise RuntimeError(f"wrong task{task:03d} lineage: {sha(data)}")
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(data)
            models.append(summary(path, task, f"task{task:03d}_base"))

    inventory = json.loads(INVENTORY.read_text())
    history: dict[str, dict[str, object]] = {}
    for index, entry in enumerate(inventory["retained"]["160"], 1):
        label = f"task160_r{index:02d}"
        source = ROOT / entry["path"]
        path = HERE / "history" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        if sha(path.read_bytes()) != entry["sha256"]:
            raise RuntimeError(f"history hash mismatch: {label}")
        history[label] = entry
        models.append(summary(path, 160, label))

    for label, source in EXTRA_HISTORY.items():
        task = int(label[4:7])
        path = HERE / "history" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        history[label] = {
            "task": task,
            "static_cost": 278,
            "sha256": sha(path.read_bytes()),
            "path": str(source.relative_to(ROOT)),
            "sources": [str(source.relative_to(ROOT))],
            "source_count": 1,
            "prior_status": "unscorable",
        }
        models.append(summary(path, task, label))

    references: dict[str, dict[str, object]] = {}
    for label, source in RULE_REFERENCES.items():
        task = int(label[4:7])
        path = HERE / "rule_references" / f"{label}.onnx"
        path.write_bytes(source.read_bytes())
        references[label] = {
            "task": task,
            "source": str(source.relative_to(ROOT)),
            "sha256": sha(path.read_bytes()),
            "provenance": "generator-rule rebuild",
        }
        models.append(summary(path, task, label))

    c4 = json.loads(C4_SCAN.read_text())
    c4_rows = [row for row in c4["rows"] if int(row.get("task", -1)) == 117]
    coverage = {
        "task117": {
            "unique_models": len(c4_rows),
            "status_counts": dict(collections.Counter(row["status"] for row in c4_rows)),
            "below_base_static": [
                row for row in c4_rows if row.get("static_cost_floor", 10**9) < 606
            ],
        },
        "task160": {
            "retained_below_base": len(inventory["retained"]["160"]),
            "entries": inventory["retained"]["160"],
        },
    }
    (HERE / "full_history_inventory.json").write_text(
        json.dumps(
            {
                "task117_source": str(C4_SCAN.relative_to(ROOT)),
                "task160_source": str(INVENTORY.relative_to(ROOT)),
                "coverage": coverage,
                "task117_rows": c4_rows,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "lane": "a25",
                "base_score": 7999.13,
                "base_archive": str(BASE_ZIP.relative_to(ROOT)),
                "base_archive_sha256": base_sha,
                "expected_task_sha256": EXPECTED_TASK_SHA256,
                "models": models,
                "history_entries": history,
                "rule_references": references,
                "history_coverage": coverage,
                "generator_hashes": {"117": "4c5c2cf0", "160": "6c434453"},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

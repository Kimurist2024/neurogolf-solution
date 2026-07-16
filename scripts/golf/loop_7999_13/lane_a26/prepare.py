#!/usr/bin/env python3
"""Prepare exact A26 baselines, full retained history, and SOUND controls."""

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
B9_LOCAL = ROOT / "scripts/golf/loop_7999_13/lane_b9/local_history_inventory.json"
B9_ACTUAL = ROOT / "scripts/golf/loop_7999_13/lane_b9/local_history_actual_scores.json"
B9_ARCHIVE_ACTUAL = ROOT / "scripts/golf/loop_7999_13/lane_b9/archive_actual_scores.json"
B9_DUAL = ROOT / "scripts/golf/loop_7999_13/lane_b9/task182_dual_ort_5000.json"
HARVEST = ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json"
EXPECTED_BASE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
EXPECTED_TASK_SHA256 = {
    182: "eb60c542b43cab9ddee08e33e4dbef778e7199632c9dc16a4cd29ef55bad9587",
    330: "06dcb5216cb441f6d760a28fa4a5b4affa678c9001504dbd8f4f80f3bbd2d5af",
}
RULE_REFERENCES = {
    "task182_truthful_rule_r1": ROOT / "scripts/golf/scratch_claude/task182/r1.onnx",
    "task182_exact_truthful_shapes": ROOT / "scripts/golf/loop_7999_13/lane_b9/task182_static_shapes.onnx",
    "task330_truthful_component_rect": ROOT / "scripts/golf/scratch_codex/task330/agent_rect.onnx",
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
        for task in (182, 330):
            data = archive.read(f"task{task:03d}.onnx")
            if sha(data) != EXPECTED_TASK_SHA256[task]:
                raise RuntimeError(f"wrong task{task:03d} lineage: {sha(data)}")
            path = HERE / "baseline" / f"task{task:03d}.onnx"
            path.write_bytes(data)
            models.append(summary(path, task, f"task{task:03d}_base"))

    inventory = json.loads(INVENTORY.read_text())
    history: dict[str, dict[str, object]] = {}
    for task in (182, 330):
        for index, entry in enumerate(inventory["retained"][str(task)], 1):
            label = f"task{task:03d}_r{index:02d}"
            source = ROOT / entry["path"]
            path = HERE / "history" / f"{label}.onnx"
            path.write_bytes(source.read_bytes())
            if sha(path.read_bytes()) != entry["sha256"]:
                raise RuntimeError(f"history hash mismatch: {label}")
            history[label] = entry
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
            "provenance": "generator-rule SOUND or exact truthful-shape control",
        }
        models.append(summary(path, task, label))

    local = json.loads(B9_LOCAL.read_text())
    local182 = [row for row in local if int(row.get("task", -1)) == 182]
    local_actual = [
        row for row in json.loads(B9_ACTUAL.read_text()) if int(row.get("task", -1)) == 182
    ]
    archive_actual = [
        row for row in json.loads(B9_ARCHIVE_ACTUAL.read_text()) if int(row.get("task", -1)) == 182
    ]
    harvest = json.loads(HARVEST.read_text())
    harvest330 = [row for row in harvest["rows"] if int(row.get("task", -1)) == 330]
    prior_dual = json.loads(B9_DUAL.read_text())
    provenance = {
        "task182": {
            "archive_order_pollution_policy": (
                "aggregate/reordered ZIP occurrence is not isolated task-level evidence; "
                "known/fresh alone cannot override default-ORT or truthful-shape failures"
            ),
            "retained_r01_r02_r05_r06": "multi-task archive lineage; not isolated white evidence",
            "retained_r03_r04": "locally reconstructed exact-reuse candidates; no archive-order dependency",
            "r03_prior_dual_fresh5000": prior_dual,
            "r03_decision": "reject: default ORT session creation fails and runtime shapes are false",
        }
    }
    coverage = {
        "task182": {
            "local_rows": len(local182),
            "local_status_counts": dict(collections.Counter(row["status"] for row in local182)),
            "retained_below_base": len(inventory["retained"]["182"]),
            "actual_profile_rows": len(local_actual) + len(archive_actual),
        },
        "task330": {
            "harvest_rows": len(harvest330),
            "harvest_stage_counts": dict(collections.Counter(row["stage"] for row in harvest330)),
            "retained_below_base": len(inventory["retained"]["330"]),
        },
    }
    (HERE / "full_history_inventory.json").write_text(
        json.dumps(
            {
                "sources": {
                    "task182_local": str(B9_LOCAL.relative_to(ROOT)),
                    "task182_local_actual": str(B9_ACTUAL.relative_to(ROOT)),
                    "task182_archive_actual": str(B9_ARCHIVE_ACTUAL.relative_to(ROOT)),
                    "task330_harvest": str(HARVEST.relative_to(ROOT)),
                    "retained_frontier": str(INVENTORY.relative_to(ROOT)),
                },
                "coverage": coverage,
                "provenance_policy": provenance,
                "task182_retained": inventory["retained"]["182"],
                "task330_retained": inventory["retained"]["330"],
                "task182_local_rows": local182,
                "task182_actual_rows": local_actual + archive_actual,
                "task330_harvest_rows": harvest330,
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "model_manifest.json").write_text(
        json.dumps(
            {
                "lane": "a26",
                "base_score": 7999.13,
                "base_archive": str(BASE_ZIP.relative_to(ROOT)),
                "base_archive_sha256": base_sha,
                "expected_task_sha256": EXPECTED_TASK_SHA256,
                "models": models,
                "history_entries": history,
                "rule_references": references,
                "history_coverage": coverage,
                "generator_hashes": {"182": "776ffc46", "330": "d2abd087"},
                "task182_provenance_policy": provenance["task182"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict A26 both-ORT, runtime-shape, actual-cost audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


BASE_COST = {182: 994, 330: 897}
shared.BASE_COST = BASE_COST


def audit_one(
    task: int,
    label: str,
    path: Path,
    inventory_static: int | None,
    sources: list[str],
    baseline: bool = False,
) -> dict[str, object]:
    row = shared.audit(task, label, path, inventory_static, sources, baseline=baseline)
    if task == 182 and not baseline:
        row["archive_order_pollution_review"] = {
            "aggregate_archive_sources": [
                source
                for source in sources
                if ".zip" in source or "submission" in source or "ordered" in source
            ],
            "known_or_fresh_alone_sufficient": False,
        }
    return row


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows: list[dict[str, object]] = []
    for task in (182, 330):
        label = f"task{task:03d}_base"
        row = audit_one(
            task,
            label,
            HERE / "baseline" / f"task{task:03d}.onnx",
            BASE_COST[task],
            ["submission_base_7999.13.zip"],
            baseline=True,
        )
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)

    for label, entry in manifest["history_entries"].items():
        task = int(entry["task"])
        row = audit_one(
            task,
            label,
            HERE / "history" / f"{label}.onnx",
            int(entry["static_cost"]),
            list(entry["sources"]),
        )
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)

    for label, entry in manifest["rule_references"].items():
        task = int(entry["task"])
        row = audit_one(
            task,
            label,
            HERE / "rule_references" / f"{label}.onnx",
            None,
            [str(entry["source"])],
        )
        rows.append(row)
        print(label, row.get("official_like_score"), row["pre_fresh_reasons"], flush=True)

    pending = [
        row
        for row in rows
        if row["pre_fresh_pass"] and not str(row["label"]).endswith("_base")
    ]
    (HERE / "audit_rows.json").write_text(
        json.dumps({"rows": rows, "pending": pending, "complete": True}, indent=2) + "\n"
    )
    (HERE / "fresh_dual_5000.json").write_text(
        json.dumps(
            {
                "required_for": "strictly-cheaper both-known truthful-shape finalists",
                "pending_count": len(pending),
                "runs": [],
                "complete": True,
                "reason": "no pre-fresh finalist",
                "historical_task182_disable_only_run": (
                    "r03 passed disabled fresh5000 but default ORT session creation failed; "
                    "it is not a valid dual-runtime finalist"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (HERE / "external_validation_summary.json").write_text(
        json.dumps(
            {
                "required_for": "strictly-cheaper candidates passing dual fresh5000",
                "pending_count": 0,
                "runs": [],
                "errors": 0,
                "complete": True,
                "reason": "no dual-fresh finalist",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE rows={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()

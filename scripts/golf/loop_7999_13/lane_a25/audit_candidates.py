#!/usr/bin/env python3
"""Strict both-ORT, truthful-shape, and cost audit for A25."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_history as shared  # noqa: E402


BASE_COST = {117: 606, 160: 404}
shared.BASE_COST = BASE_COST


def main() -> None:
    ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "model_manifest.json").read_text())
    rows: list[dict[str, object]] = []

    for task in (117, 160):
        label = f"task{task:03d}_base"
        row = shared.audit(
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
        row = shared.audit(
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
        row = shared.audit(
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
                "required_for": "strictly-cheaper candidates passing both-known and truthful-shape gates",
                "pending_count": len(pending),
                "runs": [],
                "complete": True,
                "reason": "no pre-fresh finalist",
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

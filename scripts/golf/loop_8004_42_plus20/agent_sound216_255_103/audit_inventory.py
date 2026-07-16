#!/usr/bin/env python3
"""Freeze the 8008.14 task216/task255 members and summarize prior candidates.

This lane is deliberately read-only outside its own directory.  The large
historical rescreen already deduplicated archive members by SHA-256, so reuse
that inventory rather than walking and reopening thousands of submission ZIPs.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8008.14.zip"
RESCREEN = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_expand20h_92/rescreen.json"
)
TASKS = (216, 255)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_summary(data: bytes) -> dict[str, object]:
    model = onnx.load_model_from_string(data)
    return {
        "sha256": sha256(data),
        "serialized_bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "value_info": len(model.graph.value_info),
        "opsets": [
            {"domain": opset.domain, "version": opset.version}
            for opset in model.opset_import
        ],
        "op_histogram": dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
    }


def main() -> int:
    historical = json.loads(RESCREEN.read_text())
    rows = historical["rows"]
    output: dict[str, object] = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": sha256(AUTHORITY.read_bytes()),
        },
        "historical_inventory_source": str(RESCREEN.relative_to(ROOT)),
        "tasks": {},
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            (HERE / member).write_bytes(data)
            task_rows = [row for row in rows if row["task"] == task]
            current_cost = task_rows[0]["current_actual_cost"]
            potentially_cheaper = [
                {
                    "sha256": row["sha256"],
                    "static_floor": row.get("static_floor"),
                    "actual_screen_cost": row.get("actual_screen_cost"),
                    "stage": row["stage"],
                    "reasons": row["reasons"],
                    "source_count": len(row["sources"]),
                    "sample_sources": row["sources"][:5],
                }
                for row in task_rows
                if row.get("static_floor") is not None
                and row["static_floor"] < current_cost
            ]
            output["tasks"][str(task)] = {
                "authority_member": member,
                "authority_model": model_summary(data),
                "reported_current_actual_cost": current_cost,
                "unique_historical_sha_count": len(task_rows),
                "stage_histogram": dict(Counter(row["stage"] for row in task_rows)),
                "historical_static_below_current_count": len(potentially_cheaper),
                "historical_static_below_current": potentially_cheaper,
            }
    (HERE / "inventory.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

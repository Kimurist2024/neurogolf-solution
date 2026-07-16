#!/usr/bin/env python3
"""Fast fail-closed audit of retained archive leads for a task group."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8005.16.zip"

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(data: bytes, name: str) -> dict[str, object]:
    try:
        with tempfile.TemporaryDirectory(prefix="retained_cost_", dir="/tmp") as temp:
            path = Path(temp) / name
            path.write_bytes(data)
            memory, params, cost = cost_of(str(path))
        return {"memory": memory, "params": params, "cost": cost, "error": None}
    except Exception as exc:
        return {
            "memory": None,
            "params": None,
            "cost": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def perfect(row: dict[str, object]) -> bool:
    return bool(row.get("total")) and row.get("right") == row.get("total") and row.get("errors") == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="comma-separated task numbers")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    tasks = tuple(int(value) for value in args.tasks.split(","))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    baselines = []
    leads = []
    with zipfile.ZipFile(BASE) as archive:
        for task in tasks:
            member = f"task{task:03d}.onnx"
            baseline_data = archive.read(member)
            baseline_cost = profile(baseline_data, member)
            baselines.append(
                {
                    "task": task,
                    "sha256": sha(baseline_data),
                    "actual_cost": baseline_cost,
                    "retained_count": len(inventory.get("retained", {}).get(str(task), [])),
                }
            )
            print(f"base task{task:03d} cost={baseline_cost['cost']}", flush=True)

            for record in inventory.get("retained", {}).get(str(task), []):
                path = ROOT / record["path"]
                data = path.read_bytes()
                candidate = onnx.load_model_from_string(data)
                actual = profile(data, path.name)
                disable = run_known(copy.deepcopy(candidate), task, True)
                default: dict[str, object]
                if perfect(disable):
                    default = run_known(copy.deepcopy(candidate), task, False)
                else:
                    default = {"not_run": "disable_all_not_perfect"}
                cheaper = (
                    actual.get("cost") is not None
                    and baseline_cost.get("cost") is not None
                    and int(actual["cost"]) < int(baseline_cost["cost"])
                )
                structurally_audited = cheaper and perfect(disable) and perfect(default)
                row = {
                    "task": task,
                    "path": record["path"],
                    "sha256": sha(data),
                    "reported_static_cost": record.get("static_cost"),
                    "baseline_actual_cost": baseline_cost.get("cost"),
                    "actual_cost": actual,
                    "strictly_cheaper": cheaper,
                    "known_disable_all": disable,
                    "known_default": default,
                    "structure": structure(copy.deepcopy(candidate), task) if structurally_audited else None,
                    "promising_pre_fresh": structurally_audited,
                }
                leads.append(row)
                print(
                    f"lead task{task:03d} static={record.get('static_cost')} "
                    f"actual={actual.get('cost')} known={disable.get('right')}/{disable.get('total')} "
                    f"promising={structurally_audited}",
                    flush=True,
                )

    result = {
        "baseline": {"path": BASE.name, "sha256": sha(BASE.read_bytes())},
        "targets": list(tasks),
        "baseline_rows": baselines,
        "lead_rows": leads,
        "promising": [
            {"task": row["task"], "path": row["path"], "sha256": row["sha256"]}
            for row in leads
            if row["promising_pre_fresh"]
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

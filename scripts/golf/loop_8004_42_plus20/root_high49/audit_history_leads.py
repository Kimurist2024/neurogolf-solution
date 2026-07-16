#!/usr/bin/env python3
"""Pre-screen high-cost unreported archive leads against the 8005.16 base."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
TARGETS = (37, 297, 14, 92, 398, 218, 132, 388)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def actual_cost(data: bytes, name: str) -> dict[str, object]:
    try:
        with tempfile.TemporaryDirectory(prefix="high49_cost_", dir="/tmp") as temp:
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


def main() -> None:
    inventory = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    baseline_rows = []
    lead_rows = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            model = onnx.load_model_from_string(data)
            cost = actual_cost(data, member)
            baseline_rows.append(
                {
                    "task": task,
                    "sha256": sha(data),
                    "actual_cost": cost,
                    "structure": structure(copy.deepcopy(model), task),
                }
            )
            print(f"base task{task:03d} cost={cost['cost']}", flush=True)

            retained = inventory.get("retained", {}).get(str(task), [])
            for index, record in enumerate(retained):
                path = ROOT / record["path"]
                candidate_data = path.read_bytes()
                candidate = onnx.load_model_from_string(candidate_data)
                measured = actual_cost(candidate_data, path.name)
                row = {
                    "task": task,
                    "path": record["path"],
                    "sha256": sha(candidate_data),
                    "reported_static_cost": record.get("static_cost"),
                    "baseline_actual_cost": cost.get("cost"),
                    "actual_cost": measured,
                    "structure": structure(copy.deepcopy(candidate), task),
                    "known_dual": {
                        "disable_all": run_known(copy.deepcopy(candidate), task, True),
                        "default": run_known(copy.deepcopy(candidate), task, False),
                    },
                }
                lead_rows.append(row)
                kd = row["known_dual"]["disable_all"]
                print(
                    f"lead task{task:03d} r{index + 1} cost={measured['cost']} "
                    f"known={kd['right']}/{kd['total']} err={kd['errors']}",
                    flush=True,
                )

    result = {
        "baseline": {"path": BASE.name, "sha256": sha(BASE.read_bytes())},
        "targets": list(TARGETS),
        "baseline_rows": baseline_rows,
        "lead_rows": lead_rows,
    }
    (HERE / "history_lead_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

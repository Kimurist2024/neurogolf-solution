#!/usr/bin/env python3
"""Read-only latest-baseline audit for the low37 eight-target lane."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8005.16.zip"
PREVIOUS = ROOT / "submission_base_8004.50.zip"
TARGETS = (320, 154, 393, 290, 336, 3, 58, 72)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.agent_new_low35.audit_latest import (  # noqa: E402
    structural,
)
from scripts.golf.rank_dir import cost_of  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    exact_scan = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text()
    )
    exact_task_hits = []
    for section in ("baseline_structural_failures", "candidates"):
        for row in exact_scan.get(section, []):
            if row.get("task") in TARGETS:
                exact_task_hits.append({"section": section, **row})

    rows = []
    with zipfile.ZipFile(BASE) as current, zipfile.ZipFile(PREVIOUS) as previous:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            data = current.read(member)
            old = previous.read(member)
            model = onnx.load_model_from_string(data)
            (HERE / "baselines" / member).write_bytes(data)
            with tempfile.TemporaryDirectory(prefix=f"low37_{task}_", dir="/tmp") as temp:
                path = Path(temp) / member
                path.write_bytes(data)
                memory, params, cost = cost_of(str(path))
            rows.append(
                {
                    "task": task,
                    "member": member,
                    "sha256": sha(data),
                    "file_bytes": len(data),
                    "unchanged_from_8004_50": data == old,
                    "previous_member_sha256": sha(old),
                    "actual_cost": {"memory": memory, "params": params, "cost": cost},
                    "structure": structural(model, task),
                }
            )
            print(f"task{task:03d}: cost={cost} unchanged={data == old}", flush=True)

    evidence = {
        "baseline": {
            "path": BASE.name,
            "sha256": sha(BASE.read_bytes()),
            "previous_path": PREVIOUS.name,
            "previous_sha256": sha(PREVIOUS.read_bytes()),
        },
        "targets": rows,
        "exact_wave2": {
            "baseline": exact_scan["baseline"],
            "tasks_scanned": exact_scan["tasks_scanned"],
            "accepted": exact_scan["summary"].get("accepted", 0),
            "target_hits": exact_task_hits,
        },
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict actual-cost and known screen for A15 retained models."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
from harvest import structure_gate  # noqa: E402
from lib import scoring  # noqa: E402


TASKS = (170, 234, 239, 333, 338, 374, 377)
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json"
WORK = HERE / "score_work"
OUT = HERE / "retained_scan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path, task: int, label: str) -> dict[str, object] | None:
    return scoring.score_and_verify(copy.deepcopy(onnx.load(path)), task, str(WORK), label=label, require_correct=False)


def main() -> None:
    ort.set_default_logger_severity(4)
    inventory = json.loads(INVENTORY.read_text())
    baselines: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        _, reason, floor = structure_gate(path.read_bytes())
        result = profile(path, task, f"a15_base_{task:03d}")
        if result is None:
            raise RuntimeError(f"unscorable baseline task{task:03d}")
        baselines[task] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "structure_gate": reason, "static_floor": floor, "profile": result}
        print(f"BASE task{task:03d} cost={result['cost']} correct={result['correct']} structure={reason}", flush=True)
    candidates = sum((inventory["retained"].get(str(task), []) for task in TASKS), [])
    for index, item in enumerate(candidates, 1):
        task = int(item["task"])
        path = ROOT / item["path"]
        model, reason, floor = structure_gate(path.read_bytes())
        row: dict[str, object] = {
            "task": task, "path": item["path"], "sha256": sha(path), "sources": item["sources"],
            "source_count": item["source_count"], "inventory_static_cost": item["static_cost"],
            "recomputed_static_floor": floor, "structure_gate": reason,
            "baseline_cost": baselines[task]["profile"]["cost"],
        }
        if model is None or reason != "pass":
            row.update(profile=None, complete_known_pass=False, strictly_cheaper=False, verdict="reject_structure")
        else:
            result = scoring.score_and_verify(copy.deepcopy(model), task, str(WORK), label=f"a15_{task:03d}_{index:02d}", require_correct=False)
            row["profile"] = result
            row["complete_known_pass"] = bool(result and result["correct"])
            row["strictly_cheaper"] = bool(result and result["cost"] < row["baseline_cost"])
            if result is None:
                row["verdict"] = "reject_unscorable"
            elif not result["correct"]:
                row["verdict"] = "reject_known_mismatch"
            elif result["cost"] >= row["baseline_cost"]:
                row["verdict"] = "reject_not_cheaper"
            else:
                row["verdict"] = "promote_pending_diff_and_fresh"
        rows.append(row)
        detail = "none" if row["profile"] is None else f"cost={row['profile']['cost']} correct={row['profile']['correct']}"
        print(f"CAND {index:02d}/{len(candidates)} task{task:03d} {detail} {row['verdict']} [{reason}]", flush=True)
        OUT.write_text(json.dumps({"baselines": baselines, "rows": rows, "complete": False}, indent=2) + "\n")
    pending = [row for row in rows if row["verdict"] == "promote_pending_diff_and_fresh"]
    OUT.write_text(json.dumps({"baseline_zip": "submission_base_7999.13.zip", "baselines": baselines, "rows": rows, "pending": pending, "complete": True}, indent=2) + "\n")
    print(f"DONE candidates={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()

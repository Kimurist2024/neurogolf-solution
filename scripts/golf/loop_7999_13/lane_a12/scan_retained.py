#!/usr/bin/env python3
"""Strict structural and complete-known profile screen for A12 candidates."""

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


TASKS = (198, 200, 201, 219, 302, 343)
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json"
WORK = HERE / "score_work"
OUT = HERE / "retained_scan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(path: Path, task: int, label: str) -> dict[str, object] | None:
    return scoring.score_and_verify(
        copy.deepcopy(onnx.load(path)), task, str(WORK), label=label, require_correct=False
    )


def main() -> None:
    ort.set_default_logger_severity(4)
    inventory = json.loads(INVENTORY.read_text())
    baselines: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for task in TASKS:
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        _, reason, floor = structure_gate(path.read_bytes())
        profile = score(path, task, f"a12_base_{task:03d}")
        if profile is None:
            raise RuntimeError(f"unscorable exact baseline task{task:03d}")
        baselines[task] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "structure_gate": reason,
            "static_floor": floor,
            "profile": profile,
        }
        print(f"BASE task{task:03d} cost={profile['cost']} correct={profile['correct']} structure={reason}", flush=True)

    candidates = sum((inventory["retained"].get(str(task), []) for task in TASKS), [])
    for index, entry in enumerate(candidates, 1):
        task = int(entry["task"])
        path = ROOT / entry["path"]
        data = path.read_bytes()
        model, reason, floor = structure_gate(data)
        row: dict[str, object] = {
            "task": task,
            "path": entry["path"],
            "sha256": sha(path),
            "inventory_sha256": entry["sha256"],
            "sources": entry["sources"],
            "source_count": entry["source_count"],
            "inventory_static_cost": entry["static_cost"],
            "recomputed_static_floor": floor,
            "structure_gate": reason,
            "baseline_cost": baselines[task]["profile"]["cost"],
        }
        if model is None or reason != "pass":
            row.update(
                profile=None,
                strictly_cheaper=False,
                complete_known_pass=False,
                verdict="reject_structure",
            )
        else:
            profile = scoring.score_and_verify(
                copy.deepcopy(model), task, str(WORK),
                label=f"a12_{task:03d}_{index:02d}", require_correct=False,
            )
            row["profile"] = profile
            row["strictly_cheaper"] = bool(profile and profile["cost"] < row["baseline_cost"])
            row["complete_known_pass"] = bool(profile and profile["correct"])
            if profile is None:
                row["verdict"] = "reject_unscorable"
            elif not profile["correct"]:
                row["verdict"] = "reject_known_mismatch"
            elif profile["cost"] >= row["baseline_cost"]:
                row["verdict"] = "reject_not_cheaper"
            else:
                row["verdict"] = "promote_pending_cloak_and_fresh"
        rows.append(row)
        detail = "none" if row["profile"] is None else f"cost={row['profile']['cost']} correct={row['profile']['correct']}"
        print(f"CAND {index:02d}/{len(candidates)} task{task:03d} {detail} {row['verdict']} [{reason}]", flush=True)
        OUT.write_text(json.dumps({"baselines": baselines, "rows": rows, "complete": False}, indent=2) + "\n")

    pending = [row for row in rows if row["verdict"] == "promote_pending_cloak_and_fresh"]
    OUT.write_text(
        json.dumps(
            {
                "baseline_zip": "submission_base_7999.13.zip",
                "baselines": baselines,
                "rows": rows,
                "pending": pending,
                "complete": True,
            }, indent=2,
        ) + "\n"
    )
    print(f"DONE candidates={len(rows)} pending={len(pending)}", flush=True)


if __name__ == "__main__":
    main()

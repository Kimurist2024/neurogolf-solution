#!/usr/bin/env python3
"""Profile and complete-known screen retained archive candidates for A11.

All artifacts remain in lane_a11.  Static estimates are never used as the
score authority: every structurally valid retained candidate is run through
the same complete-corpus ORT_DISABLE_ALL scorer as the exact 7999.13 member.
"""

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


TASKS = (65, 88, 189, 224, 240)
INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/inventory.json"
BASELINE_DIR = HERE / "baseline"
WORK = HERE / "score_work_retained"
OUT = HERE / "retained_scan.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path: Path, task: int, label: str) -> dict[str, object] | None:
    model = onnx.load(path)
    return scoring.score_and_verify(
        copy.deepcopy(model), task, str(WORK), label=label, require_correct=False
    )


def main() -> None:
    ort.set_default_logger_severity(4)
    inventory = json.loads(INVENTORY.read_text())
    baselines: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for task in TASKS:
        base_path = BASELINE_DIR / f"task{task:03d}.onnx"
        base_data = base_path.read_bytes()
        _, reason, static_floor = structure_gate(base_data)
        result = profile(base_path, task, f"a11_base_{task:03d}")
        if result is None:
            raise RuntimeError(f"baseline task{task:03d} unscorable")
        baselines[task] = {
            "path": str(base_path.relative_to(ROOT)),
            "sha256": digest(base_path),
            "static_floor": static_floor,
            "structure_gate": reason,
            "profile": result,
        }
        print(f"BASE task{task:03d} cost={result['cost']} correct={result['correct']}", flush=True)

    candidates = sum((inventory["retained"].get(str(task), []) for task in TASKS), [])
    for index, entry in enumerate(candidates, 1):
        task = int(entry["task"])
        path = ROOT / entry["path"]
        data = path.read_bytes()
        model, reason, recomputed_floor = structure_gate(data)
        row: dict[str, object] = {
            "task": task,
            "path": entry["path"],
            "sha256": digest(path),
            "inventory_sha256": entry["sha256"],
            "sources": entry["sources"],
            "source_count": entry["source_count"],
            "inventory_static_cost": entry["static_cost"],
            "recomputed_static_floor": recomputed_floor,
            "structure_gate": reason,
            "baseline_cost": baselines[task]["profile"]["cost"],
        }
        if model is not None and reason == "pass":
            result = scoring.score_and_verify(
                copy.deepcopy(model), task, str(WORK),
                label=f"a11_ret_{task:03d}_{index:02d}", require_correct=False,
            )
            row["profile"] = result
            row["strictly_cheaper"] = bool(
                result is not None and result["cost"] < row["baseline_cost"]
            )
            row["complete_known_pass"] = bool(result is not None and result["correct"])
            if result is None:
                row["verdict"] = "reject_unscorable"
            elif not result["correct"]:
                row["verdict"] = "reject_known_mismatch"
            elif result["cost"] >= row["baseline_cost"]:
                row["verdict"] = "reject_not_cheaper"
            else:
                row["verdict"] = "promote_to_fresh"
        else:
            row["profile"] = None
            row["strictly_cheaper"] = False
            row["complete_known_pass"] = False
            row["verdict"] = "reject_structure"
        rows.append(row)
        p = row.get("profile")
        detail = "none" if p is None else f"cost={p['cost']} correct={p['correct']}"
        print(
            f"CAND {index:02d}/{len(candidates)} task{task:03d} {detail} {row['verdict']}",
            flush=True,
        )
        OUT.write_text(
            json.dumps({"baselines": baselines, "rows": rows, "complete": False}, indent=2) + "\n"
        )

    promoted = [row for row in rows if row["verdict"] == "promote_to_fresh"]
    OUT.write_text(
        json.dumps(
            {
                "baseline_zip": "submission_base_7999.13.zip",
                "baselines": baselines,
                "rows": rows,
                "promoted": promoted,
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE candidates={len(rows)} promoted={len(promoted)}", flush=True)


if __name__ == "__main__":
    main()

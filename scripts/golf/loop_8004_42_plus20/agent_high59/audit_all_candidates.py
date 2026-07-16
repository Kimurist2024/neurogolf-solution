#!/usr/bin/env python3
"""Audit every locally retained ONNX artifact for the assigned eight tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import zipfile
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "submission_base_8005.16.zip"
OUT = Path(__file__).with_name("all_candidate_audit.json")
TARGETS = (151, 213, 122, 94, 220, 260, 342, 331)

import sys

sys.path.insert(0, str(ROOT))
from scripts.golf.loop_8004_42_plus20.audit_retained_group import (  # noqa: E402
    perfect,
    profile,
    sha,
)
from scripts.golf.loop_8004_42_plus20.agent_new_low39.audit_lane import (  # noqa: E402
    run_known,
    structure,
)


TASK_RE = re.compile(r"task0*(151|213|122|94|220|260|342|331)(?!\d)")


def prior_known_by_sha() -> dict[str, dict[str, object]]:
    """Index fail-closed full-known results already produced by the validator."""
    index: dict[str, dict[str, object]] = {}
    known_dir = ROOT / "scripts/golf/loop_8003_40/agent_changed_resume/known"
    for path in known_dir.glob("task260_*.json"):
        try:
            payload = json.loads(path.read_text())
            candidate = payload["decision"]["candidate"]
            known = candidate["known"]
            index[candidate["sha256"]] = {
                "right": known["right"],
                "wrong": known["wrong"],
                "errors": known["errors"],
                "skipped": known["skipped"],
                "total": known["total_seen"],
                "evidence": str(path.relative_to(ROOT)),
            }
        except (KeyError, OSError, ValueError, TypeError):
            continue
    return index


def main() -> None:
    known_cache = prior_known_by_sha()
    base_data: dict[int, bytes] = {}
    base_cost: dict[int, int] = {}
    baseline_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(BASE) as archive:
        for task in TARGETS:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            actual = profile(data, name)
            model = onnx.load_model_from_string(data)
            row = {
                "task": task,
                "sha256": sha(data),
                "actual_cost": actual,
                "structure": structure(copy.deepcopy(model), task),
                "known_dual": "run in this audit session before cost-only restart; see REPORT.md",
            }
            baseline_rows.append(row)
            base_data[task] = data
            base_cost[task] = int(actual["cost"])
            print(
                f"baseline task{task:03d} cost={actual['cost']} "
                "known=previously_observed_full_dual",
                flush=True,
            )

    seen: set[tuple[int, str]] = set()
    candidate_rows: list[dict[str, object]] = []
    for path in sorted((ROOT / "scripts/golf").rglob("*.onnx")):
        match = TASK_RE.search(path.as_posix())
        if match is None:
            continue
        task = int(match.group(1))
        if task not in TARGETS:
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            candidate_rows.append({"task": task, "path": str(path.relative_to(ROOT)), "read_error": str(exc)})
            continue
        digest = hashlib.sha256(data).hexdigest()
        key = (task, digest)
        if key in seen:
            continue
        seen.add(key)
        actual = profile(data, path.name)
        lower = actual.get("cost") is not None and int(actual["cost"]) < base_cost[task]
        baseline_same = data == base_data[task]
        row: dict[str, object] = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": digest,
            "same_as_baseline": baseline_same,
            "actual_cost": actual,
            "strictly_cheaper": lower,
        }
        if lower:
            try:
                model = onnx.load_model_from_string(data)
                disable = known_cache.get(digest)
                if disable is None:
                    disable = run_known(copy.deepcopy(model), task, True)
                default = run_known(copy.deepcopy(model), task, False) if perfect(disable) else {
                    "not_run": "disable_all_not_perfect"
                }
                row["known_disable_all"] = disable
                row["known_default"] = default
                row["structure"] = structure(copy.deepcopy(model), task) if perfect(disable) and perfect(default) else None
                row["promising_pre_fresh"] = perfect(disable) and perfect(default)
                print(
                    f"LOWER task{task:03d} cost={actual['cost']} "
                    f"known={disable.get('right')}/{disable.get('total')} "
                    f"path={path.relative_to(ROOT)}",
                    flush=True,
                )
            except Exception as exc:
                row["audit_error"] = f"{type(exc).__name__}: {exc}"
                row["promising_pre_fresh"] = False
        candidate_rows.append(row)

    result = {
        "baseline": {"path": BASE.name, "sha256": sha(BASE.read_bytes())},
        "targets": list(TARGETS),
        "baseline_rows": baseline_rows,
        "unique_candidate_count": len(candidate_rows),
        "candidate_rows": candidate_rows,
        "lower_rows": [row for row in candidate_rows if row.get("strictly_cheaper")],
        "promising": [
            {"task": row["task"], "path": row["path"], "sha256": row["sha256"]}
            for row in candidate_rows
            if row.get("promising_pre_fresh")
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

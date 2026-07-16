#!/usr/bin/env python3
"""Incremental loose/ZIP inventory for expansion lane 94.

The full SHA inventory for 19/20 targets already exists in mid20_84 and
mid20b_86.  This script scans the current repository, SHA-deduplicates by
task, and emits only SHAs not present in those predecessor inventories.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TARGETS = (102, 25, 250, 62, 324, 308, 8, 275, 338, 333, 268, 184, 377, 109, 160, 99, 279, 345, 170, 245)
PREDECESSORS = (
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20_84/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_task333_finite81/candidate_inventory.json",
)
MAX_BYTES = 1_440_000
FILE_RE = re.compile(r"^task(\d{3})(?:[^0-9].*)?\.onnx$", re.IGNORECASE)
MEMBER_RE = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("submission.zip is not the approved 8006.61 authority")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = {task: archive.read(f"task{task:03d}.onnx") for task in TARGETS}
    authority_sha = {task: digest(data) for task, data in authority.items()}

    old: dict[int, set[str]] = defaultdict(set)
    predecessor_meta = []
    for path in PREDECESSORS:
        report = json.loads(path.read_text())
        dedicated_task = 333 if "baseline_task333_sha256" in report else None
        for row in report["rows"]:
            task = int(row.get("task", dedicated_task))
            if task in TARGETS:
                old[task].add(row["sha256"])
        report_targets = report.get("targets")
        if report_targets is None and "baseline_task333_sha256" in report:
            report_targets = [333]
        predecessor_meta.append({
            "path": relative(path),
            "baseline_zip_sha256": report["baseline_zip_sha256"],
            "target_intersection": sorted(set(report_targets) & set(TARGETS)),
        })

    current: dict[int, dict[str, dict]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    def add(task: int, data: bytes, source: str, kind: str) -> None:
        counts[f"{kind}_observations"] += 1
        if len(data) > MAX_BYTES:
            counts["oversize_observations"] += 1
            return
        sha = digest(data)
        if sha == authority_sha[task]:
            counts["authority_duplicates"] += 1
            return
        row = current[task].setdefault(sha, {"sha256": sha, "bytes": len(data), "sources": [], "source_kinds": []})
        row["sources"].append(source)
        row["source_kinds"].append(kind)

    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "__pycache__"}]
        base = Path(directory)
        for name in files:
            match = FILE_RE.match(name)
            if not match or int(match.group(1)) not in TARGETS:
                continue
            task = int(match.group(1))
            path = base / name
            try:
                add(task, path.read_bytes(), relative(path), "loose")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": relative(path), "error": f"{type(exc).__name__}: {exc}"})

    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "__pycache__"}]
        base = Path(directory)
        for name in files:
            if not name.lower().endswith(".zip"):
                continue
            path = base / name
            counts["zip_files_seen"] += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        match = MEMBER_RE.search(member)
                        if match and int(match.group(1)) in TARGETS:
                            task = int(match.group(1))
                            add(task, archive.read(member), f"{relative(path)}::{member}", "zip")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": relative(path), "error": f"{type(exc).__name__}: {exc}"})

    new_rows = []
    per_task = {}
    for task in TARGETS:
        rows = current[task]
        unseen = []
        for sha, row in rows.items():
            row["sources"] = sorted(set(row["sources"]))
            row["source_kinds"] = sorted(set(row["source_kinds"]))
            if sha not in old[task]:
                row["task"] = task
                unseen.append(row)
                new_rows.append(row)
        per_task[str(task)] = {
            "authority_sha256": authority_sha[task],
            "previous_unique_sha": len(old[task]),
            "current_unique_nonauthority_sha": len(rows),
            "incremental_new_sha": len(unseen),
        }
    new_rows.sort(key=lambda row: (row["task"], row["sha256"]))
    report = {
        "authority_zip": relative(AUTHORITY),
        "authority_zip_sha256": AUTHORITY_SHA256,
        "targets": list(TARGETS),
        "predecessors": predecessor_meta,
        "counts": dict(counts),
        "previous_unique_sha_total_for_targets": sum(len(old[task]) for task in TARGETS),
        "current_unique_nonauthority_sha_total": sum(len(current[task]) for task in TARGETS),
        "incremental_new_sha_total": len(new_rows),
        "per_task": per_task,
        "new_rows": new_rows,
        "errors": errors,
    }
    (HERE / "inventory_delta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "counts": report["counts"],
        "previous_unique_sha_total_for_targets": report["previous_unique_sha_total_for_targets"],
        "current_unique_nonauthority_sha_total": report["current_unique_nonauthority_sha_total"],
        "incremental_new_sha_total": report["incremental_new_sha_total"],
        "incremental_by_task": {task: row["incremental_new_sha"] for task, row in per_task.items() if row["incremental_new_sha"]},
        "error_count": len(errors),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

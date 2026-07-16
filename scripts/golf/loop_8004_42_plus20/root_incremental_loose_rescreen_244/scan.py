#!/usr/bin/env python3
"""Rescreen every loose ONNX produced by the current loop against 8009.46.

This is inventory-only.  It never edits the authority, score ledger, or stage.
Rows are SHA-deduplicated per task and ranked by measured official cost.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOOP = ROOT / "scripts/golf/loop_8004_42_plus20"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
STAGE = ROOT / "others/71407"
TASK_RE = re.compile(r"task[_-]?(\d{3})(?!\d)", re.IGNORECASE)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def infer_task(path: Path) -> int | None:
    file_hits = {int(value) for value in TASK_RE.findall(path.name)}
    if len(file_hits) == 1:
        return next(iter(file_hits))
    path_hits = {int(value) for value in TASK_RE.findall(str(path.relative_to(LOOP)))}
    if len(path_hits) == 1:
        return next(iter(path_hits))
    return None


def main() -> int:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("8009.46 authority hash mismatch")

    current_cost: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            current_cost[int(row["task"].removeprefix("task"))] = int(row["cost"])

    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_sha = {
            task: sha256(archive.read(f"task{task:03d}.onnx")) for task in range(1, 401)
        }

    stage_sha: set[str] = set()
    manifest = json.loads((STAGE / "MANIFEST.json").read_text())
    for row in manifest["active_candidates"]:
        stage_sha.add(row["sha256"])

    observations = 0
    ambiguous = []
    errors = []
    unique: dict[int, dict[str, dict]] = defaultdict(dict)
    for path in sorted(LOOP.rglob("*.onnx")):
        observations += 1
        task = infer_task(path)
        if task is None or task not in current_cost:
            ambiguous.append(str(path.relative_to(ROOT)))
            continue
        try:
            data = path.read_bytes()
            digest = sha256(data)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(path.relative_to(ROOT)), "error": repr(exc)})
            continue
        if digest == authority_sha[task]:
            continue
        row = unique[task].setdefault(
            digest,
            {
                "task": task,
                "sha256": digest,
                "bytes": len(data),
                "sources": [],
                "staged": digest in stage_sha,
            },
        )
        row["sources"].append(str(path.relative_to(ROOT)))

    work = []
    for task in sorted(unique):
        for row in unique[task].values():
            work.append((task, row, ROOT / row["sources"][0]))

    helper = HERE / "measure_one.py"
    child_env = dict(os.environ)
    child_env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    def measure(item: tuple[int, dict, Path]) -> tuple[int, dict, dict | None, str | None]:
        task, row, source = item
        try:
            proc = subprocess.run(
                [sys.executable, str(helper), str(source)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=30,
                env=child_env,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            return task, row, None, repr(exc)
        if proc.returncode != 0:
            return task, row, None, f"isolated returncode {proc.returncode}"
        try:
            return task, row, json.loads(proc.stdout.strip().splitlines()[-1]), None
        except Exception as exc:  # noqa: BLE001
            return task, row, None, f"bad isolated output: {exc!r}"

    measured = 0
    lower = []
    nonlower = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(measure, work)
        for task, row, result, error in results:
            if error is not None or result is None:
                row["measure_error"] = error
                errors.append({"path": row["sources"][0], "error": error})
                continue
            cost = result["cost"]
            row.update(result)
            if min(result.values()) < 0:
                row["measure_error"] = "official profiler returned a negative component"
                errors.append({
                    "path": row["sources"][0],
                    "error": row["measure_error"],
                })
                continue
            measured += 1
            row["authority_cost"] = current_cost[task]
            row["delta"] = int(cost) - current_cost[task]
            row["sources"] = sorted(set(row["sources"]))
            if cost < current_cost[task]:
                lower.append(row)
            else:
                nonlower += 1

    lower.sort(key=lambda row: (row["delta"], row["task"], row["sha256"]))
    report = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "stage_manifest_active": len(manifest["active_candidates"]),
        "loose_observations": observations,
        "ambiguous_path_count": len(ambiguous),
        "unique_nonauthority_sha": sum(len(rows) for rows in unique.values()),
        "measured_unique_sha": measured,
        "strict_lower_count": len(lower),
        "strict_lower_unstaged_count": sum(not row["staged"] for row in lower),
        "nonlower_count": nonlower,
        "strict_lower": lower,
        "ambiguous_paths": ambiguous,
        "errors": errors,
    }
    (HERE / "scan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {
        "strict_lower", "ambiguous_paths", "errors"
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

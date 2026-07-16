#!/usr/bin/env python3
"""Find and profile local historical models for the B9 task set."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.loop_7999_13.archive_zip_sweep import (  # noqa: E402
    static_cost,
    structural_reason,
)


TARGETS = {156, 182, 216, 237, 238, 284, 379}
TASK_PATTERN = re.compile(r"(?:^|/)task(156|182|216|237|238|284|379)(?:/|[^0-9])")
EXCLUDED = {
    "lane_archive_zip_sweep",
    "lane_baseline_fresh100",
    "lane_b9/baseline",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(task: int, path: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(ROOT / "scripts/golf/verify_candidate_timeout.py"),
        "--task",
        str(task),
        "--onnx",
        str(path),
        "--timeout",
        "45",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "timed_out": True,
            "stderr_tail": str(exc.stderr or "")[-2000:],
        }
    parsed: dict[str, object] = {}
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "exit_code": completed.returncode,
        "result": parsed,
        "stderr_tail": completed.stderr[-2000:],
    }


def main() -> None:
    baselines = json.loads((HERE / "baseline_inventory.json").read_text())["tasks"]
    base_costs = {
        int(task): int(row["score"]["cost"])
        for task, row in baselines.items()
    }
    base_hashes = {
        int(task): str(row["sha256"])
        for task, row in baselines.items()
    }
    inventory: list[dict[str, object]] = []
    selected: dict[tuple[int, str], tuple[int, Path]] = {}
    paths = (
        path
        for search_root in (ROOT / "scripts/golf", ROOT / "artifacts", ROOT / "others")
        for path in search_root.rglob("*.onnx")
    )
    for path in paths:
        relative = str(path.relative_to(ROOT))
        match = TASK_PATTERN.search(relative)
        if not match or any(item in relative for item in EXCLUDED):
            continue
        task = int(match.group(1))
        sha = digest(path)
        if sha == base_hashes[task]:
            inventory.append({"task": task, "path": relative, "status": "baseline_duplicate"})
            continue
        try:
            model = onnx.load(path)
            reason = structural_reason(model)
            floor = static_cost(model) if reason is None else None
        except Exception as exc:
            reason = f"load:{type(exc).__name__}"
            floor = None
        row: dict[str, object] = {
            "task": task,
            "path": relative,
            "sha256": sha,
            "static_cost": floor,
            "structural_reason": reason,
        }
        if reason is None and floor is not None and floor < base_costs[task]:
            row["status"] = "profile"
            key = (task, sha)
            old = selected.get(key)
            if old is None or floor < old[0]:
                selected[key] = (floor, path)
        else:
            row["status"] = "prefilter_reject"
        inventory.append(row)

    jobs = [(task, floor, path) for (task, _), (floor, path) in selected.items()]
    jobs.sort(key=lambda item: (item[0], item[1], str(item[2])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: score(item[0], item[2]), jobs))
    for row, job in zip(results, jobs):
        row["static_cost"] = job[1]
        row["baseline_cost"] = base_costs[job[0]]
    results.sort(key=lambda row: (int(row["task"]), int(row["static_cost"])))

    (HERE / "local_history_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "local_history_actual_scores.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    concise = []
    for row in results:
        result = row.get("result", {})
        concise.append(
            {
                "task": row["task"],
                "path": row["path"],
                "static_cost": row["static_cost"],
                "actual_cost": result.get("cost") if isinstance(result, dict) else None,
                "correct": result.get("correct") if isinstance(result, dict) else None,
                "ok": result.get("ok") if isinstance(result, dict) else None,
            }
        )
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()

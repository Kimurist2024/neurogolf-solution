#!/usr/bin/env python3
"""Run every LB probe candidate on two independent fresh-generator seeds."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
from screen_all import fresh_dual, resolve_source  # noqa: E402


COUNT = 500
SEED_BASES = (92_000_000, 93_000_000)


def resolve(row: dict) -> bytes:
    data = resolve_source(row["path"], int(row["task"]))
    if data is None:
        raise RuntimeError(f"unresolved {row['path']}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"SHA mismatch: expected={row['sha256']} got={digest}")
    return data


def main() -> int:
    manifest = json.loads((HERE / "probe_manifest.json").read_text())
    by_task: dict[int, list[dict]] = defaultdict(list)
    for row in manifest["candidates"]:
        item = dict(row)
        item["data"] = resolve(row)
        by_task[int(row["task"])].append(item)

    runs = []
    for task, rows in sorted(by_task.items()):
        runtime_rows = [{"sha256": row["sha256"], "data": row["data"]} for row in rows]
        for seed_base in SEED_BASES:
            runs.append(fresh_dual(task, runtime_rows, COUNT, seed_base + task))

    per_sha: dict[str, dict] = {}
    metadata = {row["sha256"]: row for rows in by_task.values() for row in rows}
    for digest, row in metadata.items():
        per_sha[digest] = {
            "task": int(row["task"]),
            "sha256": digest,
            "candidate_cost": int(row["candidate_cost"]),
            "runs": [],
        }
    for run in runs:
        for digest, modes in run["candidates"].items():
            per_sha[digest]["runs"].append({
                "seed": run["seed"],
                "requested": run["requested"],
                "valid": run["valid"],
                "generation_errors": run["generation_errors"],
                "conversion_skips": run["conversion_skips"],
                "modes": modes,
            })
    for item in per_sha.values():
        mode_rates = []
        for run in item["runs"]:
            for stats in run["modes"].values():
                total = stats.get("right", 0) + stats.get("wrong", 0) + stats.get("errors", 0)
                mode_rates.append(stats.get("right", 0) / total if total else 0.0)
        item["minimum_mode_rate"] = min(mode_rates) if mode_rates else 0.0

    report = {
        "candidate_count": len(per_sha),
        "task_count": len(by_task),
        "count_per_seed": COUNT,
        "seed_bases": list(SEED_BASES),
        "runs": runs,
        "per_sha": sorted(per_sha.values(), key=lambda row: (row["task"], row["candidate_cost"], row["sha256"])),
    }
    output = HERE / "audit/fresh_two_seed.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": report["candidate_count"],
        "task_count": report["task_count"],
        "minimum_rates": {row["sha256"][:12]: row["minimum_mode_rate"] for row in report["per_sha"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

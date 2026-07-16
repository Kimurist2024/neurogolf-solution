#!/usr/bin/env python3
"""Classify every SHA-distinct collected lead without promoting anything."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/golf/scratch_codex_plus10/wave1_b/scan_candidates.py"
spec = importlib.util.spec_from_file_location("high_gain_static", SOURCE)
assert spec is not None and spec.loader is not None
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

scanner.ROOT = ROOT
scanner.HERE = HERE
scanner.TASKS = (145, 191, 204, 205, 285)
scanner.BASE_ZIP = ROOT / "submission_base_8004.50.zip"
scanner.POOL_ROOTS = (
    ROOT / "others/1/70208",
    ROOT / "others/1/70209",
    ROOT / "others/1/70210",
    ROOT / "others/2/1200",
    ROOT / "others/2/1201",
    ROOT / "others/2/1202",
    ROOT / "others/2/1203",
    ROOT / "others/3",
)
scanner.TREE_ROOTS = (ROOT / "artifacts", ROOT / "scripts/golf")

BASE_COST = {145: 5130, 191: 3444, 204: 2240, 205: 1042, 285: 8623}


def main() -> None:
    base, candidates, errors = scanner.collect()
    rows: list[dict[str, object]] = []
    for task in scanner.TASKS:
        base_sha = scanner.sha256(base[task])
        for digest, item in candidates[task].items():
            row: dict[str, object] = {
                "task": task,
                "sha256": digest,
                "sources": item["sources"],
                "serialized_bytes": len(item["bytes"]),
            }
            if digest == base_sha:
                row["status"] = "BASE_IDENTICAL"
                rows.append(row)
                continue
            ok, reason, model = scanner.static_check(item["bytes"])
            row["static_gate"] = reason
            if not ok or model is None:
                row["status"] = "REJECT_STATIC"
                rows.append(row)
                continue
            cost, memory, params = scanner.static_cost_floor(model)
            gain = math.log(BASE_COST[task] / cost) if cost > 0 else 25.0
            row.update(
                static_cost_floor=cost,
                static_memory_floor=memory,
                parameter_floor=params,
                optimistic_gain=gain,
            )
            if cost >= BASE_COST[task]:
                row["status"] = "REJECT_NOT_CHEAPER_STATIC_FLOOR"
            elif gain < 0.05:
                row["status"] = "REJECT_BELOW_GAIN_PRIORITY"
            else:
                row["status"] = "EXECUTION_AUDIT_REQUIRED"
            rows.append(row)

    counts = {
        str(task): dict(
            Counter(str(row["status"]) for row in rows if row["task"] == task)
        )
        for task in scanner.TASKS
    }
    payload = {
        "baseline": "submission_base_8004.50.zip",
        "baseline_cost": {str(k): v for k, v in BASE_COST.items()},
        "collection_errors": errors,
        "unique_sha_count": len(rows),
        "status_counts": counts,
        "rows": rows,
    }
    (HERE / "static_inventory.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()

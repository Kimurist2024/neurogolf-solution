#!/usr/bin/env python3
"""Pin the 8012.15 authority and materialize the mid/high-cost campaign scope."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"


def main() -> int:
    digest = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if digest != AUTHORITY_SHA256:
        raise RuntimeError(f"authority SHA mismatch: {digest}")

    costs: dict[int, int] = {}
    scores: dict[int, float] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            costs[task] = int(row["cost"])
            scores[task] = float(row["score"])
    if set(costs) != set(range(1, 401)):
        raise RuntimeError("all_scores.csv is not a complete 400-task census")

    scope = {
        task: cost
        for task, cost in costs.items()
        if 167 <= cost <= 500 and scores[task] < 24.999999
    }
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(AUTHORITY) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".onnx"))
        if members != [f"task{task:03d}.onnx" for task in range(1, 401)]:
            raise RuntimeError("authority is not the canonical ordered 400-task archive")
        for task in range(1, 401):
            (baseline / f"task{task:03d}.onnx").write_bytes(
                archive.read(f"task{task:03d}.onnx")
            )

    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": digest,
        "authority_lb": 8012.15,
        "scope_rule": "167 <= current cost <= 500 and not score25",
        "scope_count": len(scope),
        "scope_cost_sum": sum(scope.values()),
        "scope": [{"task": task, "cost": scope[task]} for task in sorted(scope)],
        "costs": {str(task): costs[task] for task in sorted(costs)},
        "protected_writes": "only scripts/golf/restart8012_midhigh_3w_406",
    }
    (HERE / "authority.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "base_costs.json").write_text(
        json.dumps({"costs": payload["costs"]}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "authority_sha256": digest,
        "scope_count": len(scope),
        "scope_cost_sum": sum(scope.values()),
        "baseline": str(baseline.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

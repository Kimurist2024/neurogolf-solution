#!/usr/bin/env python3
"""Locate the owning evidence report for every unstaged strict-lower SHA."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOOP = ROOT / "scripts/golf/loop_8004_42_plus20"


def owner_report(source: Path) -> Path | None:
    current = source.parent
    while current != LOOP.parent:
        report = current / "REPORT.md"
        if report.is_file():
            return report
        if current == LOOP:
            break
        current = current.parent
    return None


scan = json.loads((HERE / "scan.json").read_text())
rows = []
for candidate in scan["strict_lower"]:
    if candidate["staged"]:
        continue
    source = ROOT / candidate["sources"][0]
    report = owner_report(source)
    rows.append({
        "task": candidate["task"],
        "sha256": candidate["sha256"],
        "cost": candidate["cost"],
        "authority_cost": candidate["authority_cost"],
        "source": str(source.relative_to(ROOT)),
        "owner_report": str(report.relative_to(ROOT)) if report else None,
        "source_marked_rejected": any(
            token in str(source).lower()
            for token in ("reject", "probe", "bad", "control", "history", "baseline")
        ),
    })

result = {
    "unstaged_strict_lower": len(rows),
    "with_owner_report": sum(row["owner_report"] is not None for row in rows),
    "without_owner_report": sum(row["owner_report"] is None for row in rows),
    "without_owner_report_rows": [row for row in rows if row["owner_report"] is None],
    "rows": rows,
}
(HERE / "classification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key not in {
    "rows", "without_owner_report_rows"
}}, indent=2))

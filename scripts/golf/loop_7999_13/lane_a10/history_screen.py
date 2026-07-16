#!/usr/bin/env python3
"""Summarize repository-wide harvested history for the A10 target set."""

from __future__ import annotations

import collections
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (37, 48, 92, 222, 226, 297, 345)


def main() -> None:
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )["rows"]
    b5 = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_b5/existing_scan.json").read_text()
    )
    output: dict[str, object] = {
        "source": "lane_harvest repository-wide loose/archive inventory plus lane_b5 full actual screen",
        "targets": {},
    }
    for task in TARGETS:
        rows = [row for row in harvest if row.get("task") == task]
        actual = [row for row in rows if isinstance(row.get("actual_screen_cost"), int)]
        record: dict[str, object] = {
            "harvest_unique": len(rows),
            "harvest_stages": dict(sorted(collections.Counter(row.get("stage") for row in rows).items())),
            "minimum_harvest_static_floor": min(
                (row["static_cost_floor"] for row in rows if isinstance(row.get("static_cost_floor"), int)),
                default=None,
            ),
            "minimum_harvest_actual": min(
                (row["actual_screen_cost"] for row in actual), default=None
            ),
            "strict_harvest_known_winner": str(task)
            in json.loads(
                (ROOT / "scripts/golf/loop_7999_13/lane_harvest/known_winner_manifest.json").read_text()
            )["winners"],
        }
        if task in (297, 345):
            full = [row for row in b5 if row.get("task") == task]
            record["lane_b5_unique_actual_screen"] = len(full)
            record["lane_b5_known_correct"] = sum(row.get("correct") is True for row in full)
            record["lane_b5_strictly_cheaper"] = sum(row.get("cheaper") is True for row in full)
            record["lane_b5_cheaper_and_known_correct"] = sum(
                row.get("cheaper") is True and row.get("correct") is True for row in full
            )
            record["lane_b5_minimum_actual"] = min(
                (row["cost"] for row in full if isinstance(row.get("cost"), int)), default=None
            )
        output["targets"][str(task)] = record
    (HERE / "history_screen.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

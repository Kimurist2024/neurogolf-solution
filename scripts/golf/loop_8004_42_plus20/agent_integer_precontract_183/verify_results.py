#!/usr/bin/env python3
"""Expand graph-equivalent aliases and verify lane integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = json.loads((HERE / "actual_profile_results.json").read_text())
    screen = json.loads((HERE / "one_known_screen.json").read_text())
    assert results["candidate_count"] == 98 == len(results["rows"])
    assert results["strict_lower_count"] == 0
    assert results["root_unchanged"]
    assert screen["runtime_ok_count"] == 98
    assert screen["threshold_equal_count"] == 98
    labels = {row["label"] for row in results["rows"]}
    files = {path.stem for path in (HERE / "candidates").glob("*.onnx")}
    assert labels == files
    concrete = []
    for row in results["rows"]:
        assert row["cost_delta"] >= 0 and not row["strict_lower"]
        assert not row["new_ops"]
        assert not row["private_zero_or_approximation"]
        assert sha256(REPO / row["path"]) == row["sha256"]
        aliases = row["proof"].get("equivalent_aliases") or [row["label"]]
        for alias in aliases:
            concrete.append(
                {
                    "task": row["task"],
                    "class": row["label"],
                    "alias": alias,
                    "actual_profile": row["actual_profile"],
                    "cost_delta": row["cost_delta"],
                    "strict_lower": row["strict_lower"],
                }
            )
    assert len(concrete) == 140
    minima = {
        task: min(row["actual_profile"]["cost"] for row in concrete if row["task"] == task)
        for task in (74, 200, 211)
    }
    assert minima == {74: 135, 200: 348, 211: 66}
    assert sha256(REPO / "submission_base_8009.46.zip") == results["authority_sha256"]
    assert sha256(REPO / "submission.zip") == results["authority_sha256"]
    assert sha256(REPO / "all_scores.csv") == results["root_hashes_before"]["all_scores.csv"]
    (HERE / "concrete_costs.json").write_text(json.dumps(concrete, indent=2) + "\n")
    print(json.dumps({"verified": True, "classes": len(labels), "concrete": len(concrete), "minima": minima}, indent=2))


if __name__ == "__main__":
    main()

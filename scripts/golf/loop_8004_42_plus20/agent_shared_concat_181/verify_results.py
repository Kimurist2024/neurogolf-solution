#!/usr/bin/env python3
"""Self-check the shared-Concat lane artifacts and protected-root integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    screen = json.loads((HERE / "screen_results.json").read_text())
    costs = json.loads((HERE / "concrete_costs.json").read_text())
    manifest = json.loads((HERE / "winner_manifest.json").read_text())
    assert screen["candidate_class_count"] == 72 == len(screen["rows"])
    assert screen["concrete_candidate_count"] == 192 == len(costs)
    assert screen["strict_lower_count"] == 0
    assert screen["root_unchanged"]
    assert manifest["status"] == "REJECT_NO_STRICT_LOWER" and not manifest["accepted"]
    labels = {row["label"] for row in screen["rows"]}
    files = {path.stem for path in (HERE / "candidates").glob("*.onnx")}
    assert labels == files
    for row in screen["rows"]:
        assert row["checks"]["full_checker"]["pass"]
        assert row["checks"]["strict_data_prop"]["pass"]
        assert row["actual_profile"]["cost"] == row["baseline_profile"]["cost"] + row["cost_delta"]
        assert row["cost_delta"] > 0 and not row["strict_lower"]
        assert row["structure"]["no_new_lookup_or_hardmax"]
        assert not row["structure"]["private_zero_or_approximation"]
        assert sha256(REPO / row["path"]) == row["sha256"]
    minima = {
        task: min(item["actual_profile"]["cost"] for item in costs if item["task"] == task)
        for task in (13, 55, 99, 281)
    }
    assert minima == {13: 378, 55: 270, 99: 582, 281: 213}
    assert sha256(REPO / "submission_base_8009.46.zip") == screen["authority_sha256"]
    assert sha256(REPO / "submission.zip") == screen["authority_sha256"]
    assert sha256(REPO / "others/71407/task013.onnx") == screen["task013_sha256"]
    print(json.dumps({"verified": True, "classes": len(labels), "concrete": len(costs), "minima": minima}, indent=2))


if __name__ == "__main__":
    main()

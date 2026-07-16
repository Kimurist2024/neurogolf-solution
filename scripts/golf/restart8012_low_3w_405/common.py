#!/usr/bin/env python3
"""Shared fail-closed helpers for the 8012.15 cost<=166 campaign."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
OLD_AUTHORITY = ROOT / "submission_base_8011.05.zip"
CANDIDATES = HERE / "candidates"


def import_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PATTERN = import_path(
    "restart8012_low_pattern_helpers",
    ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/scan_patterns.py",
)
HISTORY = import_path(
    "restart8012_low_history_helpers",
    ROOT / "scripts/golf/root_cost50_history_scan_298/scan.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_authority() -> None:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("submission_base_8012.15.zip SHA-256 mismatch")


def current_costs(minimum: int = 0, maximum: int = 166) -> dict[int, int]:
    result: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            score = float(row["score"])
            if minimum <= cost <= maximum and score < 25.0:
                result[task] = cost
    return result


def authority_diff() -> dict[str, Any]:
    """Prove whether any member in this lane changed since 8011.05."""
    costs = current_costs()
    changed_all: list[int] = []
    changed_scope: list[int] = []
    with zipfile.ZipFile(OLD_AUTHORITY) as old, zipfile.ZipFile(AUTHORITY) as new:
        for task in range(1, 401):
            name = f"task{task:03d}.onnx"
            if old.read(name) != new.read(name):
                changed_all.append(task)
                if task in costs:
                    changed_scope.append(task)
    return {
        "old": str(OLD_AUTHORITY.relative_to(ROOT)),
        "new": str(AUTHORITY.relative_to(ROOT)),
        "changed_all": changed_all,
        "changed_cost_le166_non_score25": changed_scope,
        "scope_byte_identical": not changed_scope,
    }


def compact_source(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


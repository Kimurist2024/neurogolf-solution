#!/usr/bin/env python3
"""Inventory and deduplicate every reachable task333 ONNX file/member."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT_MODELS = HERE / "unique_candidates"
BASE_ZIP = ROOT / "submission_base_8005.17.zip"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SWEEP = load_module(
    "task333_wave30b",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def profile(data: bytes, label: str) -> dict[str, int]:
    try:
        return SWEEP.profiler_cost(data, 333, label)
    except Exception as exc:  # noqa: BLE001
        return {"memory": -1, "params": -1, "cost": -1, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[str]] = defaultdict(list)
    payloads: dict[str, bytes] = {}
    onnx_files_seen = 0
    zip_files_seen = 0
    zip_members_seen = 0

    for path in ROOT.rglob("*.onnx"):
        if "task333" not in path.name.lower():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        onnx_files_seen += 1
        sha = digest(data)
        sources[sha].append(rel(path))
        payloads.setdefault(sha, data)

    for archive_path in ROOT.rglob("*.zip"):
        zip_files_seen += 1
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.namelist():
                    if Path(member).name.lower() != "task333.onnx":
                        continue
                    try:
                        data = archive.read(member)
                    except (KeyError, RuntimeError, OSError):
                        continue
                    zip_members_seen += 1
                    sha = digest(data)
                    sources[sha].append(f"{rel(archive_path)}::{member}")
                    payloads.setdefault(sha, data)
        except (zipfile.BadZipFile, OSError):
            continue

    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline_data = archive.read("task333.onnx")
    baseline_sha = digest(baseline_data)
    baseline_profile = profile(baseline_data, "baseline_8005_17")

    rows = []
    for index, sha in enumerate(sorted(payloads)):
        data = payloads[sha]
        path = OUT_MODELS / f"{sha}.onnx"
        path.write_bytes(data)
        try:
            static = SWEEP.static_audit(data)
        except Exception as exc:  # noqa: BLE001
            static = {"error": f"{type(exc).__name__}: {exc}", "full_check": False}
        actual = profile(data, f"unique_{index:03d}")
        rows.append(
            {
                "sha256": sha,
                "extracted_path": rel(path),
                "source_count": len(sources[sha]),
                "sources": sorted(set(sources[sha])),
                "serialized_bytes": len(data),
                "profile": actual,
                "strictly_lower_than_8005_17": 0 < actual.get("cost", -1) < baseline_profile["cost"],
                "same_as_baseline_member": sha == baseline_sha,
                "static": static,
            }
        )
        print(index + 1, len(payloads), sha[:12], actual.get("cost"), len(sources[sha]), flush=True)

    rows.sort(key=lambda row: (row["profile"].get("cost", 10**18) if row["profile"].get("cost", -1) >= 0 else 10**18, row["sha256"]))
    result = {
        "baseline_zip": rel(BASE_ZIP),
        "baseline_zip_sha256": digest(BASE_ZIP.read_bytes()),
        "baseline_task333_sha256": baseline_sha,
        "baseline_profile": baseline_profile,
        "onnx_files_seen": onnx_files_seen,
        "zip_files_seen": zip_files_seen,
        "zip_task333_members_seen": zip_members_seen,
        "total_source_references": sum(len(items) for items in sources.values()),
        "unique_sha_count": len(rows),
        "strict_lower_unique_count": sum(row["strictly_lower_than_8005_17"] for row in rows),
        "rows": rows,
    }
    (HERE / "candidate_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("onnx_files_seen", "zip_files_seen", "zip_task333_members_seen", "unique_sha_count", "strict_lower_unique_count")}, indent=2))


if __name__ == "__main__":
    main()

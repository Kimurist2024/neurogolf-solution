#!/usr/bin/env python3
"""Inventory every repository SHA for task187/task191/task319."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
TARGETS = {187, 191, 319}
ZIP_MEMBER = re.compile(r"(?:^|/)task(187|191|319)\.onnx$", re.IGNORECASE)
LOOSE_TASK = re.compile(r"task(187|191|319)", re.IGNORECASE)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(part in {".git", ".venv", "node_modules"} for part in relative.parts) or HERE in path.parents


def add(
    unique: dict[int, dict[str, dict[str, object]]],
    task: int,
    data: bytes,
    source: str,
    counts: Counter[str],
) -> None:
    sha = digest(data)
    record = unique[task].get(sha)
    if record is None:
        unique[task][sha] = {
            "sha256": sha,
            "serialized_bytes": len(data),
            "sources": [source],
            "source_count": 1,
        }
        counts["unique_task_sha"] += 1
    else:
        record["source_count"] = int(record["source_count"]) + 1
        sources = record["sources"]
        assert isinstance(sources, list)
        if len(sources) < 100:
            sources.append(source)
        counts["duplicate_observations"] += 1


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA:
        raise RuntimeError("authority drift")
    unique: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    zip_paths = sorted(path for path in ROOT.rglob("*.zip") if not excluded(path))
    for path in zip_paths:
        counts["zip_files_seen"] += 1
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    match = ZIP_MEMBER.search(member)
                    if not match:
                        continue
                    task = int(match.group(1))
                    counts["zip_target_members_seen"] += 1
                    add(
                        unique,
                        task,
                        archive.read(member),
                        f"{path.relative_to(ROOT)}::{member}",
                        counts,
                    )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "kind": "zip",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    loose_paths = sorted(path for path in ROOT.rglob("*.onnx") if not excluded(path))
    for path in loose_paths:
        match = LOOSE_TASK.search(str(path.relative_to(ROOT)))
        if not match:
            continue
        counts["loose_target_files_seen"] += 1
        task = int(match.group(1))
        try:
            add(
                unique,
                task,
                path.read_bytes(),
                str(path.relative_to(ROOT)),
                counts,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "kind": "loose",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    authority_members: dict[int, str] = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in sorted(TARGETS):
            data = archive.read(f"task{task:03d}.onnx")
            authority_members[task] = digest(data)
            add(
                unique,
                task,
                data,
                f"{AUTHORITY.relative_to(ROOT)}::task{task:03d}.onnx",
                counts,
            )

    tasks: dict[str, object] = {}
    for task in sorted(TARGETS):
        records = sorted(unique[task].values(), key=lambda row: str(row["sha256"]))
        for record in records:
            record["is_authority"] = record["sha256"] == authority_members[task]
        tasks[str(task)] = {
            "authority_member_sha256": authority_members[task],
            "unique_sha_count": len(records),
            "records": records,
        }
    payload = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA,
        "scan_scope": "all repository ZIP task members and loose ONNX whose path contains task187/task191/task319; excludes .git/.venv/node_modules and this lane's generated artifacts",
        "counts": dict(counts),
        "errors": errors,
        "tasks": tasks,
    }
    (HERE / "history_inventory.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "counts": dict(counts),
                "errors": len(errors),
                "unique": {task: len(unique[task]) for task in sorted(TARGETS)},
            }
        )
    )


if __name__ == "__main__":
    main()

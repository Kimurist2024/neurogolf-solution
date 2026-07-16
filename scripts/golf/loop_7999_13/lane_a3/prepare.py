#!/usr/bin/env python3
"""Extract exact 7999.13 incumbents for the isolated A3 wave."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
EXPECTED = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
TASKS = (13, 88, 157, 182, 191, 280, 330)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if sha256(BASE.read_bytes()) != EXPECTED:
        raise RuntimeError("baseline identity changed")
    members: dict[str, object] = {}
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            destination = HERE / "baseline" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            members[str(task)] = {
                "path": str(destination.relative_to(ROOT)),
                "sha256": sha256(data),
                "bytes": len(data),
            }
    manifest = {"baseline_sha256": EXPECTED, "members": members}
    (HERE / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

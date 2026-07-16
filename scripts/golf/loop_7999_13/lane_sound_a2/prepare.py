#!/usr/bin/env python3
"""Extract exact 7999.13 incumbents into the isolated A2 lane."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_7999.13.zip"
EXPECTED = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
TASKS = (9, 77, 118, 173)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    if sha(BASE.read_bytes()) != EXPECTED:
        raise RuntimeError("baseline identity changed")
    result: dict[str, object] = {"baseline_sha256": EXPECTED, "members": {}}
    with zipfile.ZipFile(BASE) as archive:
        for task in TASKS:
            name = f"task{task:03d}.onnx"
            data = archive.read(name)
            path = HERE / "baseline" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            result["members"][str(task)] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(data),
                "bytes": len(data),
            }
    (HERE / "baseline_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

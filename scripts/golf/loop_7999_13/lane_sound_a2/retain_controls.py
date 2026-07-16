#!/usr/bin/env python3
"""Materialize provenance-pinned sound controls inside the isolated A2 lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = {
    9: HERE / "baseline" / "task009.onnx",
    77: ROOT / "artifacts" / "optimized" / "task077.onnx",
    173: ROOT / "artifacts" / "optimized" / "task173.onnx",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    controls = HERE / "sound_controls"
    controls.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"controls": {}}
    for task, source in SOURCES.items():
        data = source.read_bytes()
        destination = controls / f"task{task:03d}.onnx"
        destination.write_bytes(data)
        manifest["controls"][str(task)] = {
            "source": str(source.relative_to(ROOT)),
            "path": str(destination.relative_to(ROOT)),
            "sha256": sha256(data),
            "bytes": len(data),
        }
    (HERE / "sound_control_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

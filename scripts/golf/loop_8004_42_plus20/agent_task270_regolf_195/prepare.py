#!/usr/bin/env python3
"""Extract the immutable 8009.46 task270 authority with workspace guards."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
GUARDS = {
    ROOT / "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    ROOT / "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    ROOT / "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard() -> None:
    for path, expected in GUARDS.items():
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"guard failed: {path}: {actual} != {expected}")


def main() -> int:
    guard()
    with zipfile.ZipFile(ROOT / "submission_base_8009.46.zip") as archive:
        data = archive.read("task270.onnx")
    expected = "0d848124abafda1daf24fe5f779ed5249c9b8b2054854264dde838b05e27a443"
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"task270 authority SHA mismatch: {actual} != {expected}")
    out = HERE / "baseline/task270_authority.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    guard()
    manifest = ROOT / "others/71407/MANIFEST.json"
    print(f"observe-only {manifest.relative_to(ROOT)} sha256={digest(manifest)}")
    print(f"{out.relative_to(ROOT)} sha256={actual} bytes={len(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

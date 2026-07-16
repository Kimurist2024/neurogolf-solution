#!/usr/bin/env python3
"""Build the isolated task382 candidate ZIP without changing archive order."""

from __future__ import annotations

import copy
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
BASELINE = ROOT / "submission_base_7999.13.zip"
CANDIDATE = LANE / "candidates" / "task382.onnx"
OUTPUT = LANE / "submission_7999.13_task382_isolated.zip"
REPLACEMENT_NAME = "task382.onnx"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    replacement = CANDIDATE.read_bytes()
    with zipfile.ZipFile(BASELINE, "r") as source:
        infos = source.infolist()
        if sum(info.filename == REPLACEMENT_NAME for info in infos) != 1:
            raise RuntimeError(f"expected exactly one {REPLACEMENT_NAME}")
        archive_comment = source.comment
        with zipfile.ZipFile(OUTPUT, "w", allowZip64=True) as target:
            target.comment = archive_comment
            for info in infos:
                data = (
                    replacement
                    if info.filename == REPLACEMENT_NAME
                    else source.read(info)
                )
                target.writestr(copy.copy(info), data)

    with zipfile.ZipFile(BASELINE, "r") as base, zipfile.ZipFile(
        OUTPUT, "r"
    ) as candidate:
        if base.namelist() != candidate.namelist():
            raise RuntimeError("archive order changed")
        changed = [
            name for name in base.namelist() if base.read(name) != candidate.read(name)
        ]
        if changed != [REPLACEMENT_NAME]:
            raise RuntimeError(f"unexpected changed entries: {changed}")
        if candidate.testzip() is not None:
            raise RuntimeError("candidate ZIP failed CRC validation")

    print(f"wrote {OUTPUT}")
    print(f"sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()

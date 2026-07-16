#!/usr/bin/env python3
"""Extract the exact Wave15 task224/task400 members into the isolated B22 lane."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WAVE15 = HERE.parent / "submission_7999.13_wave15_candidate_meta.zip"
EXPECTED_ZIP = "0f106fa0d9599d4853397e0f9310e3ae1bcf47d6f418c6b9dec31e4a4490bc36"
EXPECTED = {
    224: "02d6386ace32270c71ee2072328187a4c3a2a8355babd6b69fdc4a0e5b6bac79",
    400: "89b419dbad732d3235ac1ab7d078ef22eef3209eb8b5f30e21d3a502ccd03389",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    if digest(WAVE15.read_bytes()) != EXPECTED_ZIP:
        raise RuntimeError("Wave15 ZIP identity mismatch")
    baseline = HERE / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(WAVE15) as archive:
        for task, expected in EXPECTED.items():
            member = f"task{task:03d}.onnx"
            data = archive.read(member)
            if digest(data) != expected:
                raise RuntimeError(f"{member} identity mismatch")
            path = baseline / member
            path.write_bytes(data)
            rows.append(
                {
                    "task": task,
                    "member": member,
                    "path": str(path.relative_to(ROOT)),
                    "bytes": len(data),
                    "sha256": expected,
                }
            )
    payload = {
        "wave15": str(WAVE15.relative_to(ROOT)),
        "wave15_sha256": EXPECTED_ZIP,
        "rows": rows,
    }
    (HERE / "source_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

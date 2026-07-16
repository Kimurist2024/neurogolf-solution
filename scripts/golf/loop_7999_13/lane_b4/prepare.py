#!/usr/bin/env python3
"""Extract exact B4 baselines without touching root artifacts."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASELINE = ROOT / "submission_base_7999.13.zip"
TASKS = (107, 156, 251, 275, 310, 328, 333)


def main() -> None:
    records = []
    with zipfile.ZipFile(BASELINE) as archive:
        names = archive.namelist()
        for task in TASKS:
            member = f"task{task:03d}.onnx"
            matches = [name for name in names if Path(name).name == member]
            if len(matches) != 1:
                raise RuntimeError(f"{member}: expected one member, got {matches}")
            payload = archive.read(matches[0])
            output = HERE / f"baseline_{member}"
            output.write_bytes(payload)
            records.append(
                {
                    "task": task,
                    "member": matches[0],
                    "member_index": names.index(matches[0]),
                    "path": str(output.relative_to(ROOT)),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    report = {
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "members": records,
    }
    (HERE / "baseline_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

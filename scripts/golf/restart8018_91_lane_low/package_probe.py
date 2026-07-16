#!/usr/bin/env python3
"""Package the strict task175@131 candidate over the immutable 8018.91 base."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8018.91.zip"
CANDIDATE = HERE / "candidates" / "task175_gauge_s_factor_reuse.onnx"
OUTPUT = HERE / "submission_PROBE_task175_cost131.zip"
BASE_SHA256 = "e43865760ec8807fbb217fba718226ca6b86d9128b911479214e3252b9f9e091"
CANDIDATE_SHA256 = "22fe38f6428dbc2f98b7135825325044f1898a7da23e2bea9b7584d97bfe4265"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha256(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("8018.91 base drift")
    candidate = CANDIDATE.read_bytes()
    if sha256(candidate) != CANDIDATE_SHA256:
        raise RuntimeError("task175 candidate drift")
    changed = []
    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "w") as target:
        names = source.namelist()
        if len(names) != 400 or "task175.onnx" not in names:
            raise RuntimeError("unexpected authority member set")
        for info in source.infolist():
            data = candidate if info.filename == "task175.onnx" else source.read(info.filename)
            target.writestr(info, data)
            if info.filename == "task175.onnx":
                changed.append(info.filename)
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        names = archive.namelist()
        if len(names) != 400 or sha256(archive.read("task175.onnx")) != CANDIDATE_SHA256:
            raise RuntimeError("probe verification failed")
        with zipfile.ZipFile(BASE, "r") as source:
            unexpected = [name for name in names
                          if name != "task175.onnx" and archive.read(name) != source.read(name)]
    if unexpected:
        raise RuntimeError(f"unexpected changed members: {unexpected}")
    result = {
        "base": str(BASE.relative_to(ROOT)), "base_sha256": BASE_SHA256,
        "output": str(OUTPUT.relative_to(ROOT)), "output_sha256": sha256(OUTPUT.read_bytes()),
        "member_count": len(names), "changed_members": changed,
        "candidate_sha256": CANDIDATE_SHA256,
    }
    (HERE / "probe_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Package rebased strict and private-zero probes without touching root files."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8019.75.zip"
TASK175 = ROOT / "scripts/golf/restart8018_91_lane_low/candidates/task175_gauge_s_factor_reuse.onnx"
TASK328 = HERE / "task328_scales/task328_r01_scale2p40.onnx"
EXPECTED_BASE_SHA = "e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3"


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def package(name: str, replacements: dict[str, Path]) -> dict[str, object]:
    output = HERE / name
    with zipfile.ZipFile(BASE, "r") as source:
        names = source.namelist()
        if len(names) != 400 or len(set(names)) != 400:
            raise RuntimeError("authority archive is not a unique 400-member submission")
        blobs = {member: source.read(member) for member in names}
    before = {member: sha256(blob) for member, blob in blobs.items()}
    for member, path in replacements.items():
        if member not in blobs:
            raise RuntimeError(f"missing authority member: {member}")
        blobs[member] = path.read_bytes()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in names:
            archive.writestr(member, blobs[member])
    with zipfile.ZipFile(output, "r") as check:
        if check.namelist() != names:
            raise RuntimeError("packaged member ordering drift")
        after_blobs = {member: check.read(member) for member in names}
    changed = [member for member in names if before[member] != sha256(after_blobs[member])]
    if changed != list(replacements):
        raise RuntimeError(f"unexpected changed members: {changed}")
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256(output.read_bytes()),
        "members": len(names),
        "changed": changed,
        "replacement_sha256": {member: sha256(path.read_bytes())
                                for member, path in replacements.items()},
    }


def main() -> int:
    if sha256(BASE.read_bytes()) != EXPECTED_BASE_SHA:
        raise RuntimeError("8019.75 authority archive drift")
    result = {
        "authority": str(BASE.relative_to(ROOT)),
        "authority_sha256": EXPECTED_BASE_SHA,
        "strict_task175": package(
            "submission_STRICT_task175_cost131.zip", {"task175.onnx": TASK175}
        ),
        "private_zero_task328": package(
            "submission_PROBE_PRIVATEZERO_task328_cost352.zip", {"task328.onnx": TASK328}
        ),
        "combined_probe": package(
            "submission_PROBE_task175_131_task328_352.zip",
            {"task175.onnx": TASK175, "task328.onnx": TASK328},
        ),
        "projected_gains": {
            "task175_140_to_131": math.log(140 / 131),
            "task328_427_to_352": math.log(427 / 352),
            "combined": math.log(140 / 131) + math.log(427 / 352),
        },
        "admission": {
            "task175": "STRICT",
            "task328": "PRIVATE_ZERO_PROBE_ONLY_NOT_STRICT_MARGIN",
        },
    }
    (HERE / "probe_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

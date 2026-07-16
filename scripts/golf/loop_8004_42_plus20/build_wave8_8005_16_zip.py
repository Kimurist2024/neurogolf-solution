#!/usr/bin/env python3
"""Rebase the admitted task009/task036 replacements onto LB 8005.16."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8005.16.zip"
OUTPUT = HERE / "submission_8005.16_wave8_task009_036_safe_meta.zip"
REPLACEMENTS = {
    "task009.onnx": HERE / "agent_near95_wave2/candidates/task009_b265f7f83d8f_cost2586.onnx",
    "task036.onnx": HERE / "agent_rebuild_mid8/candidates/task036_truthful_gather.onnx",
}
EXPECTED_BASE_SHA256 = "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if digest(BASE.read_bytes()) != EXPECTED_BASE_SHA256:
        raise RuntimeError("submission_base_8005.16.zip changed after rebase audit")
    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "w") as target:
        target.comment = source.comment
        for info in source.infolist():
            data = REPLACEMENTS[info.filename].read_bytes() if info.filename in REPLACEMENTS else source.read(info.filename)
            target.writestr(info, data)

    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "r") as built:
        if source.namelist() != built.namelist() or source.comment != built.comment:
            raise RuntimeError("archive member order/comment changed")
        changed: list[str] = []
        for source_info, built_info in zip(source.infolist(), built.infolist(), strict=True):
            if source_info.filename != built_info.filename:
                raise RuntimeError("archive order changed")
            if source.read(source_info.filename) != built.read(built_info.filename):
                changed.append(source_info.filename)
            for attr in ("date_time", "compress_type", "external_attr", "internal_attr", "flag_bits", "extra", "comment"):
                if getattr(source_info, attr) != getattr(built_info, attr):
                    raise RuntimeError(f"metadata changed for {source_info.filename}: {attr}")
        if changed != list(REPLACEMENTS) or built.testzip() is not None:
            raise RuntimeError(f"unexpected changed/corrupt members: {changed}")

    gains = {"9": 0.01268028517590921, "36": 0.033738139631850204}
    projected = 8005.16 + sum(gains.values())
    manifest = {
        "status": "ADMITTED_PENDING_LB",
        "base": str(BASE.relative_to(ROOT)),
        "base_lb_verified_score": 8005.16,
        "base_sha256": digest(BASE.read_bytes()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": digest(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": {name: digest(path.read_bytes()) for name, path in REPLACEMENTS.items()},
        "task_gains": gains,
        "projected_score_from_8005_16": projected,
        "projected_gain_from_8004_42": projected - 8004.42,
        "remaining_to_8024_42": 8024.42 - projected,
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave8_8005_16_zip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

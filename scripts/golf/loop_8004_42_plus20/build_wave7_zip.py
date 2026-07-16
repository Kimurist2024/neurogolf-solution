#!/usr/bin/env python3
"""Build the metadata-preserving admitted task009/task036 ZIP."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8004.50.zip"
OUTPUT = HERE / "submission_8004.50_wave7_task009_036_safe_meta.zip"
REPLACEMENTS = {
    "task009.onnx": HERE / "agent_near95_wave2/candidates/task009_b265f7f83d8f_cost2586.onnx",
    "task036.onnx": HERE / "agent_rebuild_mid8/candidates/task036_truthful_gather.onnx",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "w") as target:
        target.comment = source.comment
        for info in source.infolist():
            data = REPLACEMENTS[info.filename].read_bytes() if info.filename in REPLACEMENTS else source.read(info.filename)
            target.writestr(info, data)

    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "r") as built:
        assert source.namelist() == built.namelist()
        assert source.comment == built.comment
        changed = []
        for source_info, built_info in zip(source.infolist(), built.infolist(), strict=True):
            assert source_info.filename == built_info.filename
            if source.read(source_info.filename) != built.read(built_info.filename):
                changed.append(source_info.filename)
            for attr in ("date_time", "compress_type", "external_attr", "internal_attr", "flag_bits", "extra", "comment"):
                assert getattr(source_info, attr) == getattr(built_info, attr)
        assert changed == list(REPLACEMENTS)
        assert built.testzip() is None

    task009_gain = 0.01268028517590921
    task036_gain = 0.033738139631850204
    projected_score = 8004.50 + task009_gain + task036_gain
    manifest = {
        "status": "ADMITTED_PENDING_LB",
        "base": str(BASE.relative_to(ROOT)),
        "base_sha256": digest(BASE.read_bytes()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": digest(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": {name: digest(path.read_bytes()) for name, path in REPLACEMENTS.items()},
        "task_gains": {"9": task009_gain, "36": task036_gain},
        "projected_score_from_8004_50": projected_score,
        "projected_gain_from_8004_42": projected_score - 8004.42,
        "remaining_to_8024_42": 8024.42 - projected_score,
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave7_zip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

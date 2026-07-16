#!/usr/bin/env python3
"""Build the metadata-preserving task009/task343 evidence ZIP.

task343 was later identified in the private-zero task catalog and therefore
does not satisfy the strengthened 100%-fresh guarantee.  The archive produced
here is evidence-only and must not be promoted.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8004.50.zip"
OUTPUT = HERE / "submission_8004.50_wave5_task009_343_policy90_meta.zip"
REPLACEMENTS = {
    "task009.onnx": HERE / "agent_near95_wave2/candidates/task009_b265f7f83d8f_cost2586.onnx",
    "task343.onnx": ROOT / "scripts/golf/loop_7999_13/lane_c39/candidate/task343.onnx",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "w") as target:
        target.comment = source.comment
        source_infos = source.infolist()
        for info in source_infos:
            data = REPLACEMENTS[info.filename].read_bytes() if info.filename in REPLACEMENTS else source.read(info.filename)
            target.writestr(info, data)

    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUTPUT, "r") as built:
        assert source.namelist() == built.namelist()
        assert source.comment == built.comment
        changed = []
        for source_info, built_info in zip(source.infolist(), built.infolist(), strict=True):
            assert source_info.filename == built_info.filename
            old = source.read(source_info.filename)
            new = built.read(built_info.filename)
            if old != new:
                changed.append(source_info.filename)
            assert source_info.date_time == built_info.date_time
            assert source_info.compress_type == built_info.compress_type
            assert source_info.external_attr == built_info.external_attr
            assert source_info.internal_attr == built_info.internal_attr
            assert source_info.flag_bits == built_info.flag_bits
            assert source_info.extra == built_info.extra
            assert source_info.comment == built_info.comment
        assert changed == list(REPLACEMENTS)
        assert built.testzip() is None

    manifest = {
        "status": "EVIDENCE_ONLY_PRIVATE_CATALOG_REJECT",
        "promotion_allowed": False,
        "reason": "task343 is private-zero catalog and independent fresh is below 100%",
        "base": str(BASE.relative_to(ROOT)),
        "base_sha256": digest(BASE.read_bytes()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": digest(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": {
            name: digest(path.read_bytes()) for name, path in REPLACEMENTS.items()
        },
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave5_zip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

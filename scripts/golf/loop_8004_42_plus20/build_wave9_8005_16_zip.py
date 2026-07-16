#!/usr/bin/env python3
"""Build metadata-safe task009/task036/task226 promotion ZIP on LB 8005.16."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8005.16.zip"
OUTPUT = HERE / "submission_8005.16_wave9_task009_036_226_safe_meta.zip"
BASE_SHA256 = "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00"
REPLACEMENTS = {
    "task009.onnx": HERE / "agent_near95_wave2/candidates/task009_b265f7f83d8f_cost2586.onnx",
    "task036.onnx": HERE / "agent_rebuild_mid8/candidates/task036_truthful_gather.onnx",
    "task226.onnx": HERE / "agent_target_mid19/task226_sixbit.onnx",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("immutable 8005.16 base SHA mismatch")
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT, "w") as target:
        target.comment = source.comment
        for info in source.infolist():
            data = REPLACEMENTS[info.filename].read_bytes() if info.filename in REPLACEMENTS else source.read(info.filename)
            target.writestr(info, data)

    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT) as built:
        if source.namelist() != built.namelist() or source.comment != built.comment:
            raise RuntimeError("archive order/comment mismatch")
        changed: list[str] = []
        for old_info, new_info in zip(source.infolist(), built.infolist(), strict=True):
            if old_info.filename != new_info.filename:
                raise RuntimeError("archive member order mismatch")
            if source.read(old_info.filename) != built.read(new_info.filename):
                changed.append(old_info.filename)
            for attr in ("date_time", "compress_type", "external_attr", "internal_attr", "flag_bits", "extra", "comment"):
                if getattr(old_info, attr) != getattr(new_info, attr):
                    raise RuntimeError(f"metadata mismatch: {old_info.filename}/{attr}")
        if changed != list(REPLACEMENTS) or built.testzip() is not None:
            raise RuntimeError(f"changed/corrupt member mismatch: {changed}")

    gains = {
        "9": math.log(2619 / 2586),
        "36": math.log(1477 / 1428),
        "226": math.log(375 / 372),
    }
    projected = 8005.16 + sum(gains.values())
    manifest = {
        "status": "ADMITTED_PENDING_LB",
        "base": str(BASE.relative_to(ROOT)),
        "base_lb_verified_score": 8005.16,
        "base_sha256": sha(BASE.read_bytes()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": {name: sha(path.read_bytes()) for name, path in REPLACEMENTS.items()},
        "task_costs": {"9": [2619, 2586], "36": [1477, 1428], "226": [375, 372]},
        "task_gains": gains,
        "projected_score_from_8005_16": projected,
        "projected_gain_from_8004_42": projected - 8004.42,
        "remaining_to_8024_42": 8024.42 - projected,
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave9_8005_16_zip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

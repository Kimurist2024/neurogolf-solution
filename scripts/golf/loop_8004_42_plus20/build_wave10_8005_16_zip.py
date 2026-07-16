#!/usr/bin/env python3
"""Build metadata-safe five-task promotion ZIP on immutable LB 8005.16."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8005.16.zip"
OUTPUT = HERE / "submission_8005.16_wave10_task009_036_158_226_323_safe_meta.zip"
BASE_SHA256 = "73073208ec8b5b296f1fc13300a7e4df871135f0a9edd4a682ac7b48077aba00"
REPLACEMENTS = {
    "task009.onnx": HERE / "agent_near95_wave2/candidates/task009_b265f7f83d8f_cost2586.onnx",
    "task036.onnx": HERE / "agent_rebuild_mid8/candidates/task036_truthful_gather.onnx",
    "task158.onnx": HERE / "agent_task158_deep46/sound/task158_scatter_max_orientation_only.onnx",
    "task226.onnx": HERE / "agent_target_mid19/task226_sixbit.onnx",
    "task323.onnx": HERE / "root_high58/task323_cost104_robust_u10.onnx",
}
EXPECTED_REPLACEMENT_SHA256 = {
    "task009.onnx": "b265f7f83d8fbf66c9388b9edfe0111d2b77a4b610377a3994a9c483fb445d28",
    "task036.onnx": "fc83bef42ce52ddd5c726323bacca5c4bf59ecaa55ef2aa55b1571243e9b5738",
    "task158.onnx": "3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba",
    "task226.onnx": "852b6091385d97df6899e21304bf194440fb5cd3343385693093c24be0cb8203",
    "task323.onnx": "db773b15ceea8c42fac7543b7b7e93e0fd56a73493c7a8122b587327544c5926",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("immutable 8005.16 base SHA mismatch")
    actual_replacement_sha = {
        name: sha(path.read_bytes()) for name, path in REPLACEMENTS.items()
    }
    if actual_replacement_sha != EXPECTED_REPLACEMENT_SHA256:
        raise RuntimeError(f"replacement SHA mismatch: {actual_replacement_sha}")

    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT, "w") as target:
        target.comment = source.comment
        for info in source.infolist():
            data = (
                REPLACEMENTS[info.filename].read_bytes()
                if info.filename in REPLACEMENTS
                else source.read(info.filename)
            )
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
            for attr in (
                "date_time",
                "compress_type",
                "external_attr",
                "internal_attr",
                "flag_bits",
                "extra",
                "comment",
            ):
                if getattr(old_info, attr) != getattr(new_info, attr):
                    raise RuntimeError(f"metadata mismatch: {old_info.filename}/{attr}")
        if changed != list(REPLACEMENTS) or built.testzip() is not None:
            raise RuntimeError(f"changed/corrupt member mismatch: {changed}")

    costs = {
        "9": [2619, 2586],
        "36": [1477, 1428],
        "158": [7615, 7612],
        "226": [375, 372],
        "323": [106, 104],
    }
    gains = {task: math.log(before / after) for task, (before, after) in costs.items()}
    projected = 8005.16 + sum(gains.values())
    manifest = {
        "status": "ADMITTED_PENDING_LB",
        "base": str(BASE.relative_to(ROOT)),
        "base_lb_verified_score": 8005.16,
        "base_sha256": sha(BASE.read_bytes()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": actual_replacement_sha,
        "task_costs": costs,
        "task_gains": gains,
        "projected_score_from_8005_16": projected,
        "projected_gain_from_8004_42": projected - 8004.42,
        "remaining_to_8024_42": 8024.42 - projected,
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave10_8005_16_zip_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

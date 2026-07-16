#!/usr/bin/env python3
"""Build the metadata-safe five-task promotion ZIP on immutable LB 8005.17."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "submission_base_8005.17.zip"
OUTPUT = HERE / "submission_8005.17_wave14_task013_158_254_267_323_safe_meta.zip"
BASE_SHA256 = "c48fa65401a5bd26d3ed1c556eee8f85c0a2063db313be6b96c73e86159b0a04"
REPLACEMENTS = {
    "task013.onnx": HERE / "root_sweep29/reuse_contract/task013_r001.onnx",
    "task158.onnx": HERE / "agent_task158_deep46/sound/task158_scatter_max_orientation_only.onnx",
    "task254.onnx": ROOT / "scripts/golf/loop_7999_13/lane_b12/candidates/task254_r01_static42.onnx",
    "task267.onnx": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task267_r02_static30.onnx",
    "task323.onnx": HERE / "root_high58/task323_cost104_robust_u10.onnx",
}
EXPECTED_REPLACEMENT_SHA256 = {
    "task013.onnx": "ad4eb35978f3e38d1d3e2afdd55e55db871962cc2ea4c989675d9d583434103b",
    "task158.onnx": "3bfa73410f489f0bc444a1f4567f95837e445cd940d10b7282bdc50a95dd2dba",
    "task254.onnx": "814ece451a8f8eda8e9221d58e2f4fb3359fa396dfe971f6ad97693f453b15f8",
    "task267.onnx": "4ca7f921c34f87ef71512a8b680de7c984a2b42cd55b338b57aaabc012321387",
    "task323.onnx": "db773b15ceea8c42fac7543b7b7e93e0fd56a73493c7a8122b587327544c5926",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha(BASE.read_bytes()) != BASE_SHA256:
        raise RuntimeError("immutable 8005.17 base SHA mismatch")
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
                "date_time", "compress_type", "external_attr", "internal_attr",
                "flag_bits", "extra", "comment",
            ):
                if getattr(old_info, attr) != getattr(new_info, attr):
                    raise RuntimeError(f"metadata mismatch: {old_info.filename}/{attr}")
        if changed != list(REPLACEMENTS) or built.testzip() is not None:
            raise RuntimeError(f"changed/corrupt member mismatch: {changed}")

    costs = {
        "013": [638, 636],
        "158": [7615, 7612],
        "254": [76, 42],
        "267": [60, 30],
        "323": [106, 104],
    }
    gains = {task: math.log(before / after) for task, (before, after) in costs.items()}
    projected = 8005.17 + sum(gains.values())
    manifest = {
        "status": "ADMITTED_PENDING_LB",
        "base": str(BASE.relative_to(ROOT)),
        "base_lb_verified_score": 8005.17,
        "base_sha256": sha(BASE.read_bytes()),
        "base_already_contains": ["task226 cost372 LB-white"],
        "explicit_exclusions": {
            "task009": "LB-black lineage recorded in best_score.json",
            "task036": "candidate cost1428 regresses from actual baseline cost325",
        },
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha(OUTPUT.read_bytes()),
        "changed_members": changed,
        "replacement_sha256": actual_replacement_sha,
        "task_costs": costs,
        "task_gains": gains,
        "projected_score_from_8005_17": projected,
        "projected_gain_from_8004_42": projected - 8004.42,
        "remaining_to_8024_42": 8024.42 - projected,
        "archive_integrity": "PASS",
        "metadata_parity": "PASS",
    }
    (HERE / "wave14_8005_17_zip_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

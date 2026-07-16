#!/usr/bin/env python3
"""Prove that all 20 target members are byte-identical across the LB rebase."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "submission_base_8005.17.zip"
NEW = ROOT / "submission_base_8006.61.zip"
TARGETS = (68, 175, 400, 30, 224, 281, 240, 183, 376, 59, 358, 20, 190, 302, 195, 300, 383, 193, 304, 384)
FIXED_CHAMPIONS = {13, 70, 158, 254, 267, 323, 379}
LB_PROBE_REQUIRED_BLACK = {18, 48, 112, 134, 168, 198, 233, 251, 277, 286, 365, 366}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    rows = []
    with zipfile.ZipFile(OLD) as old_zip, zipfile.ZipFile(NEW) as new_zip:
        for task in TARGETS:
            member = f"task{task:03d}.onnx"
            old_data = old_zip.read(member)
            new_data = new_zip.read(member)
            rows.append(
                {
                    "task": task,
                    "member": member,
                    "old_sha256": digest(old_data),
                    "new_sha256": digest(new_data),
                    "old_bytes": len(old_data),
                    "new_bytes": len(new_data),
                    "byte_identical": old_data == new_data,
                }
            )
    report = {
        "old_authority": str(OLD.relative_to(ROOT)),
        "old_authority_sha256": digest(OLD.read_bytes()),
        "new_authority": str(NEW.relative_to(ROOT)),
        "new_authority_sha256": digest(NEW.read_bytes()),
        "expected_new_authority_sha256": "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118",
        "new_authority_sha_matches": digest(NEW.read_bytes()) == "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118",
        "target_count": len(TARGETS),
        "all_target_members_byte_identical": all(row["byte_identical"] for row in rows),
        "fixed_champions": sorted(FIXED_CHAMPIONS),
        "target_intersection_fixed_champions": sorted(set(TARGETS) & FIXED_CHAMPIONS),
        "lb_probe_required_black": sorted(LB_PROBE_REQUIRED_BLACK),
        "target_intersection_lb_probe_required_black": sorted(set(TARGETS) & LB_PROBE_REQUIRED_BLACK),
        "rows": rows,
    }
    (HERE / "authority_rebase_proof.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["new_authority_sha_matches"] or not report["all_target_members_byte_identical"]:
        raise SystemExit("authority rebase proof failed")
    print(json.dumps({key: report[key] for key in ("new_authority_sha_matches", "target_count", "all_target_members_byte_identical", "target_intersection_fixed_champions", "target_intersection_lb_probe_required_black")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

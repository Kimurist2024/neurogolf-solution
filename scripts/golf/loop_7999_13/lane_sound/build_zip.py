#!/usr/bin/env python3
"""Build the lane A ZIP while preserving every baseline ZIP detail."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
BASELINE = ROOT / "submission_base_7999.13.zip"
OUTPUT = LANE / "submission_7999.13_sound168_344.zip"
EXPECTED_BASELINE_SHA256 = (
    "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"
)
REPLACEMENTS = {
    "task168.onnx": LANE / "task168_sound_cost416.onnx",
    "task344.onnx": LANE / "task344_sound_cost197.onnx",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(info: zipfile.ZipInfo) -> tuple[object, ...]:
    return (
        info.filename,
        info.date_time,
        info.compress_type,
        info.comment,
        info.extra,
        info.create_system,
        info.create_version,
        info.extract_version,
        info.reserved,
        info.flag_bits,
        info.volume,
        info.internal_attr,
        info.external_attr,
    )


def main() -> None:
    baseline_sha = sha256(BASELINE)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"baseline hash mismatch: expected {EXPECTED_BASELINE_SHA256}, got {baseline_sha}"
        )
    for replacement in REPLACEMENTS.values():
        if not replacement.is_file():
            raise FileNotFoundError(replacement)

    with zipfile.ZipFile(BASELINE, "r") as src, zipfile.ZipFile(OUTPUT, "w") as dst:
        src_infos = src.infolist()
        src_names = [info.filename for info in src_infos]
        missing = sorted(set(REPLACEMENTS) - set(src_names))
        if missing:
            raise RuntimeError(f"replacement members missing from baseline: {missing}")
        dst.comment = src.comment
        for info in src_infos:
            data = (
                REPLACEMENTS[info.filename].read_bytes()
                if info.filename in REPLACEMENTS
                else src.read(info.filename)
            )
            dst.writestr(copy.copy(info), data)

    with zipfile.ZipFile(BASELINE, "r") as src, zipfile.ZipFile(OUTPUT, "r") as dst:
        src_infos = src.infolist()
        dst_infos = dst.infolist()
        if src.comment != dst.comment:
            raise RuntimeError("ZIP comment changed")
        if [metadata(i) for i in src_infos] != [metadata(i) for i in dst_infos]:
            raise RuntimeError("ZIP order or member metadata changed")

        changed: list[str] = []
        for src_info, dst_info in zip(src_infos, dst_infos, strict=True):
            src_data = src.read(src_info.filename)
            dst_data = dst.read(dst_info.filename)
            if src_data != dst_data:
                changed.append(src_info.filename)
        if changed != list(REPLACEMENTS):
            raise RuntimeError(
                f"unexpected changed entries: expected {list(REPLACEMENTS)}, got {changed}"
            )

        audit = {
            "baseline": str(BASELINE.relative_to(ROOT)),
            "baseline_sha256": baseline_sha,
            "output": str(OUTPUT.relative_to(ROOT)),
            "output_sha256": sha256(OUTPUT),
            "member_count": len(src_infos),
            "order_preserved": True,
            "metadata_preserved": True,
            "zip_comment_preserved": True,
            "changed_members": changed,
            "replacements": {
                member: {
                    "source": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for member, path in REPLACEMENTS.items()
            },
        }

    (LANE / "zip_build_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

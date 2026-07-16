#!/usr/bin/env python3
"""Build a replacement ZIP while preserving baseline order and ZipInfo metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import zipfile
from pathlib import Path


BASELINE_SHA256 = "a123cdc6cb04baa179044f8910d90c6b753dbfed22db9e69738a0574f276a2e1"


def digest(path: Path) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--replace", action="append", required=True)
    parser.add_argument(
        "--expected-baseline-sha256",
        default=BASELINE_SHA256,
        help="Pinned SHA-256 of the aggregate used as this wave's baseline.",
    )
    args = parser.parse_args()
    actual_baseline_sha256 = digest(args.baseline)
    if actual_baseline_sha256 != args.expected_baseline_sha256:
        raise RuntimeError(
            "baseline SHA-256 mismatch: "
            f"{actual_baseline_sha256} != {args.expected_baseline_sha256}"
        )
    replacements: dict[str, Path] = {}
    for item in args.replace:
        task_text, path_text = item.split("=", 1)
        task = int(task_text)
        member = f"task{task:03d}.onnx"
        path = Path(path_text)
        if member in replacements:
            raise ValueError(f"duplicate replacement {member}")
        if not path.is_file():
            raise FileNotFoundError(path)
        replacements[member] = path
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.baseline) as source, zipfile.ZipFile(args.output, "w") as target:
        source_infos = source.infolist()
        source_names = [info.filename for info in source_infos]
        missing = sorted(set(replacements) - set(source_names))
        if missing:
            raise ValueError(f"members absent from baseline: {missing}")
        target.comment = source.comment
        for info in source_infos:
            data = replacements[info.filename].read_bytes() if info.filename in replacements else source.read(info.filename)
            target.writestr(copy.copy(info), data)

    with zipfile.ZipFile(args.baseline) as source, zipfile.ZipFile(args.output) as target:
        source_infos = source.infolist()
        target_infos = target.infolist()
        if source.comment != target.comment:
            raise RuntimeError("ZIP comment changed")
        if [metadata(info) for info in source_infos] != [metadata(info) for info in target_infos]:
            raise RuntimeError("order or ZipInfo metadata changed")
        changed = [
            left.filename
            for left, right in zip(source_infos, target_infos, strict=True)
            if source.read(left.filename) != target.read(right.filename)
        ]
        expected = [info.filename for info in source_infos if info.filename in replacements]
        if changed != expected:
            raise RuntimeError(f"changed members {changed} != {expected}")
    payload = {
        "baseline": str(args.baseline),
        "baseline_sha256": actual_baseline_sha256,
        "output": str(args.output),
        "output_sha256": digest(args.output),
        "member_count": len(source_infos),
        "order_preserved": True,
        "metadata_preserved": True,
        "comment_preserved": True,
        "changed_members": changed,
        "replacements": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in replacements.items()
        },
    }
    args.audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

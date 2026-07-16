#!/usr/bin/env python3
"""Build a metadata-preserving one-member task192 probe on 8008.14."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT / "submission_base_8008.14.zip"
CANDIDATE = HERE / "task192_selected_masks.onnx"
OUT = ROOT / "scripts/golf/loop_8004_42_plus20/lb_probe_queue_8008_14/p07_task192_sound_cost1197.zip"
BASE_SHA = "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6"
CANDIDATE_SHA = "40244ab462644481407ebb7200984dfdff1475c0d8e6ff731ba2d588ec92ea09"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if sha(BASE.read_bytes()) != BASE_SHA:
        raise RuntimeError("authority ZIP hash changed")
    candidate = CANDIDATE.read_bytes()
    if sha(candidate) != CANDIDATE_SHA:
        raise RuntimeError("candidate hash changed")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BASE, "r") as source, zipfile.ZipFile(OUT, "w") as target:
        target.comment = source.comment
        names = source.namelist()
        if len(names) != 400 or len(set(names)) != 400 or "task192.onnx" not in names:
            raise RuntimeError("unexpected authority archive")
        for info in source.infolist():
            payload = candidate if info.filename == "task192.onnx" else source.read(info.filename)
            target.writestr(info, payload)

    with zipfile.ZipFile(BASE) as base, zipfile.ZipFile(OUT) as probe:
        if probe.namelist() != base.namelist() or probe.comment != base.comment:
            raise RuntimeError("order/comment mismatch")
        changed = []
        for name in base.namelist():
            left, right = base.read(name), probe.read(name)
            if left != right:
                changed.append(name)
            if probe.getinfo(name).CRC != zipfile.crc32(right):
                raise RuntimeError(f"CRC mismatch: {name}")
        if changed != ["task192.onnx"]:
            raise RuntimeError(f"unexpected changed members: {changed}")
    result = {
        "path": str(OUT.relative_to(ROOT)),
        "sha256": sha(OUT.read_bytes()),
        "md5": hashlib.md5(OUT.read_bytes()).hexdigest(),  # noqa: S324 - archive identity only
        "members": 400,
        "changed_members": ["task192.onnx"],
        "candidate_sha256": CANDIDATE_SHA,
        "authority_cost": 1609,
        "candidate_cost": 1197,
        "projected_gain": 0.29579444143441,
    }
    (HERE / "probe_zip.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

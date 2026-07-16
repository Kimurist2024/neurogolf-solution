#!/usr/bin/env python3
"""Build a task200-only probe from immutable 8023.08 authority."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8023.08.zip"
BASE_SHA256 = "0e29e8d57f7ac58136a9574351c9c6f3056f9debf6eeee9c181c8f2e9fac690a"
CANDIDATE = HERE / "focus/candidates/task200_POLICY90_cost342_c659ae401e4c.onnx"
OUTPUT = HERE / "submission_PROBE_task200_cost342.zip"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def main() -> None:
    if sha_file(BASE) != BASE_SHA256:
        raise RuntimeError("8023.08 authority SHA mismatch")
    protected = ("submission.zip", "all_scores.csv", "best_score.json")
    before = {name: sha_file(ROOT / name) for name in protected}
    replacement = CANDIDATE.read_bytes()
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT, "w") as target:
        infos = source.infolist()
        if sum(info.filename.endswith(".onnx") for info in infos) != 400:
            raise RuntimeError("authority does not contain 400 ONNX members")
        for info in infos:
            data = replacement if info.filename == "task200.onnx" else source.read(info)
            target.writestr(info, data)
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(OUTPUT) as probe:
        changed = [
            name for name in source.namelist()
            if source.read(name) != probe.read(name)
        ]
    if changed != ["task200.onnx"]:
        raise RuntimeError(f"unexpected changed members: {changed}")
    after = {name: sha_file(ROOT / name) for name in protected}
    if before != after:
        raise RuntimeError("protected root files changed")
    manifest = {
        "authority": str(BASE.relative_to(ROOT)),
        "authority_sha256": BASE_SHA256,
        "probe": str(OUTPUT.relative_to(ROOT)),
        "probe_sha256": sha_file(OUTPUT),
        "changed_members": changed,
        "task": 200,
        "authority_cost": 346,
        "candidate_cost": 342,
        "projected_gain": math.log(346 / 342),
        "projected_score_from_8023_08": 8023.08 + math.log(346 / 342),
        "candidate_sha256": sha_bytes(replacement),
        "known_raw_bit_identical": True,
        "root_guards_before": before,
        "root_guards_after": after,
    }
    (HERE / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

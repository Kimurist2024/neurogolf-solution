#!/usr/bin/env python3
"""Build strict task226-only probe from the immutable 8019.75 authority."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8019.75.zip"
CAND = HERE / "candidates/task226_POLICY90_cost368_05ebc8919fdd.onnx"
OUT = HERE / "submission_PROBE_task226_cost368.zip"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    guards_before = {
        name: sha(ROOT / name)
        for name in ("submission.zip", "all_scores.csv", "best_score.json")
    }
    replacement = CAND.read_bytes()
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(
        OUT, "w", zipfile.ZIP_DEFLATED
    ) as target:
        members = [name for name in source.namelist() if name.endswith(".onnx")]
        if len(members) != 400:
            raise RuntimeError(f"authority members={len(members)}")
        for name in members:
            target.writestr(
                name, replacement if name == "task226.onnx" else source.read(name)
            )
    with zipfile.ZipFile(BASE) as base, zipfile.ZipFile(OUT) as probe:
        changed = [
            name for name in base.namelist()
            if name.endswith(".onnx") and base.read(name) != probe.read(name)
        ]
    guards_after = {name: sha(ROOT / name) for name in guards_before}
    if guards_before != guards_after:
        raise RuntimeError("root authority drift during probe build")
    payload = {
        "authority": str(BASE.relative_to(ROOT)),
        "authority_sha256": sha(BASE),
        "probe": str(OUT.relative_to(ROOT)),
        "probe_sha256": sha(OUT),
        "changed_members": changed,
        "task": 226,
        "authority_cost": 369,
        "candidate_cost": 368,
        "score_gain": math.log(369 / 368),
        "candidate_sha256": hashlib.sha256(replacement).hexdigest(),
        "root_guards_before": guards_before,
        "root_guards_after": guards_after,
    }
    (HERE / "probe_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

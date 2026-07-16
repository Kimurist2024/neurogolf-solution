#!/usr/bin/env python3
"""Build evidence-only probes for the strongest exact-rewrite candidates."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "submission_base_8019.75.zip"
BASE_SHA256 = "e69058edd21e27ab7d32670d714ec5cea6d35632a9d9a620364731297717edb3"

CANDIDATES = {
    132: (HERE / "focus/candidates/task132_POLICY90_cost292_9474bab226a4.onnx", 308, 292),
    168: (HERE / "candidates/task168_POLICY90_cost384_213f102b1a57.onnx", 398, 384),
    226: (HERE / "candidates/task226_POLICY90_cost368_05ebc8919fdd.onnx", 369, 368),
    345: (HERE / "focus/candidates/task345_POLICY90_cost369_1b6b180284a6.onnx", 389, 369),
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def build(output: Path, tasks: tuple[int, ...]) -> dict[str, object]:
    replacements = {task: CANDIDATES[task][0].read_bytes() for task in tasks}
    with zipfile.ZipFile(BASE) as source, zipfile.ZipFile(output, "w") as target:
        infos = source.infolist()
        members = [info.filename for info in infos if info.filename.endswith(".onnx")]
        if len(members) != 400:
            raise RuntimeError(f"authority members={len(members)}")
        for info in infos:
            match = next(
                (task for task in tasks if info.filename == f"task{task:03d}.onnx"),
                None,
            )
            target.writestr(info, replacements[match] if match is not None else source.read(info))

    with zipfile.ZipFile(BASE) as authority, zipfile.ZipFile(output) as probe:
        changed = [
            name for name in authority.namelist()
            if authority.read(name) != probe.read(name)
        ]
    expected = [f"task{task:03d}.onnx" for task in tasks]
    if changed != expected:
        raise RuntimeError(f"changed={changed}, expected={expected}")
    gain = sum(math.log(CANDIDATES[task][1] / CANDIDATES[task][2]) for task in tasks)
    return {
        "path": str(output.relative_to(ROOT)),
        "sha256": sha_file(output),
        "tasks": list(tasks),
        "changed_members": changed,
        "projected_gain": gain,
        "projected_score_from_8019_75": 8019.75 + gain,
        "candidates": {
            str(task): {
                "authority_cost": CANDIDATES[task][1],
                "candidate_cost": CANDIDATES[task][2],
                "sha256": sha_bytes(replacements[task]),
            }
            for task in tasks
        },
    }


def main() -> None:
    if sha_file(BASE) != BASE_SHA256:
        raise RuntimeError("8019.75 authority SHA mismatch")
    root_files = ("submission.zip", "all_scores.csv", "best_score.json")
    guards_before = {name: sha_file(ROOT / name) for name in root_files}
    outputs = [
        build(HERE / "submission_PROBE_task132_cost292.zip", (132,)),
        build(HERE / "submission_PROBE_task168_cost384.zip", (168,)),
        build(HERE / "submission_PROBE_task345_cost369.zip", (345,)),
        build(HERE / "submission_EXACT3_tasks168_226_345.zip", (168, 226, 345)),
        build(
            HERE / "submission_STRICT4_tasks132_168_226_345.zip",
            (132, 168, 226, 345),
        ),
    ]
    guards_after = {name: sha_file(ROOT / name) for name in root_files}
    if guards_before != guards_after:
        raise RuntimeError("protected root files changed during probe build")
    manifest = {
        "authority": str(BASE.relative_to(ROOT)),
        "authority_sha256": BASE_SHA256,
        "classification": (
            "task168/task226/task345 are known raw-bit-identical exact rewrites; "
            "task168/task345 reconstruct constant initializers with PRelu+CastLike; "
            "task226 removes redundant Where. task132 is a separate strict-policy "
            "probe and is not raw-bit-identical."
        ),
        "outputs": outputs,
        "root_guards_before": guards_before,
        "root_guards_after": guards_after,
    }
    (HERE / "exact_probe_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

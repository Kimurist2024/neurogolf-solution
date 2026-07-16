#!/usr/bin/env python3
"""Extract the immutable 8004.50 baselines and stage bounded rebuild attempts.

The source list is deliberately explicit: this lane never scans or writes a root
submission.  Models are copied only as diagnostic starting points for truthful
shape repair / policy-90 revalidation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
TASKS = (18, 233, 286, 366)

ATTEMPTS = {
    18: [
        ROOT / "scripts/golf/loop_7999_13/lane_rebuild_c2/candidates/task018.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task018_r01_static4578.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task018_r02_static4682.onnx",
        ROOT / "scripts/golf/scratch_codex/task018/alt_agent/compact_signature_direct.onnx",
        ROOT / "scripts/golf/scratch_codex/task018/tile2x3_k22_allmode_clean.onnx",
    ],
    233: [
        ROOT / "scripts/golf/loop_8004_42/agent_exact_safe/models/task233.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task233_r01_static4936.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task233_r02_static7383.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task233_r03_static7383.onnx",
        ROOT / "scripts/golf/loop_8003_40/agent_exact_resume/candidates/task233_dedup.onnx",
        ROOT / "scripts/golf/scratch_codex/task233/graph_floor/packed_qlinear_hline8.onnx",
    ],
    286: [
        ROOT / "scripts/golf/loop_7999_13/lane_a21/candidates/task286_r01.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a21/candidates/task286_r02.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a21/candidates/task286_r03.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a21/candidates/task286_r04.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a21/sound/task286_spec_fullrow.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_a21/sound/task286_spec_unionfind.onnx",
    ],
    366: [
        ROOT / "scripts/golf/scratch_codex_plus10/wave2_actual/task366.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_b29/task366_cost7646_truthful_metadata.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task366_r01_static5246.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task366_r02_static6242.onnx",
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task366_r03_static6330.onnx",
        ROOT / "scripts/golf/scratch_claude/task366/cc_floor.onnx",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    models = HERE / "models"
    attempts = HERE / "attempts"
    models.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "baseline_zip": str(BASE_ZIP.relative_to(ROOT)),
        "baseline_zip_sha256": digest(BASE_ZIP),
        "baselines": {},
        "attempts": {},
    }
    with zipfile.ZipFile(BASE_ZIP) as archive:
        for task in TASKS:
            out = models / f"baseline_task{task:03d}.onnx"
            out.write_bytes(archive.read(f"task{task:03d}.onnx"))
            report["baselines"][str(task)] = {
                "path": str(out.relative_to(ROOT)),
                "sha256": digest(out),
            }

    for task, paths in ATTEMPTS.items():
        rows = []
        seen: set[str] = set()
        for source in paths:
            if not source.exists():
                rows.append({"source": str(source.relative_to(ROOT)), "missing": True})
                continue
            sha = digest(source)
            if sha in seen:
                continue
            seen.add(sha)
            out = attempts / f"task{task:03d}_a{len(seen):02d}.onnx"
            shutil.copyfile(source, out)
            rows.append({
                "source": str(source.relative_to(ROOT)),
                "path": str(out.relative_to(ROOT)),
                "sha256": sha,
            })
        report["attempts"][str(task)] = rows

    (HERE / "asset_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

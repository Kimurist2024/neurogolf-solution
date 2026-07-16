#!/usr/bin/env python3
"""Atomically add strictly verified NeuroGolf candidates to submission.zip.

The official LB baseline remains immutable.  This module writes a separate
promotion manifest with the projected (not LB-confirmed) gain and keeps the
pre-promotion authority backup under others/71407/auto_promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission.zip"
BASE = ROOT / "submission_base_8023.08.zip"
BASE_SCORE = 8023.08
OUT = ROOT / "others/71407/auto_promotion"
MANIFEST = OUT / "manifest.json"
BACKUP = OUT / "submission_before_auto_promotion_8023.08.zip"
STAGED = OUT / "submission_autostaged_gold.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if MANIFEST.is_file():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing.get("baseline", {}).get("sha256") == digest(BASE):
            return existing
    return {
        "baseline": {
            "score": BASE_SCORE,
            "path": str(BASE.relative_to(ROOT)),
            "sha256": digest(BASE),
        },
        "tasks": {},
        "policy": (
            "official/local gold exact + strict checker/static shapes + stable margin + "
            "fresh 2000x2 at 100%; no runtime/nonfinite/shape/small-positive errors"
        ),
    }


def replace_members(source: Path, destination: Path, replacements: dict[str, bytes]) -> None:
    temporary = destination.with_suffix(".autostage.tmp")
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(temporary, "w") as outgoing:
        names = set(incoming.namelist())
        missing = set(replacements) - names
        if missing:
            raise RuntimeError(f"submission members missing: {sorted(missing)}")
        for info in incoming.infolist():
            data = replacements.get(info.filename, incoming.read(info.filename))
            outgoing.writestr(info, data)
    os.replace(temporary, destination)


def stage_candidates(rows: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    if not BACKUP.exists():
        shutil.copy2(BASE, BACKUP)

    with (ROOT / "all_scores.csv").open(encoding="utf-8") as handle:
        import csv
        authority_costs = {
            int(row["task"].removeprefix("task")): int(row["cost"])
            for row in csv.DictReader(handle)
        }

    accepted: list[dict] = []
    skipped: list[dict] = []
    for raw in rows:
        task = int(raw["task"])
        path = Path(raw["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_sha = digest(path)
        expected_sha = str(raw["sha256"])
        if observed_sha != expected_sha:
            raise RuntimeError(f"task{task:03d} SHA mismatch: {observed_sha} != {expected_sha}")
        if raw.get("strict_gate") is not True:
            raise RuntimeError(f"task{task:03d} lacks strict_gate=true")
        authority_cost = int(authority_costs.get(task, raw["authority_cost"]))
        candidate_cost = int(raw["candidate_cost"])
        if candidate_cost >= authority_cost:
            skipped.append({
                "task": task,
                "reason": "not_cheaper_than_current_authority",
                "authority_cost": authority_cost,
                "candidate_cost": candidate_cost,
            })
            continue
        previous = manifest["tasks"].get(str(task))
        if previous and int(previous["candidate_cost"]) <= candidate_cost:
            continue
        row = {
            "task": task,
            "path": str(path.relative_to(ROOT)),
            "sha256": observed_sha,
            "authority_cost": authority_cost,
            "candidate_cost": candidate_cost,
            "gain": math.log(authority_cost / candidate_cost),
            "strict_gate": True,
            "evidence": raw.get("evidence"),
            "staged_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest["tasks"][str(task)] = row
        accepted.append(row)

    # Rebuild from the immutable LB baseline on every invocation.  Using the
    # mutable submission as the source can silently retain changes made by a
    # different session between admissions.
    replacements: dict[str, bytes] = {}
    for task_text, row in manifest["tasks"].items():
        task = int(task_text)
        path = ROOT / row["path"]
        observed_sha = digest(path)
        if observed_sha != row["sha256"]:
            raise RuntimeError(
                f"staged task{task:03d} SHA mismatch: {observed_sha} != {row['sha256']}"
            )
        replacements[f"task{task:03d}.onnx"] = path.read_bytes()
    replace_members(BASE, SUBMISSION, replacements)
    shutil.copy2(SUBMISSION, STAGED)
    gain = sum(float(row["gain"]) for row in manifest["tasks"].values())
    manifest["projected_gain"] = gain
    manifest["projected_score"] = BASE_SCORE + gain
    manifest["submission"] = {
        "path": str(SUBMISSION.relative_to(ROOT)),
        "sha256": digest(SUBMISSION),
        "staged_copy": str(STAGED.relative_to(ROOT)),
    }
    manifest["last_accepted"] = accepted
    manifest["last_skipped"] = skipped
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def current_verified_rows() -> list[dict]:
    return [
        {
            "task": 175,
            "path": "scripts/golf/restart8018_91_lane_low/candidates/task175_gauge_s_factor_reuse.onnx",
            "sha256": "22fe38f6428dbc2f98b7135825325044f1898a7da23e2bea9b7584d97bfe4265",
            "authority_cost": 140,
            "candidate_cost": 131,
            "strict_gate": True,
            "evidence": "scripts/golf/restart8018_91_lane_low/task175_cost131_evidence.json",
        },
        {
            "task": 275,
            "path": "scripts/golf/focus_task275_gold/task275_diagonal_reuse_cost419_c7ddaab77f6d.onnx",
            "sha256": "c7ddaab77f6da011a99d233775ab02964f1a5e714f4dbb02045d1ecdda57c8e2",
            "authority_cost": 428,
            "candidate_cost": 419,
            "strict_gate": True,
            "evidence": "scripts/golf/focus_task275_gold/final_evidence.json",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-verified", action="store_true")
    args = parser.parse_args()
    if not args.current_verified:
        raise SystemExit("pass --current-verified or import stage_candidates()")
    manifest = stage_candidates(current_verified_rows())
    print(json.dumps({
        "tasks": sorted(int(task) for task in manifest["tasks"]),
        "projected_gain": manifest["projected_gain"],
        "projected_score": manifest["projected_score"],
        "submission_sha256": manifest["submission"]["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

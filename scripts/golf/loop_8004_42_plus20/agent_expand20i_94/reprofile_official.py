#!/usr/bin/env python3
"""Repeat competition scorer profiles with independent labels/tempdirs."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TASKS = (62, 170, 245, 308, 338)

sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
from harvest import known_score  # noqa: E402
from screen_all import resolve_source  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority hash mismatch")
    delta = json.loads((HERE / "inventory_delta.json").read_text())
    candidate_rows = {int(row["task"]): row for row in delta["new_rows"]}
    output = {"authority_zip_sha256": AUTHORITY_SHA256, "repeat_count": 3, "tasks": {}}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            authority = archive.read(f"task{task:03d}.onnx")
            row = candidate_rows[task]
            candidate = None
            resolved = None
            for source in row["sources"]:
                data = resolve_source(source, task)
                if data is not None and digest(data) == row["sha256"]:
                    candidate, resolved = data, source
                    break
            if candidate is None:
                raise RuntimeError(f"cannot resolve task{task:03d}")
            authority_runs = [
                known_score(authority, task, False, f"expand20i94_auth_{task}_r{run}")
                for run in range(1, 4)
            ]
            candidate_runs = [
                known_score(candidate, task, False, f"expand20i94_cand_{task}_r{run}")
                for run in range(1, 4)
            ]
            output["tasks"][str(task)] = {
                "authority_sha256": digest(authority),
                "authority_runs": authority_runs,
                "authority_profiles_identical": len({json.dumps(item, sort_keys=True) for item in authority_runs}) == 1,
                "candidate_sha256": digest(candidate),
                "candidate_source": resolved,
                "candidate_runs": candidate_runs,
                "candidate_profiles_identical": len({json.dumps(item, sort_keys=True) for item in candidate_runs}) == 1,
            }
    (HERE / "audit/official_reprofile_3x.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({task: {
        "authority": row["authority_runs"][0],
        "candidate": row["candidate_runs"][0],
        "authority_identical": row["authority_profiles_identical"],
        "candidate_identical": row["candidate_profiles_identical"],
    } for task, row in output["tasks"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

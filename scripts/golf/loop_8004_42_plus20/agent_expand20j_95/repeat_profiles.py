#!/usr/bin/env python3
"""Repeat disputed official profiles with independent temporary directories."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def profile(task: int, data: bytes, label: str, repeat: int) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix=f"expand20j_repeat_{task:03d}_{repeat}_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), task, workdir, label=label, require_correct=False
        )
    if result is None:
        raise RuntimeError("score_and_verify returned None")
    return result


def main() -> int:
    models: list[tuple[str, int, bytes]] = []
    with zipfile.ZipFile(ROOT / "submission.zip") as archive:
        models.append(("authority_task014", 14, archive.read("task014.onnx")))
        models.append(("authority_task036", 36, archive.read("task036.onnx")))
    models.append(("candidate_task014", 14, (ROOT / "others/71405/task014_cost360.onnx").read_bytes()))
    models.append(("candidate_task036_737fc", 36, (ROOT / "others/2/1300/submission7300+/task036.onnx").read_bytes()))
    models.append(("candidate_task036_fc83b", 36, (ROOT / "scripts/golf/loop_8004_42_plus20/agent_rebuild_mid8/candidates/task036_truthful_gather.onnx").read_bytes()))
    with zipfile.ZipFile(ROOT / "artifacts/_BEST_7472.18.zip") as archive:
        models.append(("candidate_task036_dd794", 36, archive.read("task036.onnx")))

    rows = []
    for name, task, data in models:
        runs = [profile(task, data, name, repeat) for repeat in range(3)]
        rows.append(
            {
                "name": name,
                "task": task,
                "sha256": hashlib.sha256(data).hexdigest(),
                "runs": runs,
                "stable": len({(r["memory"], r["params"], r["cost"], r["correct"]) for r in runs}) == 1,
            }
        )
        print(name, [(r["memory"], r["params"], r["cost"], r["correct"]) for r in runs], flush=True)
    (HERE / "audit" / "repeat_official_profiles.json").write_text(json.dumps({"repeats": 3, "rows": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

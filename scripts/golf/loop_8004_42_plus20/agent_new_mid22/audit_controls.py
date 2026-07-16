#!/usr/bin/env python3
"""Measure retained specification-derived controls for all mid22 tasks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
from audit_lane import known_dual, score, structure  # noqa: E402


CONTROLS = {
    123: HERE / "current/task123.onnx",
    316: HERE / "current/task316.onnx",
    212: ROOT / "scripts/golf/scratch/task212/candidate_v13.onnx",
    301: ROOT / "scripts/golf/scratch/task301/candidate_v4.onnx",
    55: ROOT / "scripts/golf/scratch/task055/cand_v8.onnx",
    86: ROOT / "scripts/golf/scratch/task086/cand7.onnx",
    163: ROOT / "scripts/golf/scratch/task163/cand4.onnx",
    206: ROOT / "scripts/golf/scratch/task206/cand_v8.onnx",
}


def main() -> None:
    result = {}
    for task, path in CONTROLS.items():
        model = onnx.load(path)
        data = path.read_bytes()
        row = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "score": score(model, task),
            "known_dual": known_dual(model, task),
            "structure": structure(model, task),
        }
        result[str(task)] = row
        print(
            f"task{task:03d} cost={row['score'].get('cost')} "
            f"known={row['known_dual']['disable_all']['right']}/"
            f"{row['known_dual']['disable_all']['total']} "
            f"structure={row['structure']['pass']}",
            flush=True,
        )
    (HERE / "sound_controls_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

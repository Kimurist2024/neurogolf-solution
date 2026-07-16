#!/usr/bin/env python3
"""Run the reusable strict NeuroGolf model auditor for this lane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"

CASES = {
    "base_task005": (5, HERE / "baseline/task005.onnx"),
    "rejected_sound_task005": (5, HERE / "candidates/task005_sound_2389.onnx"),
    "rejected_giant_task005": (5, HERE / "candidates/task005_clean_2534.onnx"),
    "clean_task005": (5, HERE / "candidates/task005_clean_2545.onnx"),
    "base_task080": (80, HERE / "baseline/task080.onnx"),
    "base_task101": (101, HERE / "baseline/task101.onnx"),
    "sound_task101": (101, HERE / "candidates/task101_sound_7264.onnx"),
    "base_task133": (133, HERE / "baseline/task133.onnx"),
    "sound_task133": (133, HERE / "candidates/task133_sound_5570.onnx"),
}


def load_auditor():
    spec = importlib.util.spec_from_file_location("strict_auditor", AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUDITOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    auditor = load_auditor()
    output: dict[str, object] = {}
    for label, (task, path) in CASES.items():
        output[label] = auditor.audit(label, task, path)
        score = output[label].get("official_like_score")
        print(label, None if score is None else score.get("cost"), flush=True)
    (HERE / "model_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

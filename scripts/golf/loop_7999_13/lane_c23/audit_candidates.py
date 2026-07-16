#!/usr/bin/env python3
"""Audit C23 exact baselines, nearest history, and sound rebuild controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDITOR_PATH = HERE.parent / "lane_c11" / "audit_candidates.py"
SPEC = importlib.util.spec_from_file_location("lane_c11_auditor", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)

MODELS = {
    "base_task237": (237, HERE / "base" / "task237.onnx"),
    "base_task378": (378, HERE / "base" / "task378.onnx"),
    "task237_rebuild_542": (237, HERE / "candidates" / "task237_rebuild_542.onnx"),
    "task237_rebuild_543": (237, HERE / "candidates" / "task237_rebuild_543.onnx"),
    "task237_rebuild_544": (237, HERE / "candidates" / "task237_rebuild_544.onnx"),
    **{
        f"task378_r{index:02d}": (
            378,
            HERE / "candidates" / filename,
        )
        for index, filename in enumerate(
            [
                "task378_r01_static340.onnx",
                "task378_r02_static342.onnx",
                "task378_r03_static343.onnx",
                "task378_r04_static345.onnx",
                "task378_r05_static394.onnx",
                "task378_r06_static396.onnx",
                "task378_r07_static400.onnx",
                "task378_r08_static510.onnx",
            ],
            start=1,
        )
    },
    "task378_sound_fullmask": (378, HERE / "candidates" / "task378_sound_fullmask.onnx"),
    "task378_sound_k12_scaled": (378, HERE / "candidates" / "task378_sound_k12_scaled.onnx"),
}


def main() -> None:
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    for label, (task, path) in MODELS.items():
        if label in output:
            continue
        output[label] = AUDITOR.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score") or {}
        known = output[label].get("known_disable_all", {}).get("total", {})
        print(label, "cost", score.get("cost"), "correct", score.get("correct"), "known", known, flush=True)


if __name__ == "__main__":
    main()

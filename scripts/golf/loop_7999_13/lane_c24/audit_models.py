#!/usr/bin/env python3
"""Audit C24 exact baselines and any retained candidates."""

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

MODELS: dict[str, tuple[int, Path]] = {
    "base_task363": (363, HERE / "base" / "task363.onnx"),
    "base_task388": (388, HERE / "base" / "task388.onnx"),
    "task363_history_static514": (
        363,
        HERE / "candidates" / "task363_history_static514.onnx",
    ),
    "task363_history_static517": (
        363,
        HERE / "candidates" / "task363_history_static517.onnx",
    ),
    "task363_history_static525": (
        363,
        HERE / "candidates" / "task363_history_static525.onnx",
    ),
    "task363_generator_spec_core": (
        363,
        HERE / "candidates" / "task363_generator_spec_core.onnx",
    ),
    "task363_optimized_control": (
        363,
        HERE / "candidates" / "task363_optimized_control.onnx",
    ),
    "task388_r01_static81": (388, HERE / "candidates" / "task388_r01_static81.onnx"),
    "task388_r02_static82": (388, HERE / "candidates" / "task388_r02_static82.onnx"),
    "task388_r03_static82": (388, HERE / "candidates" / "task388_r03_static82.onnx"),
    "task388_r04_static83": (388, HERE / "candidates" / "task388_r04_static83.onnx"),
    "task388_r05_static85": (388, HERE / "candidates" / "task388_r05_static85.onnx"),
    "task388_history_cheaper137_wrong": (
        388,
        HERE / "candidates" / "task388_history_cheaper137_wrong.onnx",
    ),
    "task388_optimized_control": (
        388,
        HERE / "candidates" / "task388_optimized_control.onnx",
    ),
}


def main() -> None:
    output_path = HERE / "model_audit.json"
    output: dict[str, object] = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    for label, (task, path) in MODELS.items():
        if not path.exists() or label in output:
            continue
        output[label] = AUDITOR.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score") or {}
        known = output[label].get("known_disable_all", {}).get("total", {})
        print(label, "cost", score.get("cost"), "known", known, flush=True)


if __name__ == "__main__":
    main()

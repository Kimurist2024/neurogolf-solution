#!/usr/bin/env python3
"""Audit exact C22 baselines and all unique archive candidates."""

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
    "base_task009": (9, HERE / "base" / "task009.onnx"),
    "base_task076": (76, HERE / "base" / "task076.onnx"),
    "task009_r01_excluded": (9, HERE / "candidates" / "task009_r01_static2072.onnx"),
    "task009_r02": (9, HERE / "candidates" / "task009_r02_static2457.onnx"),
    "task076_r01": (76, HERE / "candidates" / "task076_r01_static1676.onnx"),
    "task076_r02": (76, HERE / "candidates" / "task076_r02_static1704.onnx"),
    "task076_r03": (76, HERE / "candidates" / "task076_r03_static1912.onnx"),
    "task076_r04": (76, HERE / "candidates" / "task076_r04_static1918.onnx"),
    "task076_r05": (76, HERE / "candidates" / "task076_r05_static1930.onnx"),
    "task076_r06": (76, HERE / "candidates" / "task076_r06_static1971.onnx"),
    "task076_r07": (76, HERE / "candidates" / "task076_r07_static1973.onnx"),
    "task076_r08": (76, HERE / "candidates" / "task076_r08_static2029.onnx"),
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

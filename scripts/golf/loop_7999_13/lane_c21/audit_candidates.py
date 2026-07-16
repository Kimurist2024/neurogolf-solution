#!/usr/bin/env python3
"""Audit exact C21 baselines and every unique cheap archive lead."""

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

BASES = {
    "base_task138": (138, HERE / "base" / "task138.onnx"),
    "base_task187": (187, HERE / "base" / "task187.onnx"),
}
CANDIDATES = {
    "task138_r01": (138, HERE / "candidates" / "task138_r01_static2588.onnx"),
    "task138_r02": (138, HERE / "candidates" / "task138_r02_static2648.onnx"),
    "task187_r01": (187, HERE / "candidates" / "task187_r01_static1368.onnx"),
    "task187_r02": (187, HERE / "candidates" / "task187_r02_static1368.onnx"),
    "task187_r03": (187, HERE / "candidates" / "task187_r03_static1371.onnx"),
    "task187_r04": (187, HERE / "candidates" / "task187_r04_static1377.onnx"),
    "task187_r05": (187, HERE / "candidates" / "task187_r05_static1383.onnx"),
    "task187_r06": (187, HERE / "candidates" / "task187_r06_static1609.onnx"),
    "task187_r07": (187, HERE / "candidates" / "task187_r07_static1621.onnx"),
    "task187_r08": (187, HERE / "candidates" / "task187_r08_static1754.onnx"),
}


def main() -> None:
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    for label, (task, path) in {**BASES, **CANDIDATES}.items():
        if label in output:
            continue
        output[label] = AUDITOR.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score") or {}
        known = output[label].get("known_disable_all", {}).get("total", {})
        print(
            label,
            "cost", score.get("cost"),
            "correct", score.get("correct"),
            "known", known,
            flush=True,
        )


if __name__ == "__main__":
    main()

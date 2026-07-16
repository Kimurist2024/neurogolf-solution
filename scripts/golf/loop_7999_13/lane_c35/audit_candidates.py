#!/usr/bin/env python3
"""Strict cost, structure, known, and runtime-shape audit for C35 task192."""

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
    "base_task192": HERE / "baseline" / "task192.onnx",
    "task192_r01_static403": HERE / "candidates" / "task192_r01_static403.onnx",
    "task192_r02_static493": HERE / "candidates" / "task192_r02_static493.onnx",
    "task192_r03_static509": HERE / "candidates" / "task192_r03_static509.onnx",
    "task192_r04_static561": HERE / "candidates" / "task192_r04_static561.onnx",
    "task192_r05_static589": HERE / "candidates" / "task192_r05_static589.onnx",
}


def main() -> None:
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = {}
    for label, path in MODELS.items():
        output[label] = AUDITOR.audit(label, 192, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score") or {}
        known_disabled = output[label].get("known_disable_all", {}).get("total", {})
        known_default = output[label].get("known_default", {}).get("total", {})
        print(label, score.get("cost"), known_disabled, known_default, flush=True)


if __name__ == "__main__":
    main()

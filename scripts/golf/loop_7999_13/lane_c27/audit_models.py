#!/usr/bin/env python3
"""Run the shared strict cost/correctness/shape audit on C27 models."""

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
    "base_task184": (184, HERE / "baseline" / "task184.onnx"),
    "base_task377": (377, HERE / "baseline" / "task377.onnx"),
    "task184_sound422_r06": (
        184,
        HERE / "candidates" / "task184_sound422_r06.onnx",
    ),
    "task184_sound422_r07": (
        184,
        HERE / "candidates" / "task184_sound422_r07.onnx",
    ),
    "task184_r06_shrink421": (
        184,
        HERE / "candidates" / "task184_r06_shrink421.onnx",
    ),
    "task377_diff5_witness408": (
        377,
        HERE / "candidates" / "task377_diff5_witness408.onnx",
    ),
}


def main() -> None:
    output_path = HERE / "evidence" / "model_audit.json"
    output: dict[str, object] = {}
    for label, (task, path) in MODELS.items():
        output[label] = AUDITOR.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score") or {}
        disabled = output[label].get("known_disable_all", {}).get("total", {})
        default = output[label].get("known_default", {}).get("total", {})
        mismatches = output[label].get("runtime_shape_trace", {}).get(
            "declared_actual_mismatches", []
        )
        print(
            label,
            "cost",
            score.get("cost"),
            "disabled",
            disabled,
            "default",
            default,
            "shape_mismatches",
            len(mismatches),
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-audit the strongest historical task018/task145 leads for lane C19."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDITOR_PATH = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("lane_c11_auditor", AUDITOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load auditor: {AUDITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CASES = {
    "base_task018": (18, HERE / "base/task018.onnx"),
    "task018_k18_signature": (
        18,
        ROOT / "scripts/golf/scratch_codex/task018/k18_fused_signature.onnx",
    ),
    "task018_compact_direct": (
        18,
        ROOT
        / "scripts/golf/scratch_codex/task018/alt_agent/compact_signature_direct.onnx",
    ),
    "task018_clean_rebuild": (
        18,
        ROOT / "scripts/golf/scratch_codex/task018/tile2x3_k22_allmode_clean.onnx",
    ),
    "base_task145": (145, HERE / "base/task145.onnx"),
    "task145_near_cost": (
        145,
        ROOT / "others/2/1203/task145_improved_v3.onnx",
    ),
    "task145_honest_numeric": (
        145,
        ROOT / "scripts/golf/scratch_codex/task145/task145_honest_numeric.onnx",
    ),
}


def main() -> None:
    ort.set_default_logger_severity(4)
    auditor = load_auditor()
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = {}
    for label, (task, path) in CASES.items():
        output[label] = auditor.audit(label, task, path)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score")
        print(
            label,
            None if score is None else score.get("cost"),
            None if score is None else score.get("correct"),
            flush=True,
        )


if __name__ == "__main__":
    main()

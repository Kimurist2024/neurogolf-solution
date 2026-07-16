#!/usr/bin/env python3
"""Actual-cost, known-set, dual-ORT, and shape audit for lane C20."""

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
    "base_task133": (133, HERE / "base/task133.onnx"),
    "task133_safe_pcolor": (
        133,
        ROOT / "scripts/golf/scratch_codex/task133/safe_pcolor_colvar.onnx",
    ),
    "task133_safe_spec_clean": (
        133,
        ROOT / "scripts/golf/scratch_codex/task133/safe_spec_clean.onnx",
    ),
    "task133_safe_spec_no_vi": (
        133,
        ROOT / "scripts/golf/scratch_codex/task133/safe_spec_no_vi.onnx",
    ),
    "task133_safe_spec_shave": (
        133,
        ROOT / "scripts/golf/scratch_codex/task133/safe_spec_shave.onnx",
    ),
    "task133_clean_rank_rebuild": (
        133,
        ROOT / "scripts/golf/scratch_codex/task133/agent_clean_rank.onnx",
    ),
    "base_task349": (349, HERE / "base/task349.onnx"),
    "task349_archive_static3547": (
        349,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task349_r01_static3547.onnx",
    ),
    "task349_archive_static3698": (
        349,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task349_r02_static3698.onnx",
    ),
    "task349_archive_static3710": (
        349,
        ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task349_r03_static3710.onnx",
    ),
    "task349_tables_3956": (
        349,
        ROOT
        / "scripts/golf/loop_7999_13/lane_b8/candidates/task349_radius_tables_len9.onnx",
    ),
    "task349_relation_3954": (
        349,
        ROOT
        / "scripts/golf/loop_7999_13/lane_b8/candidates/task349_radius_tables_len9_top_relation.onnx",
    ),
    "task349_exact_or_v2": (
        349,
        ROOT / "scripts/golf/scratch_codex/task349/agent_spec_tail_exact_or_v2.onnx",
    ),
    "task349_exact_opt": (
        349,
        ROOT / "scripts/golf/scratch_codex/task349/agent_alt_exact_opt.onnx",
    ),
    "task349_exact_or_affine": (
        349,
        ROOT / "scripts/golf/scratch_codex/task349/agent_alt_exact_or_affine.onnx",
    ),
    "task349_exact_or_tail4": (
        349,
        ROOT / "scripts/golf/scratch_codex/task349/agent_alt_exact_or_tail4.onnx",
    ),
    "task349_clean_morph": (
        349,
        ROOT / "scripts/golf/scratch_claude/task349/clean_morph.onnx",
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

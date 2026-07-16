#!/usr/bin/env python3
"""Audit C15 exact bases and selected historical/rebuild probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (112, 148, 212, 301, 316, 325, 341)
PROBES = (
    ("probe_task112_c5_cloaked", 112, ROOT / "scripts/golf/loop_7999_13/lane_c5/task112_broadcast_sign_cloaked.onnx"),
    ("probe_task112_c5_truthful", 112, ROOT / "scripts/golf/loop_7999_13/lane_c5/task112_broadcast_sign_truthful.onnx"),
    ("probe_task112_spec_opt2", 112, ROOT / "scripts/golf/scratch/task112/cand_opt2.onnx"),
    ("probe_task112_sound_gather", 112, ROOT / "scripts/golf/scratch_codex_7994/task112_sound/candidate_sound_gather.onnx"),
    ("probe_task112_handcrafted420", 112, ROOT / "artifacts/handcrafted/task112.onnx"),
    ("probe_task112_factor_sign", 112, ROOT / "scripts/golf/scratch_codex/agent_mid_b/task112_factor_sign.onnx"),
    ("probe_task148_incumbent265", 148, ROOT / "scripts/golf/scratch_codex/deep_task148/incumbent265.onnx"),
    ("probe_task148_z5hi_scalar", 148, ROOT / "scripts/golf/scratch_codex/deep_task148/z5hi_scalar.onnx"),
    ("probe_task148_spec", 148, ROOT / "scripts/golf/scratch/task148/cand.onnx"),
    ("probe_task148_v6", 148, ROOT / "scripts/golf/scratch_codex/task148/cand_v6.onnx"),
    ("probe_task212_spec_v13", 212, ROOT / "scripts/golf/scratch/task212/candidate_v13.onnx"),
    ("probe_task301_spec_v4", 301, ROOT / "scripts/golf/scratch/task301/candidate_v4.onnx"),
    ("probe_task325_prefix_outer", 325, ROOT / "scripts/golf/scratch_codex/task325/candidate_prefix_outer.onnx"),
    ("probe_task325_shift_drop29", 325, ROOT / "scripts/golf/scratch_codex/agent_mid_e/task325_shift_drop29.onnx"),
    ("probe_task325_shift_drop31", 325, ROOT / "scripts/golf/scratch_codex/agent_mid_e/task325_shift_drop31.onnx"),
    ("probe_task325_exact", 325, ROOT / "scripts/golf/scratch_codex/task325/candidate_exact.onnx"),
    ("probe_task325_local7", 325, ROOT / "scripts/golf/scratch_codex/task325/candidate_local7.onnx"),
    ("probe_task341_center_v4", 341, ROOT / "scripts/golf/scratch_codex/task341/candidate_ground_up_center_index_v4.onnx"),
    ("probe_task341_zero_carrier", 341, ROOT / "scripts/golf/scratch_codex/agent_mid_c/task341_zero_carrier.onnx"),
    ("probe_task341_cast_attr", 341, ROOT / "scripts/golf/scratch_codex/castlike_attr_sweep/task341_cast_attr.onnx"),
)


def load_auditor():
    path = HERE.parent / "lane_c11" / "audit_candidates.py"
    spec = importlib.util.spec_from_file_location("c11_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="full-match regex for labels")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    auditor = load_auditor()
    output_path = HERE / "candidate_audit.json"
    output: dict[str, object] = {}
    if args.resume and output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
    jobs: list[tuple[str, int, Path]] = [
        (f"base_task{task:03d}", task, HERE / "base" / f"task{task:03d}.onnx")
        for task in TARGETS
    ]
    jobs.extend(PROBES)
    for label, task, path in jobs:
        if args.only and re.fullmatch(args.only, label) is None:
            continue
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

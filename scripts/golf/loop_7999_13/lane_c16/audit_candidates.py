#!/usr/bin/env python3
"""Audit C16 exact bases and selected cost-floor probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (74, 93, 136, 180, 221, 278, 295)
PROBES = (
    ("probe_task074_spec_d4", 74, ROOT / "scripts/golf/scratch/task074/cand5.onnx"),
    ("probe_task074_orbit_table", 74, ROOT / "scripts/golf/scratch_codex/task074/task074_rebuild_float_table.onnx"),
    ("probe_task093_spec_final", 93, ROOT / "scripts/golf/scratch_codex/task093/candidate_final.onnx"),
    ("probe_task093_5feat", 93, ROOT / "scripts/golf/scratch_codex/task093/candidate_5feat.onnx"),
    ("probe_task093_no_conv_bias", 93, ROOT / "scripts/golf/scratch_codex/task093/candidate_no_conv_bias.onnx"),
    ("probe_task136_history_improved", 136, ROOT / "others/2/1294/task136_improved.onnx"),
    ("probe_task180_spec_priority", 180, ROOT / "scripts/golf/scratch/task180/candidate_priority_label.onnx"),
    ("probe_task180_direct_yzp", 180, ROOT / "scripts/golf/scratch_codex/task180/direct_yzp.onnx"),
    ("probe_task180_direct_bias", 180, ROOT / "scripts/golf/scratch_codex/task180/direct_bias.onnx"),
    ("probe_task180_no_z0", 180, ROOT / "scripts/golf/scratch_codex/task180/variants/no_z0.onnx"),
    ("probe_task221_shrink_qi", 221, ROOT / "scripts/golf/scratch_codex/deep_task221/task221_shrink_qi.onnx"),
    ("probe_task221_output_scalar", 221, ROOT / "scripts/golf/scratch_codex/task221/output_scalar_decl.onnx"),
    ("probe_task221_idx_cloaks", 221, ROOT / "scripts/golf/scratch_codex/task221/shrink_idx_cloaks.onnx"),
    ("probe_task221_uint8_indices", 221, ROOT / "scripts/golf/scratch_codex/task221/uint8_indices.onnx"),
    ("probe_task278_cloak_conv", 278, ROOT / "scripts/golf/scratch_codex/task278/cloak_conv.onnx"),
    ("probe_task278_minmax", 278, ROOT / "scripts/golf/scratch_codex/task278/minmax.onnx"),
    ("probe_task278_shrink_bgc8", 278, ROOT / "scripts/golf/scratch_codex/task278/variants/shrink_bgc8.onnx"),
    ("probe_task278_cast_attr", 278, ROOT / "scripts/golf/scratch_codex/task278/task278_cast_attr.onnx"),
    ("probe_task295_best", 295, ROOT / "scripts/golf/scratch_agents/task295/best.onnx"),
    ("probe_task295_f16_inits", 295, ROOT / "scripts/golf/scratch_codex/task295/cand_f16_inits.onnx"),
    ("probe_task295_spec_v2", 295, ROOT / "scripts/golf/scratch/task295/candidate_v2.onnx"),
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
    for path in sorted((HERE / "task074_mode_probes").glob("*.onnx")):
        jobs.append((f"probe_task074_mode_{path.stem}", 74, path))
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

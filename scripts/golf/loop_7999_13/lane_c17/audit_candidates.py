#!/usr/bin/env python3
"""C17 audit for task233/task366 exact-base and retained candidate graphs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
C11 = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
SPEC = importlib.util.spec_from_file_location("c11_audit", C11)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

CASES = {
    "base_task233": (233, HERE / "base/task233.onnx"),
    "task233_shape_cloak_8992": (
        233,
        ROOT / "scripts/golf/scratch_codex/task233/agent_candidate_shrink/cand_9213_all_extended.onnx",
    ),
    "task233_shape_cloak_9189": (
        233,
        ROOT / "scripts/golf/scratch_codex/task233/agent_graph_audit/rebased_extended2.onnx",
    ),
    "task233_sound_floor_17007": (
        233,
        ROOT / "scripts/golf/scratch_codex/task233/graph_floor/packed_qlinear_hline8.onnx",
    ),
    "base_task366": (366, HERE / "base/task366.onnx"),
    "task366_known_only_7646": (
        366,
        ROOT / "others/2/1201/7120/task366_further_improved.onnx",
    ),
    "task366_known_only_7916": (
        366,
        ROOT / "others/2/1203/task366_further_improved.onnx",
    ),
    "task366_known_only_7985": (
        366,
        ROOT / "others/2/1203/task366_improved.onnx",
    ),
    "task366_known_fail_5246": (
        366,
        ROOT / "others/7906/task366_improved_v2 (1).onnx",
    ),
    "task366_cc_core_floor_13309": (
        366,
        ROOT / "scripts/golf/scratch_claude/task366/cc_floor.onnx",
    ),
}


def main() -> None:
    output = {}
    target = HERE / "candidate_audit.json"
    for label, (task, path) in CASES.items():
        output[label] = AUDIT.audit(label, task, path)
        target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        score = output[label].get("official_like_score", {})
        print(label, score.get("cost"), score.get("correct"), flush=True)


if __name__ == "__main__":
    main()

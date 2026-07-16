#!/usr/bin/env python3
"""Reuse the strict task158 audit stack against the cost-7529 baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(
    0,
    str(
        ROOT
        / "scripts/golf/loop_8004_42_plus20/agent_task158_current_108"
    ),
)
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_a20"))
import audit_lane as previous  # noqa: E402
import audit_history as shared  # noqa: E402


shared.BASE_COST = {158: 7529}


def main() -> None:
    paths = {
        "baseline_7529": HERE / "baseline/task158.onnx",
        "uint8_rank_topk_rejected_control": (
            HERE / "candidates/task158_uint8_rank_topk.onnx"
        ),
        "anchor_bias_shift_expected_7527": (
            HERE / "candidates/task158_anchor_bias_shift.onnx"
        ),
        "anchor_bias_shift_alias_expected_7526": (
            HERE / "candidates/task158_anchor_bias_shift_alias.onnx"
        ),
        "anchor_bias_shift_scaled_alias_expected_7525": (
            HERE / "sound/task158_exact_regolf.onnx"
        ),
    }
    rows = []
    for label, path in paths.items():
        row = shared.audit(
            158,
            label,
            path,
            None,
            [
                "scripts/golf/loop_8004_42_plus20/"
                "agent_task158_current_108/sound/"
                "task158_exact_repair_cost7529.onnx"
            ],
            baseline=label.startswith("baseline"),
        )
        row["known_raw_equivalence_to_trusted_7612"] = (
            previous.known_raw_equivalence(path)
        )
        rows.append(row)
        (HERE / "evidence/audit.json").write_text(
            json.dumps({"rows": rows, "complete": False}, indent=2) + "\n"
        )
        profile = row.get("official_like_score") or {}
        print(
            label,
            profile.get("cost"),
            row.get("known_disable_all"),
            row.get("known_default"),
            row.get("pre_fresh_reasons"),
            flush=True,
        )
    (HERE / "evidence/audit.json").write_text(
        json.dumps({"rows": rows, "complete": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

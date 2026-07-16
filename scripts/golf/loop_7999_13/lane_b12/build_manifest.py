#!/usr/bin/env python3
"""Build the B12 rejection/winner manifest from independent audit evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    audit = json.loads((HERE / "structural_audit.json").read_text(encoding="utf-8"))
    reasons = {
        254: ["33-input giant Einsum (immediate-reject structure)"],
        267: ["37-input giant Einsum (immediate-reject structure)"],
        322: [
            "ConvTranspose bias length 9 for 10 output channels (undefined behavior)",
            "nonfinite initializer X and nonfinite raw output cells",
        ],
        323: [
            "56-input giant Einsum (immediate-reject structure)",
            "extreme float32 raw magnitude up to 6.237532218840826e34",
        ],
        372: [
            "ConvTranspose bias length 9 for 10 output channels (undefined behavior)",
            "nonfinite initializer X and nonfinite raw output cells",
        ],
    }
    rule_summaries = {
        254: "fixed 9x9 gray bars; retain only the minimum-height bar in red and maximum-height bar in blue",
        267: "fixed 7x7 creature; replace its source color by the marker color stored at row 6, column 0",
        322: "fixed 3x3; extend each column's single colored seed downward to the bottom row",
        323: "fixed 13x13; emit the bounded alternating two-down/two-side gray staircase from the cyan seed",
        372: "merge the two 5-row colored halves around the gray separator into one 5x11 output",
    }
    task_rows: dict[str, object] = {}
    rejected_hypothetical_gain = 0.0
    for task_text, row in audit["tasks"].items():
        task = int(task_text)
        base_cost = int(row["baseline_actual_profile"]["cost"])
        candidate_cost = int(row["candidate_actual_profile"]["cost"])
        gain = math.log(base_cost / candidate_cost)
        rejected_hypothetical_gain += gain
        candidate_structure = row["candidate_structure"]
        task_rows[task_text] = {
            "decision": "REJECT",
            "reasons": reasons[task],
            "generator_hash": row["generator_hash"],
            "generator_rule": rule_summaries[task],
            "lineage": row["lineage"],
            "baseline_actual_cost": row["baseline_actual_profile"],
            "candidate_actual_cost": row["candidate_actual_profile"],
            "baseline_static_cost": row["baseline_structure"]["static_cost"],
            "candidate_static_cost": candidate_structure["static_cost"],
            "hypothetical_score_gain_rejected": gain,
            "full_checker": candidate_structure["checker_full"],
            "strict_shape_inference_data_prop": candidate_structure[
                "strict_shape_inference_data_prop"
            ],
            "standard_domains": candidate_structure["standard_domains"],
            "runtime_shape_mismatch_count": len(
                row["candidate_runtime_shapes"]["shape_mismatches"]
            ),
            "conv_bias_findings": candidate_structure["conv_bias_findings"],
            "giant_einsum_nodes": candidate_structure["giant_einsum_nodes"],
            "lookup_ops": candidate_structure["lookup_ops"],
            "nonfinite_initializers": candidate_structure["nonfinite_initializers"],
            "known_disable_all": row["known_disable_all"],
            "known_default": row["known_default"],
            "fresh5000": "not run: immediate structural/UB rejection",
        }
    manifest = {
        "campaign": "loop_7999_13_lane_b12",
        "baseline_score": 7999.13,
        "baseline_zip": audit["baseline_zip"],
        "baseline_zip_sha256": audit["baseline_zip_sha256"],
        "policy": {
            "immediate_reject": [
                "giant Einsum",
                "Conv/ConvTranspose bias length mismatch",
                "undefined behavior",
            ],
            "fresh5000_only_after_early_structural_gate": True,
        },
        "winner_manifest": [],
        "rejected_tasks": [254, 267, 322, 323, 372],
        "accepted_tasks": [],
        "aggregate_cost_delta": 0,
        "aggregate_score_delta": 0.0,
        "hypothetical_rejected_score_gain": rejected_hypothetical_gain,
        "fresh5000_runs": [],
        "fresh5000_omission_reason": "All five candidates hit an immediate giant-Einsum or ConvTranspose-bias/UB rejection gate.",
        "tasks": task_rows,
        "evidence": ["structural_audit.json", "REPORT.md"],
        "root_submission_mutations": [],
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "baseline_score": 7999.13,
                "winners": [],
                "reason": manifest["fresh5000_omission_reason"],
                "root_submission_mutations": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"winners": 0, "rejected": 5, "score_delta": 0.0}, indent=2))


if __name__ == "__main__":
    main()

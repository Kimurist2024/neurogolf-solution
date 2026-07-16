#!/usr/bin/env python3
"""Assemble machine-readable final evidence for the mid20_84 lane."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESCAN = HERE / "authority_rescan/rescreen.json"
AUTHORITY = HERE / "authority_official_profiles.json"
OLD_COSTS = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
CAND297 = HERE / "candidates/task297_5b1b2dbc6f86_cost361.onnx"
STANDARD297 = ROOT / "scripts/golf/loop_7999_13/lane_a10/task297_standard_profiles.json"
CONFIRM297 = HERE / "policy90_confirmation.json"
MECHANICAL = HERE / "mechanical_reductions.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    audit_dir = HERE / "audit"
    audit_dir.mkdir(exist_ok=True)
    data = json.loads(RESCAN.read_text())
    rows = data["rows"]
    authority = json.loads(AUTHORITY.read_text())["tasks"]
    old_costs = json.loads(OLD_COSTS.read_text())["costs"]

    per_task = []
    for task in data["targets"]:
        task_rows = [row for row in rows if int(row["task"]) == task]
        stage_counts = Counter(row["stage"] for row in task_rows)
        screened = [int(row["actual_screen_cost"]) for row in task_rows if row.get("actual_screen_cost") is not None]
        official = [
            row["official_like_score"]
            for row in task_rows
            if isinstance(row.get("official_like_score"), dict)
        ]
        reasons = Counter(reason for row in task_rows for reason in row.get("reasons", []))
        conclusion = "no_strictly_lower_safe_known_correct_candidate"
        if task in {112, 168}:
            conclusion = "fail_closed_private_or_unsound_without_complete_true_rule_proof"
        if task == 297:
            conclusion = "only_lower_correct_candidate_uses_negative_conv_padding_and_is_quarantined"
        per_task.append(
            {
                "task": task,
                "authority_cost": int(authority[str(task)]["cost"]),
                "authority_sha256": authority[str(task)]["sha256"],
                "unique_nonauthority_sha": len(task_rows),
                "stage_counts": dict(stage_counts),
                "actual_screened": len(screened),
                "minimum_actual_screen_cost": min(screened) if screened else None,
                "official_like_profiles_returned": len(official),
                "minimum_correct_official_like_cost": min((int(item["cost"]) for item in official if item.get("correct")), default=None),
                "top_reasons": reasons.most_common(5),
                "conclusion": conclusion,
            }
        )

    inventory_summary = {
        "baseline_zip": data["baseline_zip"],
        "baseline_zip_sha256": data["baseline_zip_sha256"],
        "targets": data["targets"],
        "observed_unique_nonauthority_sha": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "inventory": data["inventory"],
        "per_task": per_task,
    }
    (audit_dir / "inventory_summary.json").write_text(json.dumps(inventory_summary, indent=2) + "\n")

    model = onnx.load(CAND297)
    negative_pads = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "ConvTranspose"}:
            continue
        for attr in node.attribute:
            if attr.name == "pads" and any(value < 0 for value in attr.ints):
                negative_pads.append({"output": node.output[0], "pads": list(attr.ints)})
    task297 = {
        "task": 297,
        "candidate": str(CAND297.relative_to(ROOT)),
        "sha256": sha256(CAND297),
        "authority_cost": 371,
        "candidate_cost": 361,
        "projected_gain_if_schema_extension_were_allowed": 0.027324104274554176,
        "known_dual": "265/265 in both ORT modes",
        "fresh_confirmation": json.loads(CONFIRM297.read_text()),
        "negative_conv_pads": negative_pads,
        "schema_decision": "REJECT_NEGATIVE_CONV_PADS_FAIL_CLOSED",
        "standard_schema_alternatives": json.loads(STANDARD297.read_text()),
        "conclusion": "No schema-compliant strictly cheaper task297 candidate exists in the tested inventory/variants.",
    }
    (audit_dir / "task297_schema_rejection.json").write_text(json.dumps(task297, indent=2) + "\n")

    stale = []
    for task in data["targets"]:
        old = int(old_costs[str(task)])
        actual = int(authority[str(task)]["cost"])
        if old != actual:
            stale.append({"task": task, "old_table_cost": old, "exact_zip_profile_cost": actual})
    mechanical = json.loads(MECHANICAL.read_text())
    result = {
        "status": "NO_SAFE_WINNER",
        "baseline_zip": data["baseline_zip"],
        "baseline_zip_sha256": data["baseline_zip_sha256"],
        "target_count": len(data["targets"]),
        "targets": data["targets"],
        "authority_costs": {str(task): int(authority[str(task)]["cost"]) for task in data["targets"]},
        "inventory_unique_sha": len(rows),
        "actual_screen_jobs_submitted": sum(
            1 for row in rows if row["stage"] in {"actual_reject", "known_reject", "known_dual_reject", "shape_reject", "fresh500_reject", "fresh500_pass"}
        ),
        "actual_screen_results_with_cost": sum(1 for row in rows if row.get("actual_screen_cost") is not None),
        "official_like_known_jobs": sum(1 for row in rows if row["stage"] == "known_reject"),
        "safe_pre_fresh": 0,
        "admitted": 0,
        "projected_total_gain": 0.0,
        "stale_cost_table_mismatches_corrected": stale,
        "mechanical_exact_reduction_candidates": mechanical["candidate_count"],
        "task297_quarantined_candidate": {
            "cost": 361,
            "reason": "Conv pads=[0,0,0,-24] violates non-negative schema contract",
            "fresh": "500/500 plus two independent 5000/5000 dual-ORT passes; still rejected structurally",
        },
        "protected_root_artifacts_modified": [],
        "evidence": {
            "rescreen": str(RESCAN.relative_to(ROOT)),
            "inventory_summary": str((audit_dir / "inventory_summary.json").relative_to(ROOT)),
            "task297_schema_rejection": str((audit_dir / "task297_schema_rejection.json").relative_to(ROOT)),
            "authority_profiles": str(AUTHORITY.relative_to(ROOT)),
            "mechanical_reductions": str(MECHANICAL.relative_to(ROOT)),
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "status": "NO_SAFE_WINNER",
                "baseline_zip": data["baseline_zip"],
                "baseline_zip_sha256": data["baseline_zip_sha256"],
                "count": 0,
                "projected_total_gain": 0.0,
                "candidates": [],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

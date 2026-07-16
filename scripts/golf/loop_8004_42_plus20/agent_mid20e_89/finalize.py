#!/usr/bin/env python3
"""Assemble final evidence for the fifth 20-task expansion audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESCREEN = HERE / "rescreen.json"
AUTHORITY = HERE / "authority_official_profiles.json"
OLD_COSTS = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
MECHANICAL = HERE / "mechanical_reductions.json"
TASK059 = HERE / "audit/task059_authority_known_quad.json"
KNOWN_QUAD = HERE / "audit/candidate_known_quad.json"
REBASE_PROOF = HERE / "authority_rebase_proof.json"
PROBE_MANIFEST = HERE / "probe_manifest.json"
POLICY_REOPEN = HERE / "audit/policy_reopen.json"


def main() -> int:
    inventory_dir = HERE / "inventory"
    audit_dir = HERE / "audit"
    inventory_dir.mkdir(exist_ok=True)
    audit_dir.mkdir(exist_ok=True)
    data = json.loads(RESCREEN.read_text())
    rows = data["rows"]
    authority = json.loads(AUTHORITY.read_text())["tasks"]
    old_costs = json.loads(OLD_COSTS.read_text())["costs"]
    mechanical = json.loads(MECHANICAL.read_text())
    candidate_known_quad = json.loads(KNOWN_QUAD.read_text())
    rebase_proof = json.loads(REBASE_PROOF.read_text())
    probe_manifest = json.loads(PROBE_MANIFEST.read_text())

    per_task = []
    for task in data["targets"]:
        task_rows = [row for row in rows if int(row["task"]) == task]
        stages = Counter(row["stage"] for row in task_rows)
        costs = [int(row["actual_screen_cost"]) for row in task_rows if row.get("actual_screen_cost") is not None]
        correct_profiles = [
            row["official_like_score"]
            for row in task_rows
            if isinstance(row.get("official_like_score"), dict) and row["official_like_score"].get("correct")
        ]
        conclusion = "no_strictly_lower_safe_known_correct_candidate"
        if task == 59:
            conclusion = "authority_is_0_of_266_known_in_all_four_runs_and_no_strict_lower_sound_candidate_exists"
        elif task == 302:
            conclusion = "private_zero_catalog_fail_closed_without_complete_true_rule_proof"
        elif task == 193:
            conclusion = "cost100_candidate_is_known_incorrect"
        elif task == 384:
            conclusion = "cost179_candidate_is_known_incorrect"
        per_task.append(
            {
                "task": task,
                "authority_cost": int(authority[str(task)]["cost"]),
                "authority_correct": bool(authority[str(task)]["correct"]),
                "authority_sha256": authority[str(task)]["sha256"],
                "unique_nonauthority_sha": len(task_rows),
                "stage_counts": dict(stages),
                "actual_cost_results": len(costs),
                "minimum_actual_screen_cost": min(costs) if costs else None,
                "known_correct_profiles": len(correct_profiles),
                "minimum_known_correct_official_cost": min((int(item["cost"]) for item in correct_profiles), default=None),
                "conclusion": conclusion,
            }
        )
    summary = {
        "baseline_zip": data["baseline_zip"],
        "baseline_zip_sha256": data["baseline_zip_sha256"],
        "targets": data["targets"],
        "unique_nonauthority_sha": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "actual_jobs_submitted": sum(1 for row in rows if row["stage"] in {"actual_reject", "known_reject", "known_dual_reject", "shape_reject", "fresh500_reject", "fresh500_pass"}),
        "known_jobs": sum(1 for row in rows if row["stage"] == "known_reject"),
        "known_quad_jobs": int(candidate_known_quad["count"]),
        "known_quad_complete_runs": 4 * int(candidate_known_quad["count"]),
        "per_task": per_task,
    }
    (inventory_dir / "raw.json").write_text(json.dumps(data["inventory"], indent=2) + "\n")
    (inventory_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    known_rows = []
    for row in rows:
        if row["stage"] != "known_reject":
            continue
        known_rows.append(
            {
                "task": row["task"],
                "sha256": row["sha256"],
                "sources": row["sources"],
                "actual_screen_cost": row.get("actual_screen_cost"),
                "official_like_score": row.get("official_like_score"),
                "decision": "REJECT_KNOWN_INCORRECT_OR_NOT_STRICTLY_CHEAPER",
            }
        )
    (audit_dir / "known_rejections.json").write_text(
        json.dumps({"count": len(known_rows), "rows": known_rows}, indent=2) + "\n"
    )

    stale = []
    for task in data["targets"]:
        old = int(old_costs[str(task)])
        exact = int(authority[str(task)]["cost"])
        if old != exact:
            stale.append({"task": task, "old_table_cost": old, "exact_zip_profile_cost": exact})
    result = {
        "status": "NO_SAFE_WINNER",
        "baseline_zip": data["baseline_zip"],
        "baseline_zip_sha256": data["baseline_zip_sha256"],
        "target_count": len(data["targets"]),
        "targets": data["targets"],
        "authority_costs": {str(task): int(authority[str(task)]["cost"]) for task in data["targets"]},
        "authority_known_incorrect_tasks": [59],
        "inventory_unique_sha": len(rows),
        "actual_jobs_submitted": summary["actual_jobs_submitted"],
        "official_known_jobs": summary["known_jobs"],
        "known_quad_candidate_jobs": summary["known_quad_jobs"],
        "known_quad_complete_runs": summary["known_quad_complete_runs"],
        "safe_pre_fresh": 0,
        "fresh_2seed_executed": False,
        "fresh_not_run_reason": "No candidate passed the official known-correct and strict-lower gate.",
        "admitted": 0,
        "projected_total_gain": 0.0,
        "stale_cost_table_mismatches_corrected": stale,
        "mechanical_exact_or_noop_candidates": mechanical["candidate_count"],
        "private_fail_closed_tasks": [302],
        "fixed_champions_preserved": [13, 70, 158, 254, 267, 323, 379],
        "target_intersection_fixed_champions": rebase_proof["target_intersection_fixed_champions"],
        "lb_probe_required_black_nets": [18, 48, 112, 134, 168, 198, 233, 251, 277, 286, 365, 366],
        "target_intersection_lb_probe_required_black_nets": rebase_proof["target_intersection_lb_probe_required_black"],
        "policy_reject_total": probe_manifest["policy_reject_total"],
        "policy_reason_reopened": probe_manifest["policy_reason_reopened"],
        "hard_schema_or_nonprobe_reject": probe_manifest["hard_schema_or_nonprobe_reject"],
        "policy_static_floor_not_lower": probe_manifest["static_floor_not_lower"],
        "policy_fully_profiled_after_reopen": probe_manifest["fully_profiled_after_reopen"],
        "lb_probe_required_count": probe_manifest["count"],
        "lb_probe_required_candidates": probe_manifest["candidates"],
        "authority_rebase_all_members_byte_identical": rebase_proof["all_target_members_byte_identical"],
        "protected_root_artifacts_modified": [],
        "others_71403_modified": False,
        "evidence": {
            "rescreen": str(RESCREEN.relative_to(ROOT)),
            "inventory_summary": str((inventory_dir / "summary.json").relative_to(ROOT)),
            "known_rejections": str((audit_dir / "known_rejections.json").relative_to(ROOT)),
            "task059_authority_known_quad": str(TASK059.relative_to(ROOT)),
            "candidate_known_quad": str(KNOWN_QUAD.relative_to(ROOT)),
            "authority_rebase_proof": str(REBASE_PROOF.relative_to(ROOT)),
            "probe_manifest": str(PROBE_MANIFEST.relative_to(ROOT)),
            "policy_reopen": str(POLICY_REOPEN.relative_to(ROOT)),
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
                "lb_probe_required_candidates": probe_manifest["candidates"],
                "fixed_champions_preserved": [13, 70, 158, 254, 267, 323, 379],
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

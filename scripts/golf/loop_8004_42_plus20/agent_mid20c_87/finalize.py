#!/usr/bin/env python3
"""Assemble the final evidence bundle for the third 20-task expansion."""

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
CRASH_SHA = "2df7617db09373acc416cbf505fff79823fabb22194d7f5c554c56975f43625a"
CRASH_PRIOR = ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/evidence/task124_validate_exit.json"


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

    def terminal_stage(row):
        return "runtime_crash_reject" if row["sha256"] == CRASH_SHA else row["stage"]

    per_task = []
    for task in data["targets"]:
        task_rows = [row for row in rows if int(row["task"]) == task]
        stages = Counter(terminal_stage(row) for row in task_rows)
        screen_costs = [int(row["actual_screen_cost"]) for row in task_rows if row.get("actual_screen_cost") is not None]
        correct_profiles = [
            row["official_like_score"]
            for row in task_rows
            if isinstance(row.get("official_like_score"), dict) and row["official_like_score"].get("correct")
        ]
        conclusion = "no_strictly_lower_safe_known_correct_candidate"
        if task in {178, 169, 174, 325}:
            conclusion = "private_or_unsound_fail_closed_without_complete_true_rule_proof"
        elif task == 124:
            conclusion = "only_static_lower_probe_crashes_ORT_with_SIGSEGV"
        elif task == 91:
            conclusion = "known_correct_history_floor_266_is_not_below_authority_265"
        per_task.append(
            {
                "task": task,
                "authority_cost": int(authority[str(task)]["cost"]),
                "authority_sha256": authority[str(task)]["sha256"],
                "unique_nonauthority_sha": len(task_rows),
                "stage_counts": dict(stages),
                "actual_cost_results": len(screen_costs),
                "minimum_actual_screen_cost": min(screen_costs) if screen_costs else None,
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
        "raw_stage_counts": dict(Counter(row["stage"] for row in rows)),
        "terminal_stage_counts": dict(Counter(terminal_stage(row) for row in rows)),
        "actual_jobs_submitted": sum(
            1
            for row in rows
            if terminal_stage(row) in {
                "runtime_crash_reject", "actual_reject", "known_reject", "known_dual_reject",
                "shape_reject", "fresh500_reject", "fresh500_pass",
            }
        ),
        "known_jobs": sum(1 for row in rows if row["stage"] == "known_reject"),
        "per_task": per_task,
    }
    (inventory_dir / "raw.json").write_text(json.dumps(data["inventory"], indent=2) + "\n")
    (inventory_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    known_rows = []
    for row in rows:
        if row["stage"] != "known_reject":
            continue
        profile = row.get("official_like_score")
        known_rows.append(
            {
                "task": row["task"],
                "sha256": row["sha256"],
                "actual_screen_cost": row.get("actual_screen_cost"),
                "official_like_score": profile,
                "decision": (
                    "REJECT_NOT_STRICTLY_CHEAPER"
                    if isinstance(profile, dict) and profile.get("correct")
                    else "REJECT_KNOWN_INCORRECT"
                ),
            }
        )
    (audit_dir / "known_rejections.json").write_text(
        json.dumps({"count": len(known_rows), "rows": known_rows}, indent=2) + "\n"
    )

    crash_row = next(row for row in rows if row["sha256"] == CRASH_SHA)
    crash = {
        "task": 124,
        "sha256": CRASH_SHA,
        "current_scan_row": crash_row,
        "prior_isolated_reproduction": json.loads(CRASH_PRIOR.read_text()),
        "decision": "REJECT_RUNTIME_CRASH",
        "reason": "Omitting the unused variadic Split output reproducibly exits validation with SIGSEGV/139.",
    }
    (audit_dir / "task124_runtime_crash.json").write_text(json.dumps(crash, indent=2) + "\n")

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
        "inventory_unique_sha": len(rows),
        "actual_jobs_submitted": summary["actual_jobs_submitted"],
        "known_jobs": summary["known_jobs"],
        "safe_pre_fresh": 0,
        "fresh_2seed_required": False,
        "fresh_not_run_reason": "No candidate was both known-correct and strictly cheaper after official profiling.",
        "admitted": 0,
        "projected_total_gain": 0.0,
        "stale_cost_table_mismatches_corrected": stale,
        "mechanical_exact_or_noop_candidates": mechanical["candidate_count"],
        "runtime_crash_rejections": [CRASH_SHA],
        "private_fail_closed_tasks": [178, 169, 174, 325],
        "protected_root_artifacts_modified": [],
        "evidence": {
            "rescreen": str(RESCREEN.relative_to(ROOT)),
            "inventory_summary": str((inventory_dir / "summary.json").relative_to(ROOT)),
            "known_rejections": str((audit_dir / "known_rejections.json").relative_to(ROOT)),
            "task124_runtime_crash": str((audit_dir / "task124_runtime_crash.json").relative_to(ROOT)),
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

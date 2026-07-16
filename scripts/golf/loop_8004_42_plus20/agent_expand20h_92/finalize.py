#!/usr/bin/env python3
"""Assemble inventory, result, and fixed-adoption manifest."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def quad_complete(quad):
    return quad and all(v.get("right") and not v.get("wrong") and not v.get("errors") and not v.get("session_error") for v in quad.values())


def main() -> int:
    inventory_dir = HERE / "inventory"
    inventory_dir.mkdir(exist_ok=True)
    rescreen = json.loads((HERE / "rescreen.json").read_text())
    authority = json.loads((HERE / "authority_official_profiles.json").read_text())["tasks"]
    probe = json.loads((HERE / "probe_manifest.json").read_text())
    policy = json.loads((HERE / "audit/policy_reopen.json").read_text())
    known_all = json.loads((HERE / "audit/known_four_all_actual_lower.json").read_text())
    fresh_two_seed = json.loads((HERE / "audit/fresh_two_seed.json").read_text())
    classification = json.loads((HERE / "audit/probe_classification.json").read_text())
    rows = rescreen["rows"]

    per_task = []
    for task in rescreen["targets"]:
        task_rows = [row for row in rows if int(row["task"]) == task]
        costs = [int(row["actual_screen_cost"]) for row in task_rows if row.get("actual_screen_cost") is not None]
        per_task.append({
            "task": task,
            "authority_cost": int(authority[str(task)]["cost"]),
            "authority_sha256": authority[str(task)]["sha256"],
            "unique_nonauthority_sha": len(task_rows),
            "stage_counts": dict(Counter(row["stage"] for row in task_rows)),
            "actual_cost_results": len(costs),
            "minimum_actual_screen_cost": min(costs) if costs else None,
            "probe_count": sum(1 for item in probe["candidates"] if int(item["task"]) == task),
        })
    summary = {
        "baseline_zip": rescreen["baseline_zip"], "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "targets": rescreen["targets"], "unique_nonauthority_sha": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)), "per_task": per_task,
    }
    (inventory_dir / "raw.json").write_text(json.dumps(rescreen["inventory"], indent=2) + "\n")
    (inventory_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    policy_full = [item for item in policy["rows"] if item.get("known_four_result")]
    policy_known_complete = sum(quad_complete(item["known_four_result"].get("known_four")) for item in policy_full)
    best = {}
    for item in probe["candidates"]:
        task = int(item["task"])
        if task not in best or float(item["gain"]) > float(best[task]["gain"]):
            best[task] = item
    best_summary = [
        {"task": task, "sha256": item["sha256"], "candidate_cost": item["candidate_cost"], "gain": item["gain"]}
        for task, item in sorted(best.items())
    ]
    probe_ladder = []
    for task in sorted({int(item["task"]) for item in probe["candidates"]}):
        task_items = [item for item in probe["candidates"] if int(item["task"]) == task]
        task_items.sort(key=lambda item: (int(item["candidate_cost"]), -float(item["fresh_two_seed"]["minimum_mode_rate"]), item["sha256"]))
        probe_ladder.append({
            "task": task,
            "candidates": [
                {
                    "sha256": item["sha256"],
                    "candidate_cost": item["candidate_cost"],
                    "gain": item["gain"],
                    "minimum_two_seed_fresh_rate": item["fresh_two_seed"]["minimum_mode_rate"],
                    "probe_priority": item.get("probe_priority", "NORMAL"),
                }
                for item in task_items
            ],
            "safer_fallback_sha256": max(
                task_items,
                key=lambda item: (float(item["fresh_two_seed"]["minimum_mode_rate"]), int(item["candidate_cost"]), item["sha256"]),
            )["sha256"] if len(task_items) > 1 else None,
        })
    result = {
        "status": "LB_PROBE_REQUIRED_ONLY_NO_FIXED_ADOPTION",
        "baseline_zip": rescreen["baseline_zip"], "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "authority_retained_sources": [
            "submission.zip",
            "others/71403/lb_verified_8006.61/submission.zip",
        ],
        "scan_time_baseline_alias_note": "submission_base_8006.61.zip matched the authority SHA during the scan but drifted afterward; use an exact retained source",
        "target_count": len(rescreen["targets"]), "targets": rescreen["targets"],
        "inventory_unique_sha": len(rows),
        "ordinary_actual_jobs": sum(1 for row in rows if row["stage"] in {"actual_reject", "known_reject", "known_dual_reject", "shape_reject", "fresh500_reject", "fresh500_pass"}),
        "ordinary_actual_lower_known_four_jobs": known_all["result_count"],
        "ordinary_known_four_complete": known_all["complete_known_four_count"],
        "policy_total": probe["policy_total"], "policy_reason_reopened": probe["policy_reason_reopened"],
        "policy_static_lower_jobs": probe["policy_static_lower_jobs"], "policy_actual_lower_jobs": probe["policy_actual_lower_jobs"],
        "policy_known_four_complete": policy_known_complete,
        "pre_history_probe_count": probe.get("pre_history_count", probe["count"]),
        "probe_classification_counts": probe.get("classification_counts", {}),
        "known_lb_black_excluded": probe.get("known_lb_black_excluded", 0),
        "false_accept_excluded": probe.get("false_accept_excluded", 0),
        "lb_probe_required_count": probe["count"],
        "lb_probe_required_tasks": sorted({int(item["task"]) for item in probe["candidates"]}),
        "best_probe_per_task": best_summary,
        "probe_ladder_per_task": probe_ladder,
        "best_probe_projected_gain_sum_not_lb_verified": sum(float(item["gain"]) for item in best.values()),
        "fixed_adoption_count": 0, "fixed_projected_gain": 0.0,
        "fixed_champions_preserved": [13, 70, 158, 254, 267, 323, 379],
        "target_intersection_fixed_champions": [],
        "protected_root_artifacts_modified": [], "others_modified": False,
        "evidence": {
            "rescreen": str((HERE / "rescreen.json").relative_to(ROOT)),
            "probe_manifest": str((HERE / "probe_manifest.json").relative_to(ROOT)),
            "policy_reopen": str((HERE / "audit/policy_reopen.json").relative_to(ROOT)),
            "known_four_all": str((HERE / "audit/known_four_all_actual_lower.json").relative_to(ROOT)),
            "fresh_two_seed": str((HERE / "audit/fresh_two_seed.json").relative_to(ROOT)),
            "probe_classification": str((HERE / "audit/probe_classification.json").relative_to(ROOT)),
            "exact_sha_lb_history": str((HERE / "audit/lb_history_exact_sha.json").relative_to(ROOT)),
            "inventory_summary": str((inventory_dir / "summary.json").relative_to(ROOT)),
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({
        "status": "NO_FIXED_ADOPTION", "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"], "count": 0,
        "projected_total_gain": 0.0, "candidates": [],
        "lb_probe_required_manifest": str((HERE / "probe_manifest.json").relative_to(ROOT)),
        "lb_probe_required_count": probe["count"],
        "fixed_champions_preserved": [13, 70, 158, 254, 267, 323, 379],
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

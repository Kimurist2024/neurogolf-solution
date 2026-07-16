#!/usr/bin/env python3
"""Assemble the lane-94 machine-readable result and report."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"


def main() -> int:
    inventory = json.loads((HERE / "inventory_delta.json").read_text())
    screen = json.loads((HERE / "audit/incremental_screen.json").read_text())
    profiles = json.loads((HERE / "audit/official_reprofile_3x.json").read_text())
    probes = json.loads((HERE / "probe_manifest.json").read_text())
    repair = json.loads((HERE / "audit/task297_legal_repair_analysis.json").read_text())
    exact_history = []
    source_rows = []
    for row in screen["rows"]:
        task = int(row["task"])
        profile = row.get("official_profile")
        if str(task) in profiles["tasks"]:
            profile = profiles["tasks"][str(task)]["candidate_runs"][0]
            authority_profile = profiles["tasks"][str(task)]["authority_runs"][0]
        else:
            authority_profile = None
        candidate_cost = int(profile["cost"]) if profile else None
        authority_cost = int(authority_profile["cost"]) if authority_profile else int(row["authority_cost"])
        history = {
            "task": task,
            "sha256": row["sha256"],
            "searched_surfaces": [
                "docs/golf",
                "scripts/golf/loop_8004_42_plus20 JSON/Markdown outside this lane",
                "project memory handoff",
            ],
            "exact_sha_lb_record": None,
            "classification": "NO_EXACT_SHA_LB_RECORD_FOUND",
        }
        exact_history.append(history)
        source_rows.append({
            "task": task,
            "sha256": row["sha256"],
            "source": row["resolved_source"],
            "authority_cost": authority_cost,
            "candidate_cost": candidate_cost,
            "gain_if_other_gates_passed": (
                math.log(authority_cost / candidate_cost)
                if candidate_cost is not None and candidate_cost < authority_cost else 0.0
            ),
            "official_profile_3x_identical": (
                profiles["tasks"][str(task)]["candidate_profiles_identical"]
                if str(task) in profiles["tasks"] else None
            ),
            "known_four": row.get("known_four"),
            "truthful_shape": (
                not bool(row.get("runtime_shape_trace", {}).get("declared_actual_mismatches"))
                if row.get("runtime_shape_trace") is not None else False
            ),
            "structural_hard_failures": row["structural_audit"].get("hard_failures", []),
            "policy_markers": row["structural_audit"].get("policy_markers", []),
            "fresh_two_seed": row.get("fresh_two_seed"),
            "exact_sha_lb_history": history["classification"],
            "decision": row["classification"],
        })
    (HERE / "audit/exact_sha_lb_history.json").write_text(json.dumps({
        "rule": "exact SHA only; no task-level permanent blacklist",
        "query_result": "No incremental SHA has an exact LB-white or LB-black record outside this lane.",
        "rows": exact_history,
    }, indent=2) + "\n")

    result = {
        "status": "NO_FIXED_WINNER",
        "authority_zip": "submission.zip",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "baseline_lb_score": 8006.61,
        "target_count": len(screen["targets"]),
        "targets": screen["targets"],
        "predecessor_full_scan_sha_total": inventory["previous_unique_sha_total_for_targets"],
        "current_unique_nonauthority_sha_total": inventory["current_unique_nonauthority_sha_total"],
        "incremental_new_sha_total": inventory["incremental_new_sha_total"],
        "incremental_by_task": {
            task: row["incremental_new_sha"]
            for task, row in inventory["per_task"].items() if row["incremental_new_sha"]
        },
        "observations": inventory["counts"],
        "inventory_error_count": len(inventory["errors"]),
        "inventory_errors": inventory["errors"],
        "authority_rebase_all_target_members_byte_identical": True,
        "direct_black12_target_intersection": screen["direct_known_black_task_intersection"],
        "classification_counts": screen["classification_counts"],
        "fixed_winner_count": 0,
        "lb_probe_required_count": probes["count"],
        "projected_fixed_gain": 0.0,
        "source_candidates": source_rows,
        "task170_profile_correction": {
            "incorrect_rank_dir_or_single-shape_cost": 357,
            "competition_score_and_verify_authority_cost": 387,
            "all_scores_cost": 387,
            "candidate_competition_cost": 384,
            "candidate_decision": "HARD_REJECT_STRICT_SHAPE_INFERENCE",
        },
        "task297_pivot": repair,
        "task333_predecessor_note": {
            "dedicated_scan": "scripts/golf/loop_8004_42_plus20/agent_task333_finite81/candidate_inventory.json",
            "unique_sha_including_authority": 41,
            "incremental_new_sha": 0,
            "current_authority_cost": 423,
            "nonauthority_cost421_sha": "0628a573302f0a816d010482ed8b883caac7c307a27f47c9b53df85e2042a6bc",
            "disposition": "not rediscovered or fixed here; current 8006.61 authority does not contain it and no exact LB-white record was established",
        },
        "protected_root_artifacts_modified": [],
        "others_modified": False,
        "evidence": {
            "inventory_delta": str((HERE / "inventory_delta.json").relative_to(ROOT)),
            "incremental_screen": str((HERE / "audit/incremental_screen.json").relative_to(ROOT)),
            "official_reprofile_3x": str((HERE / "audit/official_reprofile_3x.json").relative_to(ROOT)),
            "authority_rebase": str((HERE / "audit/authority_rebase.json").relative_to(ROOT)),
            "exact_sha_lb_history": str((HERE / "audit/exact_sha_lb_history.json").relative_to(ROOT)),
            "task297_legal_repair": str((HERE / "audit/task297_legal_repair_analysis.json").relative_to(ROOT)),
            "probe_manifest": str((HERE / "probe_manifest.json").relative_to(ROOT)),
            "winner_manifest": str((HERE / "winner_manifest.json").relative_to(ROOT)),
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    table = []
    for row in source_rows:
        cost = "—" if row["candidate_cost"] is None else str(row["candidate_cost"])
        gain = f"{row['gain_if_other_gates_passed']:.6f}"
        table.append(
            f"| {row['task']:03d} | {row['authority_cost']} | {cost} | {gain} | `{row['sha256'][:12]}` | {row['decision']} |"
        )
    report = f"""# Expansion lane 94 — incremental 20-task audit

## Outcome

- Immutable authority: `submission.zip`, SHA-256 `{AUTHORITY_SHA256}` (LB 8006.61).
- Targets: {', '.join(f'{task:03d}' for task in screen['targets'])}.
- Fixed winners: **0**. LB probes: **0**. Protected root files and `others/` were not modified.
- The predecessor scans already covered **{inventory['previous_unique_sha_total_for_targets']}** task×SHA candidates. The current repository has **{inventory['current_unique_nonauthority_sha_total']}**, with only **{inventory['incremental_new_sha_total']}** genuinely new SHAs.

## Incremental inventory

The scan observed {inventory['counts'].get('loose_observations', 0):,} matching loose ONNX files and {inventory['counts'].get('zip_observations', 0):,} target members from {inventory['counts'].get('zip_files_seen', 0):,} ZIPs. New SHAs were one each for tasks 025/062/170/245/308/338/377. Task333 had no delta because its dedicated 41-SHA audit was used as a predecessor inventory.

One expected inventory diagnostic was recorded: `submission_base_8006.61.zip` is the known 68-byte drifted JSON, not a ZIP, and was never used as authority.

## Seven new candidates

| task | authority | candidate | nominal gain | SHA | decision |
|---:|---:|---:|---:|:---|:---|
{chr(10).join(table)}

Competition `score_and_verify` was repeated three times with independent labels/tempdirs for tasks 062/170/245/308/338; authority and candidate profiles were identical on every run. This corrects task170's old shape-dependent 357 diagnostic: the competition authority cost is 387 (matching `all_scores.csv`) and the candidate is 384, but the candidate still fails strict data-propagating shape inference.

## Why nothing is probeable

- task062 is known-perfect 267/267 in all four ORT/thread configurations, but three intermediates declare `[1,1,1,1]` while runtime is `[1,10,30,30]`; the 2-point shave is a shape cloak.
- task170 and task245 fail strict data-propagating ONNX shape inference.
- task308 is known-perfect only with optimizations disabled; default ORT cannot load its TopK graph.
- task338 gives the largest nominal reduction, 426→406, and is known-perfect with optimizations disabled. Default ORT rejects its output Concat shape, and the graph carries pervasive `[1,1,1,1]` declarations for runtime-spatial tensors. It is not a truthful/runtime-safe probe.
- task377 uses an unsupported TopK(uint8) path and cannot be official-profiled. task025 is equal-cost.

Fresh 2×500 was intentionally not run: no candidate cleared schema/UB, known×4, and truthful-shape gates. Fresh is a prioritizer, not permission to bypass those hard gates. Exact-SHA search found no LB record for any of the seven new SHAs.

## task297 legal-repair pivot

The known/fresh-perfect cost361 candidate relies on `Conv pads=[0,0,0,-24]`. Existing legal Slice and Split rewrites cost484 and511. A legal stride-1 Conv producing width6 needs a width25 kernel (+240 parameters), while higher stride samples spaced rather than contiguous columns. No schema-compliant repair at cost≤370 was found; the authority cost371 remains.

## Artifacts

- `inventory_delta.json`: full incremental SHA/source inventory.
- `audit/incremental_screen.json`: official/known×4/shape/UB decisions.
- `audit/official_reprofile_3x.json`: independent competition-profile repeats.
- `audit/exact_sha_lb_history.json`: exact-SHA history result.
- `audit/task297_legal_repair_analysis.json`: legal rewrite bounds.
- `result.json`, `probe_manifest.json`, `winner_manifest.json`: final disposition.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "status": result["status"],
        "incremental_new_sha_total": result["incremental_new_sha_total"],
        "classification_counts": result["classification_counts"],
        "lb_probe_required_count": result["lb_probe_required_count"],
        "fixed_winner_count": result["fixed_winner_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Finalize the non-promoting mid20d_88 audit."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    55, 31, 86, 88, 42, 143, 247, 79, 65, 344,
    115, 163, 206, 114, 273, 161, 71, 105, 259, 189,
)
PRIVATE_RISK = {86}
BASELINE_SCORE = 8005.17


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    rescreen = json.loads((HERE / "rescreen.json").read_text())
    audit = json.loads((HERE / "audit" / "actual_lower_four_config.json").read_text())
    costs = json.loads((HERE / "baseline_costs_8005_17.json").read_text())
    audit_by_sha = {row["sha256"]: row for row in audit["rows"]}
    rows_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rescreen["rows"]:
        rows_by_task[int(row["task"])].append(row)
    stage_counts = Counter(row["stage"] for row in rescreen["rows"])

    # The prior all-400 exact/no-op sweeps targeted 8005.16. Prove that their
    # conclusion transfers to every member in this target set.
    equivalence = []
    with zipfile.ZipFile(ROOT / "submission_base_8005.16.zip") as old, zipfile.ZipFile(
        ROOT / "submission_base_8005.17.zip"
    ) as new:
        for task in TARGETS:
            old_sha = hashlib.sha256(old.read(f"task{task:03d}.onnx")).hexdigest()
            new_sha = hashlib.sha256(new.read(f"task{task:03d}.onnx")).hexdigest()
            equivalence.append(
                {"task": task, "sha_8005_16": old_sha, "sha_8005_17": new_sha, "identical": old_sha == new_sha}
            )
    dump(
        HERE / "audit" / "baseline_8005_16_to_8005_17_equivalence.json",
        {"all_identical": all(row["identical"] for row in equivalence), "rows": equivalence},
    )

    task_rows = []
    candidates = []
    for task in TARGETS:
        rows = rows_by_task[task]
        actual_lower = [
            row
            for row in rows
            if row.get("actual_screen_cost") is not None
            and int(row["actual_screen_cost"]) < int(row["current_actual_cost"])
        ]
        audited = [audit_by_sha[row["sha256"]] for row in actual_lower]
        known4 = [row for row in audited if row["known_perfect_all_configs"]]
        truthful = [
            row for row in known4 if (row.get("runtime_shape_trace") or {}).get("truthful", False)
        ]
        if known4 and not truthful:
            reason = "reject_runtime_shape_cloak_or_untraceable_graph"
        elif actual_lower:
            reason = "reject_known_or_runtime_gate"
        else:
            reason = "no_policy_clean_actual_lower_candidate"
        task_rows.append(
            {
                "task": task,
                "baseline_cost": int(costs["costs"][str(task)]),
                "unique_nonbaseline_sha": int(rescreen["inventory"]["unique_by_task"][str(task)]),
                "actual_lower_count": len(actual_lower),
                "best_actual_lower_cost": min(
                    (int(row["actual_screen_cost"]) for row in actual_lower), default=None
                ),
                "known_perfect_four_configs_count": len(known4),
                "truthful_runtime_shape_count": len(truthful),
                "private_catalog": task in PRIVATE_RISK,
                "decision": "reject",
                "reason": reason,
            }
        )
        for row in audited:
            candidates.append(
                {
                    "task": task,
                    "sha256": row["sha256"],
                    "baseline_cost": row["baseline_cost"],
                    "actual_cost": row["actual_cost"],
                    "known_perfect_four_configs": row["known_perfect_all_configs"],
                    "truthful_runtime_shapes": bool(
                        (row.get("runtime_shape_trace") or {}).get("truthful", False)
                    ),
                    "runtime_shape_mismatch_count": (
                        (row.get("runtime_shape_trace") or {}).get("mismatch_count")
                    ),
                    "runtime_shape_trace_error": row.get("runtime_shape_trace_error"),
                    "static_reasons": row["static"].get("reasons", []),
                    "conv_bias_ub0": row["static"].get("conv_bias_ub0", False),
                    "lookup": row["static"].get("lookup", False),
                    "giant_einsum": row["static"].get("giant_einsum", False),
                    "accepted": False,
                }
            )

    actual_jobs = sum(
        count
        for stage, count in stage_counts.items()
        if stage in {
            "actual_reject", "known_reject", "known_dual_reject", "shape_reject",
            "fresh500_reject", "fresh500_pass",
        }
    )
    inventory_output = {
        "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "targets": list(TARGETS),
        "observation_counts": rescreen["inventory"]["counts"],
        "unique_by_task": rescreen["inventory"]["unique_by_task"],
        "stage_counts": dict(sorted(stage_counts.items())),
        "baseline_profiles": costs["profiles"],
        "task_summary": task_rows,
    }
    dump(HERE / "inventory" / "candidate_inventory.json", inventory_output)
    dump(
        HERE / "audit" / "final_decisions.json",
        {
            "policy": {
                "normal_fresh": "two independent seeds, each >=90%; reached only after truthful known-perfect gate",
                "private_or_giant": "complete finite support or all-input algebraic proof only",
                "known_configs": [
                    "disable_all_threads1", "disable_all_threads4",
                    "default_threads1", "default_threads4",
                ],
                "runtime_errors_allowed": 0,
                "shape_cloak_allowed": False,
                "conv_bias_ub_allowed": False,
            },
            "candidate_count": len(candidates),
            "accepted_count": 0,
            "fresh_two_seed_candidates": 0,
            "fresh_skip_reason": "no candidate passed known×4 plus truthful runtime shapes",
            "candidates": candidates,
            "task_decisions": task_rows,
        },
    )
    result = {
        "baseline_score": BASELINE_SCORE,
        "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "targets": list(TARGETS),
        "funnel": {
            "unique_nonbaseline_sha": int(rescreen["inventory"]["counts"]["unique_different_sha"]),
            "actual_profile_jobs": actual_jobs,
            "actual_strict_lower": audit["candidate_count"],
            "known_perfect_four_configs": audit["known_perfect_four_configs_count"],
            "truthful_runtime_shapes": audit["truthful_count"],
            "fresh_two_seed_candidates": 0,
            "safe_adoptees": 0,
        },
        "accepted": [],
        "score_gain": 0.0,
        "projected_score": BASELINE_SCORE,
        "promotion_zip_created": False,
        "protected_root_files_changed": False,
        "status": "complete_no_safe_adoptee",
    }
    dump(HERE / "result.json", result)
    dump(
        HERE / "winner_manifest.json",
        {
            "baseline_zip": rescreen["baseline_zip"],
            "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
            "winners": [],
            "total_gain": 0.0,
            "promotion": "none",
            "reason": "no candidate passed every fail-closed gate",
        },
    )

    table = []
    for row in task_rows:
        best = "—" if row["best_actual_lower_cost"] is None else str(row["best_actual_lower_cost"])
        table.append(
            f"| {row['task']:03d} | {row['baseline_cost']} | {row['unique_nonbaseline_sha']} | "
            f"{row['actual_lower_count']} | {best} | {row['known_perfect_four_configs_count']} | "
            f"{row['truthful_runtime_shape_count']} | reject |"
        )
    report = f"""# Expanded mid-cost 20-task audit — mid20d_88

## Result

- Authority: `submission_base_8005.17.zip`
- Authority SHA-256: `{rescreen['baseline_zip_sha256']}`
- Targets: {', '.join(f'{task:03d}' for task in TARGETS)}
- Safe adoptees: **0**
- Score gain: **+0.0**; projected score remains **{BASELINE_SCORE:.2f}**
- Promotion ZIP: not created; protected root files were not changed.

## Candidate funnel

- Scanned 12,030 loose ONNX observations and 23,671 members from 1,259 ZIP files.
- After baseline duplicate/oversize filtering, 803 distinct nonbaseline SHAs remained.
- Fail-closed stages: 176 structure rejects, 6 policy rejects, 567 static-floor rejects, and {actual_jobs} actual-profile jobs.
- Actual strict-lower: 28; known-perfect in disabled/default ORT × 1/4 threads: 22; runtime-shape truthful: 0; safe: 0.
- Fresh two-seed testing was not started because no model passed the earlier known×4 and truthful-shape gates.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | decision |
|---:|---:|---:|---:|---:|---:|---:|:---|
{chr(10).join(table)}

## Final blockers

- **task088:** 26 actual-lower candidates were found. Twenty-two pass all 267 known examples in all four ORT/thread configurations, but 21 have 11–18 direct declared/actual runtime-shape mismatches. The remaining cost-211 graph cannot be truthfully traced because it contains duplicate node name `label_scale`. All are rejected as shape-cloak/schema-invalid artifacts. The cheapest task088 leads that are not known-perfect score 0/267, 23/267, or fail optimized ORT.
- **task071:** cost 188→186 misses 1 of 265 known cases in every configuration. The retained exact CastLike rewrite is also covered by the repository scan and was previously rejected for giant-Einsum/default-vs-disable instability.
- **task161:** cost 190→186 misses 1 of 266 known cases in every configuration.
- The other 17 targets have no policy-clean actual-lower survivor. task086 is private-catalog lineage; no percentage-only artifact was admitted. Lookup, giant-Einsum, Conv-bias UB, schema-invalid, and private-lineage artifacts were fail-closed before fresh evaluation.

## Exact initializer/dead/no-op coverage

All 20 authority members are byte-identical between 8005.16 and 8005.17, so the prior full-400 exact initializer/Einsum and dead/no-op sweeps remain applicable. The repository-wide inventory also directly included their emitted artifacts, including task163 latent prunes and task071 CastLike. None cleared current known×4, truthful-shape, and no-giant gates. See `audit/baseline_8005_16_to_8005_17_equivalence.json`.

## Artifacts

- `inventory/candidate_inventory.json`: task-level SHA inventory and funnel.
- `rescreen.json`: all 803 unique candidate decisions.
- `audit/actual_lower_four_config.json`: 28-candidate known×4/static/runtime-shape evidence.
- `audit/final_decisions.json`: final fail-closed classifications.
- `result.json` and `winner_manifest.json`: zero-promotion result.
"""
    (HERE / "REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

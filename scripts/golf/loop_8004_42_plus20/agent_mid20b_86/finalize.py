#!/usr/bin/env python3
"""Materialize the non-promoting final report and manifests for mid20b_86."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TARGETS = (
    102, 25, 324, 308, 338, 134, 268, 184, 377, 170,
    239, 222, 48, 234, 264, 200, 387, 132, 388, 228,
)
PRIVATE_RISK = {48, 102, 134, 170, 222, 264, 377}
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
    task_rows = []
    final_candidates = []
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
            row
            for row in known4
            if (row.get("runtime_shape_trace") or {}).get("truthful", False)
        ]
        if task == 48 and truthful:
            final_reason = (
                "reject_private_guarantee: seven cost378 candidates are 457/500 "
                "on legal fresh data (43 counterexamples), so private-zero 100% guarantee is absent"
            )
        elif known4 and not truthful:
            final_reason = "reject_runtime_shape_cloak"
        elif actual_lower:
            final_reason = "reject_known_or_runtime_gate"
        else:
            final_reason = "no_policy_clean_actual_lower_candidate"
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
                "private_catalog_or_black_history": task in PRIVATE_RISK,
                "decision": "reject",
                "reason": final_reason,
            }
        )
        for row in audited:
            rescreen_row = next(item for item in actual_lower if item["sha256"] == row["sha256"])
            fresh = rescreen_row.get("fresh_dual")
            final_candidates.append(
                {
                    "task": task,
                    "sha256": row["sha256"],
                    "baseline_cost": row["baseline_cost"],
                    "actual_cost": row["actual_cost"],
                    "known_perfect_four_configs": row["known_perfect_all_configs"],
                    "truthful_runtime_shapes": bool(
                        (row.get("runtime_shape_trace") or {}).get("truthful", False)
                    ),
                    "static_reasons": row["static"].get("reasons", []),
                    "conv_bias_ub0": row["static"].get("conv_bias_ub0", False),
                    "lookup": row["static"].get("lookup", False),
                    "giant_einsum": row["static"].get("giant_einsum", False),
                    "fresh500": fresh,
                    "private_risk": task in PRIVATE_RISK,
                    "accepted": False,
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
                "normal_fresh_threshold": 0.90,
                "private_or_giant_requirement": "complete finite support or all-input algebraic proof",
                "known_configs": [
                    "disable_all_threads1",
                    "disable_all_threads4",
                    "default_threads1",
                    "default_threads4",
                ],
                "runtime_errors_allowed": 0,
                "shape_cloak_allowed": False,
                "conv_bias_ub_allowed": False,
            },
            "candidate_count": len(final_candidates),
            "accepted_count": 0,
            "candidates": final_candidates,
            "task_decisions": task_rows,
        },
    )
    result = {
        "baseline_score": BASELINE_SCORE,
        "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "target_count": len(TARGETS),
        "targets": list(TARGETS),
        "funnel": {
            "unique_nonbaseline_sha": int(rescreen["inventory"]["counts"]["unique_different_sha"]),
            "static_actual_jobs": 98,
            "actual_strict_lower": audit["candidate_count"],
            "known_perfect_four_configs": audit["known_perfect_four_configs_count"],
            "truthful_runtime_shapes": audit["truthful_count"],
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
    report = f"""# Expanded mid-cost 20-task audit — mid20b_86

## Result

- Authority: `submission_base_8005.17.zip`
- Authority SHA-256: `{rescreen['baseline_zip_sha256']}`
- Targets: {', '.join(f'{task:03d}' for task in TARGETS)}
- Safe adoptees: **0**
- Score gain: **+0.0**; projected score remains **{BASELINE_SCORE:.2f}**
- Promotion ZIP: not created; protected root files were not changed.

## Exhaustive candidate funnel

- Scanned 12,479 loose ONNX observations and 23,666 task members from 1,259 ZIP files.
- After 4,077 baseline duplicates and 118 oversize observations, 1,140 distinct nonbaseline SHAs remained.
- Fail-closed screen: 319 structure rejects, 58 policy rejects, 665 static-floor rejects, and 98 actual-profile jobs.
- Actual strict-lower candidates: 24. Four-configuration known-perfect: 8. Runtime-shape truthful: 7. Final safe: 0.
- Every actual-lower model was checked on known train/test/arc-gen under disabled/default ORT and 1/4 threads, with runtime/nonfinite/output-shape evidence recorded in `audit/actual_lower_four_config.json`.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | decision |
|---:|---:|---:|---:|---:|---:|---:|:---|
{chr(10).join(table)}

## Final blockers

- **task048:** seven cost 378 candidates pass all 270 known examples in all four ORT/thread configurations, strict/data-prop, runtime-shape truth, finite/margin, and Conv-bias UB0. However all score only 457/500 (91.4%) on legal fresh inputs, with the first counterexample at case 11. task048 is in the private-zero catalog; because legal counterexamples exist, no all-input guarantee is possible and the user's private-zero exception does not apply.
- **task134:** cost 423→412 is known-perfect in all four configurations, but direct unsanitized tracing finds six declared/actual shape mismatches (for example `[1,2,1,1]` declared versus `[1,10,30,30]` actual). Rejected as shape cloak.
- **Other 16 actual-lower models:** fail known correctness or an ORT/runtime gate. Notable examples are task102 cost 491 and task377 cost 408 failing optimized ORT, task200 cost 344 scoring 0/84 in all four configurations, and task228 cost 294 failing known.
- Lookup/private-lineage/giant-Einsum/Conv-bias candidates were rejected before fresh scoring. Private/giant artifacts were never admitted from a percentage score alone.

## Exact mechanical coverage

The repository-wide inventory includes retained exact initializer-fusion, outer-product, singleton-axis, dead/no-op, and reuse candidates. The only exact-family strict-lower survivors in this set were the task048 cost-378 variants, which have reproducible legal counterexamples. No initializer-alias/dead-operand artifact in the 20-task set cleared all gates.

## Artifacts

- `inventory/candidate_inventory.json`: target-by-target SHA counts, costs, and funnel summary.
- `rescreen.json`: all 1,140 unique candidates and their stage/reason.
- `audit/actual_lower_four_config.json`: full 24-candidate known×4/static/shape evidence.
- `audit/final_decisions.json`: fail-closed decision record.
- `result.json` and `winner_manifest.json`: zero-promotion result.
"""
    (HERE / "REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

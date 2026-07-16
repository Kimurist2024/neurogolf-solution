#!/usr/bin/env python3
"""Consolidate the fail-closed low42 audit into final lane artifacts."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (339, 126, 21, 171, 346, 227, 318, 332)

FINDINGS = {
    339: {
        "lead": "truthful cost-53 incumbent; alternate history floors 53, 59, and 140",
        "reason": "The complete archive has no below-53 graph and the exact rewrite scan found no alias, dead code, no-op, duplicate producer, or optional output. The compact true-rule graph already spends only 19 parameters and 34 bytes of declared intermediates; its count/color decode and int8 ramp are all live.",
    },
    126: {
        "lead": "history floors 61, 64, 66, 73, 79, 90, and 605 versus cost 52",
        "reason": "No numeric lower lead exists. The incumbent has 19 declared/runtime shape contradictions plus GatherND/ScatterElements, so it cannot seed a truthful candidate; conventional true-rule history is already more expensive.",
    },
    21: {
        "lead": "truthful cost-51 incumbent; only different harvested graph costs 177",
        "reason": "No below-51 graph exists in the complete archive. The exact scan found no mechanical rewrite; both Conv/Einsum size probes, reciprocal coordinates, and the four-value RoiAlign descriptor are live.",
    },
    171: {
        "lead": "history floors 78 and 414 versus cost 50",
        "reason": "No numeric lower lead exists. The incumbent has eight runtime-shape contradictions and fails every known case in default ORT at session/runtime (0/54, 54 errors), so no safe derivation is possible.",
    },
    346: {
        "lead": "history floors 92, 109, 111, and 1254 versus cost 50",
        "reason": "No below-50 graph exists. This task is in the explicit private-zero catalog and the incumbent has six runtime-shape contradictions; the guarantee exception therefore cannot apply to a derived dust rewrite.",
    },
    227: {
        "lead": "history floors 72 and 100 versus cost 49",
        "reason": "No below-49 graph exists. The compact QLinearConv chain has a declared/runtime GroupNormalization shape contradiction, and the exact scan found no safe structural reduction.",
    },
    318: {
        "lead": "history floors 72 and 100 versus cost 49",
        "reason": "No below-49 graph exists. The compact QLinearConv chain has a declared/runtime GroupNormalization shape contradiction, and the exact scan found no safe structural reduction.",
    },
    332: {
        "lead": "cost-49 historical tie and cost-80 alternate",
        "reason": "Only a tie exists. The three-node incumbent has three runtime-shape contradictions, while the 41 live parameters already dominate its declared cost; no truthful strictly-lower graph appears in history or exact rewrites.",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    exact = json.loads((HERE / "scan_report.json").read_text())
    baseline_by_task = {row["task"]: row for row in baseline["targets"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}
    known_by_task = {row["task"]: row for row in known["rows"]}

    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    retained = [
        row
        for task_rows in archive["retained"].values()
        for row in task_rows
        if row.get("task") in TARGETS
    ]
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    harvest_rows = [row for row in harvest["rows"] if row.get("task") in TARGETS]
    history = {
        "baseline": baseline["baseline"],
        "archive_inventory": {
            "stats": archive["stats"],
            "retained_numeric_lower_leads": retained,
            "interpretation": "The 13,591-unique-graph complete archive retains no below-baseline candidate for any low42 target.",
        },
        "focused_harvest": {
            "inventory": harvest["inventory"]["counts"],
            "target_rows": harvest_rows,
            "interpretation": "Every different harvested graph is a tie or more expensive than the immutable baseline.",
        },
        "narrow_exact_scan": {
            "summary": exact["summary"],
            "records": exact["records"],
            "interpretation": "No exact structural rewrite opportunity exists for the eight baselines.",
        },
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")

    rows = []
    for task in TARGETS:
        base = baseline_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shape_trace"]
        rows.append({
            "task": task,
            "rule_summary": rule_by_task[task]["rule_summary"],
            "true_rule_known": rule_by_task[task]["known"],
            "baseline_member": base["member"],
            "baseline_sha256": base["sha256"],
            "baseline_cost": base["actual_cost"],
            "unchanged_from_8004_50": base["unchanged_from_8004_50"],
            "baseline_known_dual": known_by_task[task],
            "private_zero_catalog": task == 346,
            "structure_observation": {
                "ops": structure["op_histogram"],
                "node_count": structure["node_count"],
                "max_node_inputs": structure["max_node_inputs"],
                "giant_einsum": structure["giant_einsum"],
                "lookup_or_scatter": structure["lookup_or_scatter_nodes"],
                "runtime_shape_cloak": runtime.get("shape_cloak", True),
                "runtime_shape_mismatches": runtime.get("mismatch_count"),
                "checker_full": structure["checker_full"],
                "strict_shape_data_prop": structure["strict_shape_data_prop"],
                "standard_domains": structure["standard_domains"],
                "conv_bias_findings": structure["conv_bias_findings"],
            },
            "best_probe": FINDINGS[task]["lead"],
            "reason": FINDINGS[task]["reason"],
            "candidate": None,
            "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
            "candidate_known_dual": "NOT_RUN_NO_NUMERIC_LOWER_CANDIDATE",
            "candidate_fresh_seed_1_dual": "NOT_RUN_NO_NUMERIC_LOWER_CANDIDATE",
            "candidate_fresh_seed_2_dual": "NOT_RUN_NO_NUMERIC_LOWER_CANDIDATE",
            "candidate_runtime_errors": "NO_CANDIDATE",
            "candidate_conv_bias_ub0": "NO_CANDIDATE",
            "projected_gain": 0.0,
        })

    result = {
        "lane": "agent_new_low42",
        "baseline": baseline["baseline"],
        "policy": {
            "normal": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_zero": "decoded true-rule guarantee or exact LB-white lineage plus known/fresh100 dual and every normal structural/runtime gate",
            "fail_closed": "fresh is not run when no strict-lower candidate passes the price and structure pre-gates",
        },
        "targets_requested": list(TARGETS),
        "targets_completed": len(rows),
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "zip_integration": False,
        "rows": rows,
        "final_verdict": "NO_SAFE_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({
        "lane": "agent_new_low42",
        "baseline_sha256": baseline["baseline"]["sha256"],
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

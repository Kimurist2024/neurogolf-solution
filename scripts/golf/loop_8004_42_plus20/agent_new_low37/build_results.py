#!/usr/bin/env python3
"""Consolidate the fail-closed low37 audit into final lane artifacts."""

from __future__ import annotations

import collections
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (320, 154, 393, 290, 336, 3, 58, 72)

BEST_PROBES = {
    320: {
        "lead": "no graph below cost 80; one conventional truthful history model has static floor 770",
        "reason": "The incumbent is already an output-only 14-operand rank-two Einsum with only 80 parameters. Complete archive, focused harvest, and exact-rewrite scans expose no strict decrease.",
    },
    154: {
        "lead": "44/48-operand giant-Einsum alternates and a conventional static-floor 2679 model",
        "reason": "Every compact lineage is a prohibited giant contraction; the only truthful conventional lineage is far above the cost-88 floor.",
    },
    393: {
        "lead": "latest cost 86; historical clean/alternate floors are 95, 117, or 121",
        "reason": "The current 8004.50-era one-parameter shave is already present. It has three runtime shape contradictions and default TopK session failure; truthfully computing color counts and ranking exceeds cost 86.",
    },
    290: {
        "lead": "archive static floors 73/75/88",
        "reason": "Runtime profiling raises them to costs 91/93/97. The cost-73 lead is exact-known/fresh-correct but only ties the latest actual cost 91, so there is no strict decrease; all inherit shape-cloak behavior.",
    },
    336: {
        "lead": "no below-92 safe graph; conventional generator-compiled history is cost 4746",
        "reason": "The incumbent is a prohibited 42-operand output-only Einsum. The harvested conventional model has a Conv bias-shape UB finding and is orders of magnitude above the floor.",
    },
    3: {
        "lead": "one cost-78 same-cost lineage and a conventional static-floor 260 model",
        "reason": "The only compact alternate ties. The incumbent itself relies on two runtime shape contradictions, while making those tensors truthful destroys the 78-cost floor.",
    },
    58: {
        "lead": "historical static floors 86, 94, and 213 versus cost 78",
        "reason": "No history member is strictly lower. The incumbent is already a single output-only 16-operand contraction with 78 parameters.",
    },
    72: {
        "lead": "one cost-78 same-cost lineage; other history floors are 120 and 180",
        "reason": "The tie cannot promote and the incumbent has two runtime shape contradictions. A truthful XOR implementation exceeds the 78-cost floor.",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    base_by_task = {row["task"]: row for row in baseline["targets"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}

    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    retained = [
        row
        for task_rows in archive["retained"].values()
        for row in task_rows
        if row.get("task") in TARGETS
    ]
    quick = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/lower_quick_k20.json").read_text()
    )
    quick_relevant = [row for row in quick if row.get("task") in TARGETS]
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    harvest_relevant = [row for row in harvest["rows"] if row.get("task") in TARGETS]
    history = {
        "baseline": baseline["baseline"],
        "rebase": {
            "all_members_unchanged_from_8004_50": all(
                row["unchanged_from_8004_50"] for row in baseline["targets"]
            ),
            "note": "The exhaustive archive used 7999.13 costs; latest actual costs in baseline_audit.json are authoritative (notably task393 is now 86 rather than 87).",
        },
        "archive_inventory": {
            "stats": archive["stats"],
            "retained_target_lower_static_leads": retained,
            "quick_runtime_screens": quick_relevant,
        },
        "focused_harvest": {
            "inventory": harvest["inventory"]["counts"],
            "target_rows": harvest_relevant,
            "per_task_stage_counts": {
                str(task): dict(
                    collections.Counter(
                        row.get("stage") for row in harvest_relevant if row.get("task") == task
                    )
                )
                for task in TARGETS
            },
        },
        "exact_wave2": baseline["exact_wave2"],
        "other_exact_scans": {
            "task393_opset17": "REJECT: GroupNormalization has no opset-17 schema; axes initializer is also reused to produce TopK k=3.",
            "all_target_hits": "No accepted target hit in complete exact-Wave2, optional-default, no-op, fold-shape, or initializer-alias scans.",
        },
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")

    rows = []
    for task in TARGETS:
        base = base_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shape_trace"]
        probe = BEST_PROBES[task]
        rows.append(
            {
                "task": task,
                "rule_summary": rule_by_task[task]["rule_summary"],
                "true_rule_known": rule_by_task[task]["known"],
                "baseline_member": base["member"],
                "baseline_sha256": base["sha256"],
                "baseline_cost": base["actual_cost"],
                "unchanged_from_8004_50": base["unchanged_from_8004_50"],
                "baseline_known_dual": known[str(task)],
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
                "best_probe": probe["lead"],
                "reason": probe["reason"],
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "candidate_known_dual": "NOT_RUN_PRE_GATE_FAILED",
                "candidate_fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
                "candidate_fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
                "candidate_runtime_errors": "NO_CANDIDATE",
                "candidate_conv_bias_ub0": "NO_CANDIDATE",
                "projected_gain": 0.0,
            }
        )

    result = {
        "lane": "agent_new_low37",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful runtime shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and require decoded true-rule guarantee or exact LB-white lineage",
            "fail_closed": "known/fresh candidate runs do not start when cost or structure gates fail",
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
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "lane": "agent_new_low37",
                "baseline_sha256": baseline["baseline"]["sha256"],
                "accepted": [],
                "accepted_count": 0,
                "projected_gain": 0.0,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

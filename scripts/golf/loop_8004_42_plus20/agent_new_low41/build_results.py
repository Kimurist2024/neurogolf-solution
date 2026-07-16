#!/usr/bin/env python3
"""Consolidate the fail-closed low41 audit into final lane artifacts."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (380, 242, 298, 26, 261, 351, 274, 317)

BEST_PROBES = {
    380: {
        "lead": "truthful cost-60 output-only 9-input Einsum incumbent; best different history floor 99",
        "reason": "No below-60 graph exists in the 13,591-graph archive or 1,134-graph focused harvest. Exact Wave2 found no alias, dead-code, no-op, producer, optional-output, or annotation rewrite. The initializer has shape 30x2; removing its zero rows breaks the Einsum dimension contract and materializing a pad/crop adds a counted intermediate.",
    },
    242: {
        "lead": "cost-58 tie and cost-64/422 alternates",
        "reason": "No strict decrease exists in the complete history. The incumbent itself has seven declared/runtime shape contradictions, so it is not a safe structure from which to derive a candidate.",
    },
    298: {
        "lead": "best different history floor 59 versus incumbent 58",
        "reason": "All different harvested graphs cost at least 59. The incumbent has a runtime shape cloak, and the exact rewrite pass emits no candidate.",
    },
    26: {
        "lead": "best different history floor 58 versus incumbent 57",
        "reason": "All different harvested graphs cost at least 58. The QLinearConv/ConvInteger incumbent relies on a declared/runtime shape contradiction, and no safe exact rewrite exists.",
    },
    261: {
        "lead": "best different history floor 59 versus incumbent 57",
        "reason": "All different harvested graphs cost at least 59. The incumbent has eight runtime shape contradictions and an allocator-sensitive CenterCropPad chain.",
    },
    351: {
        "lead": "cost-57 revalidated tie; other history floors 75 and 995",
        "reason": "Only a tie exists. The incumbent has ten runtime shape contradictions; no truthful strictly-cheaper graph appears in history or exact rewrites.",
    },
    274: {
        "lead": "best different history floor 64 versus incumbent 55",
        "reason": "The incumbent uses five GatherND nodes and two runtime shape contradictions. No below-55 graph exists, and the private-risk safety gate forbids lookup/cloak lineage.",
    },
    317: {
        "lead": "cost-55 rebuilt tie; other history floor 146",
        "reason": "Only a tie exists. The incumbent uses nondeterministic Bernoulli plus two runtime shape contradictions, so it cannot seed a safe lower candidate.",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    baseline_by_task = {row["task"]: row for row in baseline["targets"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}

    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    archive_relevant = [
        row
        for task_rows in archive["retained"].values()
        for row in task_rows
        if row.get("task") in TARGETS
    ]
    harvest_relevant = [row for row in harvest["rows"] if row.get("task") in TARGETS]

    history = {
        "baseline": baseline["baseline"],
        "archive_inventory": {
            "stats": archive["stats"],
            "relevant_retained_lower": archive_relevant,
            "interpretation": "The complete archive retains no below-baseline candidate for any low41 target.",
        },
        "focused_harvest": {
            "inventory": harvest["inventory"]["counts"],
            "relevant_rows": harvest_relevant,
            "interpretation": "All different low41 graphs are ties or more expensive than the immutable baseline.",
        },
        "exact_wave2": baseline["exact_wave2"],
    }
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")

    rows = []
    for task in TARGETS:
        base = baseline_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shape_trace"]
        rule = rule_by_task[task]
        probe = BEST_PROBES[task]
        rows.append({
            "task": task,
            "rule_summary": rule["rule_summary"],
            "true_rule_known": rule["known"],
            "baseline_member": base["member"],
            "baseline_sha256": base["sha256"],
            "baseline_cost": base["actual_cost"],
            "unchanged_from_8004_50": base["unchanged_from_8004_50"],
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
            "known_dual": "NOT_RUN_PRE_GATE_FAILED",
            "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
            "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
            "runtime_errors": "NO_CANDIDATE",
            "conv_bias_ub0": "NO_CANDIDATE",
            "projected_gain": 0.0,
        })

    result = {
        "lane": "agent_new_low41",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must be demonstrably SOUND",
            "fail_closed": "known/fresh are not run if price or structural gates fail",
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
        "lane": "agent_new_low41",
        "baseline_sha256": baseline["baseline"]["sha256"],
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

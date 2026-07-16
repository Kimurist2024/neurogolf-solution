#!/usr/bin/env python3
"""Consolidate the fail-closed low35 audit into final lane artifacts."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (50, 329, 350, 356, 371, 360, 214, 83)

BEST_PROBES = {
    50: {
        "lead": "four cost-84 common-transition factorizations",
        "reason": "All four fail train[0] in both ORT modes. The dedicated scan covered 566 files / 18 unique task050 graphs and found no other below-88 model.",
        "evidence": "scripts/golf/loop_7999_13/lane_c30/audit.json",
    },
    329: {
        "lead": "no below-88 graph; cost-88 incumbent and cost-1050 conventional control",
        "reason": "The 13,591-graph archive found no lower member. A harvested alternate increases Einsum fan-in from 15 to 19 and still offers no established lower actual cost.",
        "evidence": "scripts/golf/loop_7999_13/lane_harvest/scan_results.json",
    },
    350: {
        "lead": "cost-60 rank-one initializer approximation",
        "reason": "It is not an exact factorization (max reconstruction error 32) and scores 0/100 in both ORT modes, with 100/100 runtime/output failures. The complete archive screen also gives known/fresh 0%.",
        "evidence": "scripts/golf/loop_7999_13/lane_rank1_wave13/dual100.json",
    },
    356: {
        "lead": "cost-60 rank-one initializer approximation",
        "reason": "It is not an exact factorization (max reconstruction error 32) and scores 0/100 in both ORT modes, with 100/100 runtime/output failures. The complete archive screen also gives known/fresh 0%.",
        "evidence": "scripts/golf/loop_7999_13/lane_rank1_wave13/dual100.json",
    },
    371: {
        "lead": "cost-88 same-cost giant-Einsum alternates",
        "reason": "Every compact alternate only ties and retains 20-input Einsum lineage; the conventional harvested graph has a static floor of 467. No strict decrease exists.",
        "evidence": "scripts/golf/loop_7999_13/lane_harvest/scan_results.json",
    },
    360: {
        "lead": "cost-86 incumbent; conventional harvested graph has static floor 340",
        "reason": "The exhaustive archive and focused harvest expose no below-86 graph, while exact rewrites produced no candidate.",
        "evidence": "scripts/golf/loop_7999_13/lane_harvest/scan_results.json",
    },
    214: {
        "lead": "static cost-75 archive graph",
        "reason": "It fails both known gold implementations and all 20 fresh cases. Its actual scorer cost is unavailable, so it cannot establish a safe strict decrease; Identity removal also fails strict inference.",
        "evidence": "scripts/golf/loop_7999_13/lane_archive_all400/lower_quick_k20.json",
    },
    83: {
        "lead": "giant-Einsum alternate with 49 inputs; conventional graph static floor 210",
        "reason": "The alternate is structurally forbidden and has no established lower actual cost; all valid history is cost >=84. Exact rewrites produced no candidate.",
        "evidence": "scripts/golf/loop_7999_13/lane_harvest/scan_results.json",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    rule_audit = json.loads((HERE / "true_rule_audit.json").read_text())
    baseline_by_task = {row["task"]: row for row in baseline["targets"]}
    rule_by_task = {row["task"]: row for row in rule_audit["rows"]}

    archive = json.loads(
        (
            ROOT
            / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json"
        ).read_text()
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
    harvest_relevant = [
        row for row in harvest["rows"] if row.get("task") in TARGETS
    ]

    history = {
        "baseline": baseline["baseline"],
        "archive_inventory": {
            "stats": archive["stats"],
            "relevant_retained_lower": archive_relevant,
            "interpretation": "Only tasks 214, 350 and 356 have below-baseline static leads; every one fails its prerequisite correctness screen.",
        },
        "focused_harvest": {
            "inventory": harvest["inventory"]["counts"],
            "relevant_rows": harvest_relevant,
        },
        "task050_dedicated": {
            "files": 566,
            "unique_models": 18,
            "below_baseline_probes": 4,
            "probe_cost": 84,
            "known_result": "all four fail train[0] in both ORT modes",
            "evidence": "scripts/golf/loop_7999_13/lane_c30/audit.json",
        },
        "rank_one_rejections": {
            "tasks": [350, 356],
            "cost": 60,
            "dual_result_each": "0/100 in ORT_DISABLE_ALL and 0/100 default; runtime_or_output_failures=100/100",
            "evidence": "scripts/golf/loop_7999_13/lane_rank1_wave13/dual100.json",
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
        rows.append(
            {
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
                "evidence": probe["evidence"],
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "known_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
                "runtime_errors": "NO_CANDIDATE",
                "conv_bias_ub0": "NO_CANDIDATE",
                "projected_gain": 0.0,
            }
        )

    result = {
        "lane": "agent_new_low35",
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
    (HERE / "winner_manifest.json").write_text(
        json.dumps(
            {
                "lane": "agent_new_low35",
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

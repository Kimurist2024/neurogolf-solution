#!/usr/bin/env python3
"""Consolidate the completed low43 search into fail-closed artifacts."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGETS = (6, 334, 244, 249, 347, 386, 146, 291)

PROBES = {
    6: {
        "lead": "actual-cost 30/38/40/40 history candidates",
        "reason": "All four strict numeric leads fail complete-known dual validation: 0/266, runtime failure on 266/266, runtime failure on 266/266, and 27/266. The spec-derived truthful Conv/ConvTranspose floor is cost 56, above the cost-45 immutable member.",
    },
    334: {
        "lead": "no below-43 graph; focused-history floor 85",
        "reason": "The complete archive, focused harvest, local exact scan, exact Wave2, and all-400 exact Einsum scan contain no strict decrease. The prior spec-derived supported rebuild floor was cost 49, already above the current cost 43.",
    },
    244: {
        "lead": "best different focused-history floor 43",
        "reason": "All harvested alternatives cost at least 43 versus 41. The spec-derived safe rebuild costs 70; its cost-67 shortcut is generator-wrong. No exact local or all-400 rewrite applies.",
    },
    249: {
        "lead": "cost-41 tie; other focused-history floor 206",
        "reason": "The only competitive different graph ties rather than improves. The supported spec-derived construction costs 57, and no exact price reduction exists.",
    },
    347: {
        "lead": "best different focused-history floor 51",
        "reason": "No graph below 41 appears in history or exact scans. A truthful spec-derived Conv/ConvTranspose rebuild costs 56; sub-floor grouped direct routes cannot route source colors 3/4 to output color 6.",
    },
    386: {
        "lead": "best different focused-history floor 60",
        "reason": "No graph below 41 appears in history or exact scans. A truthful spec-derived rebuild costs 78; the smaller direct/grouped families are incorrect or cannot satisfy the full output contract.",
    },
    146: {
        "lead": "two static-38 history leads reprofile to actual cost 67",
        "reason": "Both apparent lower leads exceed the cost-40 baseline after runtime-shape accounting, declare false output/runtime shapes, and fail default ORT session creation. No exact local or all-400 rewrite applies.",
    },
    291: {
        "lead": "truthful cost-40 one-node Einsum; cost-30 channel-free probe",
        "reason": "The cost-30 probe is 0/265 in both ORT modes because it cannot distinguish the answer channel. The exact spec-derived Einsum needs sign[10] plus edge[30], has zero intermediate memory, and already sits at cost 40; history offers ties only.",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    lower = json.loads((HERE / "lower_leads_dual.json").read_text())
    exact = json.loads((HERE / "exact_candidate_scan.json").read_text())

    base_by_task = {row["task"]: row for row in baseline["targets"]}
    known_by_task = {row["task"]: row for row in known["rows"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}
    lower_by_task = {
        task: [row for row in lower["rows"] if row["task"] == task]
        for task in TARGETS
    }
    exact_by_task = {row["task"]: row for row in exact["rows"]}

    rows = []
    for task in TARGETS:
        base = base_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shapes"]
        rows.append({
            "task": task,
            "rule_summary": rule_by_task[task]["rule_summary"],
            "true_rule_known": rule_by_task[task]["known"],
            "baseline_member": base["member"],
            "baseline_sha256": base["sha256"],
            "baseline_cost": base["actual_cost"],
            "unchanged_from_8004_50": base["unchanged_from_8004_50"],
            "baseline_known_dual": known_by_task[task],
            "structure_observation": {
                "ops": structure["ops"],
                "node_count": structure["node_count"],
                "max_node_inputs": structure["max_node_inputs"],
                "giant_einsum": structure["giant_einsum"],
                "lookup_or_scatter": structure["lookup_or_scatter"],
                "runtime_shape_cloak": runtime.get("shape_cloak", True),
                "runtime_shape_mismatches": runtime.get("mismatch_count"),
                "checker_full": structure["checker_full"],
                "strict_shape_data_prop": structure["strict_data_prop"],
                "standard_domains": structure["standard_domains"],
                "conv_bias_findings": structure["conv_bias_findings"],
            },
            "lower_leads": lower_by_task[task],
            "exact_scan": exact_by_task[task],
            "best_probe": PROBES[task]["lead"],
            "reason": PROBES[task]["reason"],
            "candidate": None,
            "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
            "candidate_known_dual": "FAILED_OR_NOT_RUN_PRE_GATE",
            "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
            "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
            "runtime_errors": "NO_ADOPTABLE_CANDIDATE",
            "conv_bias_ub0": "NO_ADOPTABLE_CANDIDATE",
            "projected_gain": 0.0,
        })

    result = {
        "lane": "agent_new_low43",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must have a decoded true-rule guarantee or exact LB-white lineage",
            "fail_closed": "fresh gates are not run when price, known, runtime, or structure gates fail",
        },
        "targets_requested": list(TARGETS),
        "targets_completed": len(rows),
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "zip_integration": False,
        "rows": rows,
        "evidence": {
            "baseline": "baseline_audit.json",
            "known_dual": "known_baseline_dual.json",
            "true_rules": "true_rule_audit.json",
            "history": "history_audit.json",
            "lower_leads": "lower_leads_dual.json",
            "exact": "exact_candidate_scan.json",
        },
        "final_verdict": "NO_SAFE_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({
        "lane": "agent_new_low43",
        "baseline_sha256": baseline["baseline"]["sha256"],
        "targets_completed": len(rows),
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

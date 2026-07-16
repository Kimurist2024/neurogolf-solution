#!/usr/bin/env python3
"""Consolidate low38 evidence into fail-closed final lane manifests."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGETS = (141, 4, 254, 49, 287, 78, 95, 7)

PROBES = {
    141: {
        "lead": "no below-78 archive or exact-rewrite graph",
        "reason": "The complete archive retains no lower graph. The initializer-alias and exact Wave2 scans build none; a conventional diagonal-sum implementation cannot beat the output-only 78-parameter incumbent.",
        "evidence": "history_audit.json",
    },
    4: {
        "lead": "focused-harvest static floors 78, 79, 80 and 4866",
        "reason": "Every historical graph is at or above the incumbent's actual cost 77. The incumbent itself is a fixed LB-white 1,991-input/shape-cloak authority and is not a safe construction template.",
        "evidence": "history_audit.json",
    },
    254: {
        "lead": "cost-42/68 giant-Einsum archive graphs and 60 safe TT rebuild attempts",
        "reason": "The cost-42 graph uses a 33-input giant Einsum and disagrees with the exact baseline on 412/500 external cases. Cost-68 still needs 20 operands. Sixty <=16-operand TT candidates failed and the exact TT-family floor is 114.",
        "evidence": "history_audit.json",
    },
    49: {
        "lead": "static-cost 69/70/72 archive graphs",
        "reason": "The focused actual-cost screen measures the strongest lower-static graph at cost 88, above the incumbent's actual 75. Remaining controls have floors >=84/344, so no actual strict decrease exists.",
        "evidence": "history_audit.json",
    },
    287: {
        "lead": "cost-30 Gather archive graph",
        "reason": "It is wrong on all four train examples (263/267 known overall), with 24-80 differing cells per failed example. The 44-cost apparent saving cannot pass the known100 pre-gate.",
        "evidence": "history_audit.json",
    },
    78: {
        "lead": "focused-harvest static floors 80 and 82",
        "reason": "Both historical graphs are above the incumbent's actual cost 72; the complete archive and exact scans retain no below-72 graph.",
        "evidence": "history_audit.json",
    },
    95: {
        "lead": "38-input giant-Einsum alternate and conventional floor 208",
        "reason": "The compact alternate is structurally forbidden and the conventional graph is far above actual cost 72. No safe lower graph survives the complete archive or exact scans.",
        "evidence": "history_audit.json",
    },
    7: {
        "lead": "cost-68 archive contraction",
        "reason": "It scores only 260/266 known in both ORT modes when the in-grid background channel is correctly one-hot encoded; six generator cases emit extra positive cells. It fails known100 before fresh testing.",
        "evidence": "history_audit.json",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    baseline_by_task = {row["task"]: row for row in baseline["targets"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}

    rows = []
    for task in TARGETS:
        base = baseline_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shape_trace"]
        probe = PROBES[task]
        rows.append(
            {
                "task": task,
                "rule_summary": rule_by_task[task]["rule_summary"],
                "true_rule_known": rule_by_task[task]["known"],
                "baseline_member": base["member"],
                "baseline_sha256": base["sha256"],
                "baseline_cost": base["actual_cost"],
                "unchanged_from_8004_50": base["unchanged_from_8004_50"],
                "structure_observation": {
                    "ops": structure["op_histogram"],
                    "node_count": structure["node_count"],
                    "max_node_inputs": structure["max_node_inputs"],
                    "giant_einsum": structure["giant_einsum"],
                    "huge_fanin": structure["huge_fanin"],
                    "lookup_or_scatter": structure["lookup_or_scatter_nodes"],
                    "runtime_shape_cloak": runtime.get("shape_cloak", True),
                    "runtime_shape_mismatches": runtime.get("mismatch_count"),
                    "runtime_error": runtime.get("error"),
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
                "known_dual": "FAILED_OR_NOT_RUN_PRE_GATE",
                "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
                "runtime_errors": "NO_ELIGIBLE_CANDIDATE",
                "conv_bias_ub0": "NO_ELIGIBLE_CANDIDATE",
                "projected_gain": 0.0,
            }
        )

    result = {
        "lane": "agent_new_low38",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must have a decoded true-rule guarantee or exact LB-white lineage",
            "fail_closed": "fresh is not run if price, structure, or known100 gates fail",
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
                "lane": "agent_new_low38",
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

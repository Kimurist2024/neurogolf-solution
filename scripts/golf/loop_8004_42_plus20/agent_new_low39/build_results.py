#!/usr/bin/env python3
"""Consolidate low39 evidence into fail-closed final manifests."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGETS = (32, 41, 215, 211, 120, 235, 258, 292)

PROBES = {
    32: {
        "lead": "cost-46 Q[6] row-coefficient probe",
        "reason": "The only below-70 archive graph fails all 266 known cases in both ORT modes: its six-row coefficient cannot broadcast against the static 30-row output. All other complete-history graphs cost at least 76/80 or the clean spec floor 94.",
    },
    41: {
        "lead": "no below-70 archive or exact-rewrite graph",
        "reason": "The complete archive retains no numeric lower graph; the focused harvest begins at 74/78 and the clean generator-compiled interval-fill model costs 925. Exact Wave2 and initializer-alias scans emit nothing.",
    },
    215: {
        "lead": "same-cost 70 rebuild only",
        "reason": "History contains two cost-70 output-only rebuilds and no strictly lower member. The incumbent itself uses an 18-input giant Einsum; no new clean below-70 lineage was found.",
    },
    211: {
        "lead": "cost-64 output-only 25-input Einsum",
        "reason": "The apparent two-point saving is a prohibited giant contraction and is only 9/266 correct in both ORT modes. All other history starts at 66 or 126, and exact scans emit nothing.",
    },
    120: {
        "lead": "static-41 AveragePool rebuild",
        "reason": "Runtime profiling measures the static-41 lead at actual cost 2738, far above current 64, with seven declared/runtime shape contradictions. Lower-looking scratch variants are shape-cloaked or fail ORT; no truthful candidate is cheaper.",
    },
    235: {
        "lead": "spec rebuild cost115; reduced variants cost107/91",
        "reason": "No history member is below actual 64. The decoded clean rebuild costs 115, while its smaller 107/91 variants are wrong; exact Wave2 and initializer scans find no shave in the cost-64 member.",
    },
    258: {
        "lead": "no below-64 graph; clean grouped-Conv floor160",
        "reason": "The complete archive/focused harvest supplies no strict lower lead. The direct spec-derived grouped Conv is correct at cost160; every loss/shift reduction below the historical clean floor either fails correctness/runtime or relies on false shapes.",
    },
    292: {
        "lead": "cost-50/50/54 rank-one/sign-core graphs",
        "reason": "All three truthful sub-64 graphs are 0/28 known in both ORT modes. They cannot retain the width-independent background branch while changing yellow by column phase; the next rank-one candidate costs70 and is also wrong.",
    },
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    base_by_task = {row["task"]: row for row in baseline["targets"]}
    known_by_task = {row["task"]: row for row in known["rows"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}

    rows = []
    for task in TARGETS:
        base = base_by_task[task]
        structure = base["structure"]
        runtime = structure["runtime_shapes"]
        rows.append(
            {
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
                    "huge_fanin": structure["huge_fanin"],
                    "lookup_or_scatter": structure["lookup_or_scatter"],
                    "runtime_shape_cloak": runtime.get("shape_cloak", True),
                    "runtime_shape_mismatches": runtime.get("mismatch_count"),
                    "runtime_error": runtime.get("error"),
                    "checker_full": structure["checker_full"],
                    "strict_data_prop": structure["strict_data_prop"],
                    "standard_domains": structure["standard_domains"],
                    "conv_bias_findings": structure["conv_bias_findings"],
                },
                "best_probe": PROBES[task]["lead"],
                "reason": PROBES[task]["reason"],
                "evidence": "history_audit.json and lower_leads_dual.json",
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "candidate_known_dual": "FAILED_OR_NOT_RUN_PRE_GATE",
                "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
                "runtime_errors": "NO_ELIGIBLE_CANDIDATE",
                "conv_bias_ub0": "NO_ELIGIBLE_CANDIDATE",
                "projected_gain": 0.0,
            }
        )

    result = {
        "lane": "agent_new_low39",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must have a decoded true-rule guarantee or exact LB-white lineage",
            "fail_closed": "fresh is not run if actual price, structure, runtime-shape, or known100 pre-gates fail",
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
                "lane": "agent_new_low39",
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

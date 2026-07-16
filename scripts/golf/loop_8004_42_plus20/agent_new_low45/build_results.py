#!/usr/bin/env python3
"""Finalize the low45 no-winner contract and human report."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent

BEST = {
    24: (
        "no graph below cost 30; focused alternate costs 56",
        "The current 9-input, one-output Einsum has zero counted memory. The all-archive, focused-harvest, exact-Wave2, and initializer-alias passes expose no lower member.",
    ),
    113: (
        "cost-30 one-node Gather tie",
        "Its 30 row indices are the entire cost. Task-specific searches tested 268,968 additional one-node candidates plus earlier Pad/pool/Conv/Resize families without a sub-30 survivor.",
    ),
    385: (
        "five apparent static-cost 0/1 archive graphs",
        "All five fail both gold implementations and all 20 fresh cases. The exact one-node Gather already has zero memory and only 30 row indices.",
    ),
    389: (
        "cost-20 seven-factor approximation",
        "The only numeric lower archive lead is false on both gold implementations and all 20 fresh cases. Other observed graphs cost 30 or 55.",
    ),
    296: (
        "factored ConvTranspose selector area 18 ties cost 28",
        "The calibrated exact family found no solution with selector area <=17; area 18 reproduces the incumbent's 18 selector plus 10 bias parameters.",
    ),
    399: (
        "cost-25 same-cost history member",
        "The current graph is already below the earlier cost-34 search floor. The focused harvest contains only a cost-25 tie and a cost-60 dominated model; exact passes find no shave.",
    ),
    359: (
        "no sound graph below giant-Einsum cost 24",
        "Generator-exact logic needs row/column histograms and orientation scoring; one natural 300-value histogram already exceeds cost 24. All six broadcast-prune candidates are known false.",
    ),
    110: (
        "only cost-24/low giant-Einsum lineages",
        "Focused history contains 18/29-input giant Einsums or a cost-10327 conventional graph. The current 37-input graph is fixed LB-white evidence, not a safe template for a new candidate.",
    ),
}


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    known = json.loads((HERE / "known_baseline_dual.json").read_text())
    rules = json.loads((HERE / "true_rule_audit.json").read_text())
    known_by_task = {row["task"]: row for row in known["rows"]}
    rule_by_task = {row["task"]: row for row in rules["rows"]}
    rows = []
    for record in baseline["targets"]:
        task = record["task"]
        structure = record["structure"]
        runtime = structure["runtime_shape_trace"]
        rows.append(
            {
                "task": task,
                "rule_summary": rule_by_task[task]["rule_summary"],
                "true_rule_known": rule_by_task[task]["known"],
                "baseline_member": record["member"],
                "baseline_sha256": record["sha256"],
                "baseline_cost": record["actual_cost"],
                "unchanged_from_8004_50": record["unchanged_from_8004_50"],
                "baseline_known_dual": known_by_task[task],
                "structure": {
                    "ops": structure["op_histogram"],
                    "node_count": structure["node_count"],
                    "max_node_inputs": structure["max_node_inputs"],
                    "giant_einsum": structure["giant_einsum"],
                    "lookup_or_scatter": structure["lookup_or_scatter_nodes"],
                    "runtime_shape_cloak": runtime.get("shape_cloak"),
                    "runtime_shape_mismatches": runtime.get("mismatch_count"),
                    "checker_full": structure["checker_full"],
                    "strict_shape_data_prop": structure["strict_shape_data_prop"],
                    "standard_domains": structure["standard_domains"],
                    "conv_bias_findings": structure["conv_bias_findings"],
                },
                "best_probe": BEST[task][0],
                "reason": BEST[task][1],
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "candidate_known_dual": "NOT_RUN_NO_CANDIDATE",
                "candidate_fresh_seed_1_dual": "NOT_RUN_NO_CANDIDATE",
                "candidate_fresh_seed_2_dual": "NOT_RUN_NO_CANDIDATE",
                "candidate_runtime_errors": "NO_CANDIDATE",
                "candidate_conv_bias_ub0": "NO_CANDIDATE",
                "projected_gain": 0.0,
            }
        )
    result = {
        "lane": "agent_new_low45",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, checker full, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must have a true-rule guarantee or exact LB-white lineage",
            "fail_closed": "candidate known/fresh gates are not run when no proposal clears price and structure",
        },
        "targets_requested": [24, 113, 385, 389, 296, 399, 359, 110],
        "targets_completed": len(rows),
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "zip_integration": False,
        "rows": rows,
        "final_verdict": "NO_SAFE_CANDIDATE",
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "lane": "agent_new_low45",
        "baseline": baseline["baseline"],
        "accepted": [],
        "accepted_count": 0,
        "projected_gain": 0.0,
        "zip_integration": False,
        "verdict": "NO_SAFE_CANDIDATE",
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    table_rows = []
    for row in rows:
        disable = row["baseline_known_dual"]["disable_all"]
        default = row["baseline_known_dual"]["default"]
        op_text = ", ".join(f"{key}x{value}" for key, value in row["structure"]["ops"].items())
        table_rows.append(
            f"| {row['task']:03d} | {row['baseline_cost']['cost']} | {op_text} | "
            f"{disable['right']}/{disable['total']} | {default['right']}/{default['total']} | {row['best_probe']} | reject |"
        )
    report = f"""# Low45 target expansion — 8-task audit

## Outcome

The additional eight files were independently audited against
`submission_base_8005.16.zip`. **No safe strictly-cheaper candidate exists in
the searched exact, complete-history, task-specific, and decoded-rule
families.** This lane contributes **+0.0**, emits no candidate, and does not
build or modify a submission ZIP.

- baseline SHA-256: `{baseline['baseline']['sha256']}`;
- completed: **8/8**;
- accepted: **0**;
- all eight members are byte-identical to the 8004.50 base;
- all eight incumbents pass full checker, strict/data-prop, truthful runtime
  shapes, standard-domain, Conv-bias UB0, and known100 in both ORT modes;
- every decoded Sakana rule reproduces the complete known corpus;
- protected ZIPs, score files, and CSVs were not modified.

## Per-task decision

| task | cost | current structure | known disable | known default | strongest lower/tie lead | decision |
|---:|---:|---|---:|---:|---|---|
{chr(10).join(table_rows)}

Task359 and task110 are fixed LB-white giant-Einsum incumbents. Their lineage
does not authorize a new unsafe model: any replacement would still need to
pass the requested no-giant/truthful/dual-ORT/fresh guarantee gates. No such
strictly-cheaper replacement exists.

## Search evidence

1. The all-400 inventory covers **1,196 ZIPs, 448,568 ZIP members, 233,751
   loose observations, and 13,591 unique non-baseline graphs**. It retains only
   six numeric lower leads for these targets: five task385 artifacts and one
   task389 artifact. All six fail both gold checks and all 20 fresh cases.
2. The focused harvest finds no strict safe decrease: task024's alternate
   costs 56; task113/task385/task389 have cost-30 ties or dominated graphs;
   task296 costs 90; task399 has a cost-25 tie; task110's low artifacts remain
   18/29-input giant Einsums; and task359 has no sound lower authority.
3. The complete exact Wave2 pass produces zero opportunities or candidates for
   these eight members. The all-400 initializer-alias scan also builds zero
   candidate for them.
4. Task113's dedicated search tested 268,968 additional low-parameter
   final-output candidates, alongside earlier Pad/pool/Conv/Resize screens,
   without a sub-30 survivor.
5. Task296's calibrated factored ConvTranspose search finds no valid selector
   area <=17; area 18 only ties the current cost 28.
6. Task359's generator-exact reconstruction needs row/column histograms and
   orientation scoring. Even one natural 300-value histogram exceeds its
   fixed cost-24 giant-Einsum incumbent.
7. Task110's periodic-pattern restoration likewise has no ordinary one-node
   operator that fits below cost 24; the only compact history remains giant
   Einsum, which is forbidden for a new candidate.

## Gate disposition

No proposal clears the prerequisite strictly-lower actual-cost and safe-
structure gates. Candidate known and fresh tests therefore were not started;
fresh evaluation cannot make a same-cost, known-false, cost-dominated, or
structurally forbidden graph adoptable.

Authoritative evidence:

- `baseline_audit.json` — latest hashes, actual costs, checker/strict,
  runtime-shape, domain, op/fan-in, and Conv-bias evidence;
- `known_baseline_dual.json` — complete known corpus under both ORT modes;
- `true_rule_audit.json` — readable rule summaries and known reproduction;
- `history_audit.json` — exhaustive archive, focused harvest, exact passes,
  and task-specific proof pointers;
- `result.json` — eight final decisions;
- `winner_manifest.json` — empty authoritative promotion list.
"""
    (HERE / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()

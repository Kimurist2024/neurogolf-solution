#!/usr/bin/env python3
"""Complete the bounded audit after the expensive task304 fresh run was stopped."""

from __future__ import annotations

import json

import audit_lane as audit


def main() -> None:
    output = audit.EVIDENCE / "audit.json"
    report = json.loads(output.read_text(encoding="utf-8"))

    task = 383
    path = audit.AUTHORITY / "task383.onnx"
    data = path.read_bytes()
    known = audit.load_known(task)
    row = {
        "generator": f"inputs/arc-gen-repo/tasks/task_{audit.HASHES[task]}.py",
        "generator_sha256": audit.sha256_file(
            audit.ROOT / "inputs/arc-gen-repo/tasks" / f"task_{audit.HASHES[task]}.py"
        ),
        "raw_rule": "inputs/sakana-gcg-2025/raw/task383.py",
        "structural": audit.structural_audit(task, path),
        "runtime_shape": audit.runtime_shape_trace(data, known[0]),
        "known_independent_truth": audit.independent_truth(task, known),
        "known_runtime": audit.runtime_group(data, known),
        "optimizer_profiles": audit.optimizer_scan(task, data),
        "fresh": {},
    }
    for seed in audit.FRESH_SEEDS:
        fresh = audit.make_fresh(task, seed)
        row["fresh"][str(seed)] = {
            "independent_truth": audit.independent_truth(task, fresh),
            "runtime": audit.runtime_group(data, fresh),
        }
    report["tasks"][str(task)] = row

    # The task304 authority itself is not a candidate and no optimizer profile
    # is strict-lower.  Its 4-config known audit completed; the stopped fresh
    # authority-only extension is explicitly recorded rather than hidden.
    report["tasks"]["304"]["fresh_not_run"] = (
        "authority-only task304 fresh x four configs stopped on parent request; "
        "no strict-lower optimizer candidate existed"
    )

    probes = [
        (263, audit.build_task263_truthful_bypass()),
        (304, audit.build_task304_precontract()),
        *[(304, audit.build_task304_rank_drop(i)) for i in range(4)],
        *[(304, audit.build_task304_t_drop(i)) for i in range(2)],
    ]
    manual = []
    for probe_task, probe_path in probes:
        # One known witness in all four configs is enough to hard-reject a
        # wrong probe; non-lower probes are rejected before fresh gating.
        manual.append(audit.audit_probe(probe_task, probe_path, audit.load_known(probe_task)[:1]))
    report["manual_probes"] = manual

    optimizer_lower = [
        {"task": int(task_num), **profile}
        for task_num, task_row in report["tasks"].items()
        for profile in task_row["optimizer_profiles"]
        if profile.get("strict_lower")
    ]
    manual_lower = [row for row in manual if row.get("strict_lower")]
    report["summary"] = {
        "optimizer_profiles": len(audit.TASKS) * len(audit.PASS_SETS),
        "optimizer_strict_lower_count": len(optimizer_lower),
        "optimizer_strict_lower": optimizer_lower,
        "manual_strict_lower_count": len(manual_lower),
        "manual_strict_lower": [
            {
                "task": row["task"],
                "path": row["path"],
                "sha256": row["sha256"],
                "cost": row.get("structural", {}).get("profile", {}).get("cost"),
                "known_first_failure": row.get("known_four_configs", {}).get("first_failure"),
            }
            for row in manual_lower
        ],
        "accepted_winners": [],
        "exact_cost_gain": 0,
        "exact_score_gain": 0.0,
        "decision": "NO_STRICT_LOWER_SUPPORT_SAFE_WINNER",
    }
    report["guard_after"] = audit.guard_snapshot()
    report["guards_unchanged"] = report["guard_before"] == report["guard_after"]
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

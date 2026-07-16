#!/usr/bin/env python3
"""Isolated one-candidate worker; ORT crashes cannot kill the coordinator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import onnx

import audit_leads as audit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((HERE / "inventory.json").read_text())
    tasks = [int(task) for task in inventory["targets"]]
    authority_costs = {
        int(row["task"]): int(row["authority_cost"])
        for row in inventory["summary"]
    }
    leads = [
        row
        for task in tasks
        for row in inventory["rows_by_task"][str(task)]
        if row.get("could_be_actual_strict_lower") and row.get("candidate_path")
    ]
    lead = leads[args.index - 1]
    task = int(lead["task"])
    path = ROOT / str(lead["candidate_path"])
    model = onnx.load(path)
    known = audit.cases(task)
    disabled, cost = audit.profile_disabled_all(task, model, known)
    actual_lower = bool(
        cost is not None and 0 <= cost["cost"] < authority_costs[task]
    )
    known_four = {"disable_all_t1": disabled}
    if actual_lower and disabled.get("known_perfect"):
        for label, optimization, threads in audit.CONFIGS[1:]:
            known_four[label] = audit.audit_config(model, known, optimization, threads)
    known4_complete = len(known_four) == 4 and all(
        row.get("known_perfect") for row in known_four.values()
    )
    if actual_lower and known4_complete:
        static = audit.structure(task, model, known)
    else:
        static = {
            "pass": False,
            "not_run": "requires actual strict lower and complete known×4",
            "lookup_or_cloak_ops": lead.get("structure", {}).get("lookup_or_cloak_ops", []),
            "giant_einsum": lead.get("structure", {}).get("giant_einsum", False),
            "custom_domains": lead.get("structure", {}).get("custom_domains", []),
        }
    exact = bool(lead.get("exact_computational_graph_equivalent"))
    any_nonfinite = any(
        int(row.get("nonfinite_values", 0)) > 0 for row in known_four.values()
    )
    any_near = any(
        int(row.get("near_positive_values_0_to_0_25", 0)) > 0
        for row in known_four.values()
    )
    if actual_lower and known4_complete and static.get("pass"):
        if exact and not any_nonfinite and not any_near:
            decision = "EXACT_FIXED_CANDIDATE"
        else:
            decision = "LB_PROBE_REQUIRED"
    elif not actual_lower:
        decision = "REJECT_ACTUAL_NOT_STRICT_LOWER_OR_UNSCORABLE"
    elif not known4_complete:
        decision = "REJECT_KNOWN_OR_RUNTIME"
    else:
        decision = "REJECT_SCHEMA_SHAPE_OR_UB"
    risks: list[str] = []
    source_structure = lead.get("structure", {})
    if source_structure.get("giant_einsum"):
        risks.append("giant_einsum")
    if source_structure.get("lookup_or_cloak_ops"):
        risks.append("lookup_or_cloak")
    if source_structure.get("custom_domains"):
        risks.append("custom_domain")
    if any_nonfinite:
        risks.append("nonfinite_output")
    if any_near:
        risks.append("near_positive_output")
    row = {
        "audit_index": args.index,
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": audit.digest(path),
        "sources": lead["sources"],
        "source_count": lead["source_count"],
        "authority_cost": authority_costs[task],
        "static_cost_floor": lead.get("static_cost_floor"),
        "actual_cost": cost,
        "actual_strict_lower": actual_lower,
        "gain_if_valid": (
            math.log(authority_costs[task] / cost["cost"])
            if actual_lower and cost and cost["cost"] > 0
            else 0.0
        ),
        "exact_computational_graph_equivalent": exact,
        "known_complete_four_configs": known_four,
        "known_four_complete_pass": known4_complete,
        "structure": static,
        "risk_classification": sorted(set(risks)),
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"task{task:03d} cost={cost['cost'] if cost else None}/{authority_costs[task]} "
        f"known={disabled.get('right', 0)}/{disabled.get('total', '?')} {decision}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

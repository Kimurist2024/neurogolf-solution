#!/usr/bin/env python3
"""Bounded actual-cost/known/dual-ORT/shape audit for the four assigned tasks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
C11_PATH = ROOT / "scripts/golf/loop_7999_13/lane_c11/audit_candidates.py"
SPEC = importlib.util.spec_from_file_location("high3_c11", C11_PATH)
assert SPEC is not None and SPEC.loader is not None
C11 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C11)

TASKS = (18, 233, 286, 366)
BASE_COSTS = {18: 4754, 233: 7432, 286: 7481, 366: 7987}

# These labels are provenance decisions from the audited source families, not
# guesses based on op names.  A correction table may use ordinary ONNX ops.
LINEAGE = {
    "task018_a03": "private_zero_lookup_lineage",
    "task233_a02": "private_zero_lookup_lineage",
    "task233_a03": "private_zero_lookup_lineage",
    "task233_a04": "private_zero_lookup_lineage",
    "task286_a01": "public_fixture_rcorr_lookup_lineage",
    "task286_a02": "public_fixture_rcorr_lookup_lineage",
    "task286_a03": "public_fixture_rcorr_lookup_lineage",
    "task286_a04": "public_fixture_rcorr_lookup_lineage",
}


def safe_total(record: dict[str, object], key: str) -> dict[str, int] | None:
    block = record.get(key)
    if not isinstance(block, dict):
        return None
    total = block.get("total")
    return total if isinstance(total, dict) else None


def classify(label: str, task: int, record: dict[str, object]) -> dict[str, object]:
    score = record.get("official_like_score")
    cost = score.get("cost") if isinstance(score, dict) else None
    disable = safe_total(record, "known_disable_all")
    default = safe_total(record, "known_default")
    trace = record.get("runtime_shape_trace")
    shape_truthful = (
        isinstance(trace, dict)
        and not trace.get("error")
        and not trace.get("declared_actual_mismatches")
    )
    structural = (
        record.get("full_check") is True
        and record.get("strict_shape_data_prop") is True
        and not record.get("nonstandard_domains")
        and not record.get("banned_ops")
        and not record.get("nested_graph_attributes")
        and record.get("function_count") == 0
        and record.get("sparse_initializer_count") == 0
        and not record.get("lookup_red_flags", {}).get("tfidf")
        and not record.get("lookup_red_flags", {}).get("hardmax")
        and not record.get("lookup_red_flags", {}).get("giant_einsum_nodes")
        and all(item.get("safe") for item in record.get("conv_bias_findings", []))
    )
    known_dual = all(
        total is not None and total.get("wrong") == 0 and total.get("errors") == 0
        for total in (disable, default)
    )
    lineage = LINEAGE.get(label, "spec_or_nonlookup_history")
    pre_fresh = (
        isinstance(cost, int)
        and cost < BASE_COSTS[task]
        and structural
        and shape_truthful
        and known_dual
        and lineage == "spec_or_nonlookup_history"
    )
    reasons = []
    if not isinstance(cost, int) or cost >= BASE_COSTS[task]:
        reasons.append("not_strictly_cheaper_actual_cost")
    if not structural:
        reasons.append("structural_or_ub_gate_failed")
    if not shape_truthful:
        reasons.append("runtime_shape_truthfulness_failed")
    if not known_dual:
        reasons.append("known100_dualORT_failed")
    if lineage != "spec_or_nonlookup_history":
        reasons.append(lineage)
    return {
        "actual_cost": cost,
        "baseline_cost": BASE_COSTS[task],
        "structural_gate": structural,
        "shape_truthful": shape_truthful,
        "known100_dualORT": known_dual,
        "lineage": lineage,
        "eligible_for_fresh2000": pre_fresh,
        "reasons": reasons,
    }


def main() -> None:
    C11.ort.set_default_logger_severity(4)
    manifest = json.loads((HERE / "asset_manifest.json").read_text())
    target = HERE / "attempt_audit.json"
    if target.exists():
        prior = json.loads(target.read_text())
        records: dict[str, object] = prior.get("records", {})
        decisions: dict[str, object] = prior.get("decisions", {})
    else:
        records, decisions = {}, {}
    for task in TASKS:
        base_label = f"baseline_task{task:03d}"
        base_path = ROOT / manifest["baselines"][str(task)]["path"]
        if base_label not in records:
            print(f"audit {base_label}", flush=True)
            records[base_label] = C11.audit(base_label, task, base_path)
            target.write_text(json.dumps({"records": records, "decisions": decisions}, indent=2) + "\n")
        for row in manifest["attempts"][str(task)]:
            if row.get("missing"):
                continue
            path = ROOT / row["path"]
            label = path.stem
            if label in records:
                decisions[label] = classify(label, task, records[label])
                continue
            print(f"audit {label}", flush=True)
            record = C11.audit(label, task, path)
            records[label] = record
            decisions[label] = classify(label, task, record)
            target.write_text(json.dumps({"records": records, "decisions": decisions}, indent=2) + "\n")
            print(label, decisions[label], flush=True)
    winners = [label for label, decision in decisions.items() if decision["eligible_for_fresh2000"]]
    payload = {
        "records": records,
        "decisions": decisions,
        "pre_fresh_winners": winners,
        "attempt_count": len(decisions),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attempt_count": len(decisions), "pre_fresh_winners": winners}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize the exact-resume lane with conservative policy decisions."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(name: str, payload: dict) -> None:
    (HERE / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    scan = read(HERE / "scan_report.json")
    audit48 = read(HERE / "audit_task048.json")
    validation333 = read(
        HERE.parent / "agent_exact_scanners/validation_task333_shared_sign.json"
    )
    profile233 = read(
        HERE.parent / "duplicate_initializer_candidates/task233_profile.json"
    )

    requested = {row["task"]: row for row in scan["existing_candidate_copies"]}
    task48 = {
        "task": 48,
        "candidate": requested[48]["path"],
        "candidate_sha256": requested[48]["sha256"],
        "baseline_sha256": audit48["baseline_member_sha256"],
        "rewrite": "Einsum inputs 4 -> 3 by exact initializer outer-product fusion",
        "cost_before": audit48["baseline_score"]["cost"],
        "cost_after": audit48["candidate_score"]["cost"],
        "cost_reduction": audit48["cost_reduction"],
        "projected_gain": audit48["projected_gain"],
        "known": audit48["modes"]["disable_all"]["known"],
        "fresh": {
            "requested": audit48["fresh_requested"],
            "executable": audit48["fresh_executable"],
            "disable_all": audit48["modes"]["disable_all"]["fresh"],
            "default": audit48["modes"]["default"]["fresh"],
            "generation_errors": audit48["fresh_generation_errors"],
        },
        "checker": requested[48]["structural_gate"],
        "runtime_errors": 0,
        "verdict": "REJECT",
        "reasons": [
            "fresh ground-truth accuracy is 1818/2000 = 90.9%, below the authorized 95% gate",
            "catalogued high-risk task",
            "floating Einsum contraction structure changes for only +0.002642 projected gain",
        ],
    }
    write("candidate_task048.json", task48)

    task333 = {
        "task": 333,
        "candidate": requested[333]["path"],
        "candidate_sha256": requested[333]["sha256"],
        "baseline_sha256": validation333["baseline"]["sha256"],
        "rewrite": "absorb GE sign into shared HC and compensate GHHT; Einsum inputs 36 -> 35",
        "cost_before": validation333["baseline"]["cost"],
        "cost_after": validation333["candidate"]["cost"],
        "cost_reduction": validation333["decision"]["cost_reduction"],
        "projected_gain": validation333["decision"]["projected_gain"],
        "known": validation333["candidate"]["known"],
        "raw_differential": validation333["differential"],
        "fresh_ground_truth": {
            "status": "not_required_after_policy_rejection",
            "count": 0,
        },
        "checker": requested[333]["structural_gate"],
        "runtime_errors": validation333["candidate"]["known"]["errors"],
        "verdict": "REJECT",
        "reasons": [
            "policy rejects giant/multi-input floating Einsum contraction changes even when sampled raw outputs match",
            "catalogued monitored task; operand count/order changes can be platform-sensitive",
            "projected gain is only +0.004739",
        ],
    }
    write("candidate_task333.json", task333)

    task233 = {
        "task": 233,
        "candidate": requested[233]["path"],
        "candidate_sha256": requested[233]["sha256"],
        "baseline_sha256": profile233["baseline"]["sha256"],
        "rewrite": "replace audit_one_i16 with byte-identical one_i8 initializer",
        "cost_before": profile233["baseline"]["cost"],
        "cost_after": profile233["candidate"]["cost"],
        "cost_reduction": profile233["decision"]["cost_reduction"],
        "projected_gain": profile233["decision"]["projected_gain"],
        "known": profile233["candidate"]["known"],
        "fresh_ground_truth": {
            "status": "not_completed_before_policy_rejection",
            "count": 0,
        },
        "exact_initializer_proof": scan["initializer_dedup"][0]["rewrite"],
        "checker": requested[233]["structural_gate"],
        "runtime_errors": profile233["candidate"]["known"]["errors"],
        "verdict": "REJECT",
        "reasons": [
            "catalogued highest-risk task",
            "dust gain of +0.000134562 does not justify changing its submitted bytes",
            "fresh adoption gate was not completed; policy rejection makes further execution unnecessary",
        ],
    }
    write("candidate_task233.json", task233)

    report = {
        "baseline": {
            "zip": scan["baseline"],
            "sha256": scan["baseline_sha256"],
            "leaderboard_score": 8003.40,
        },
        "scope": {
            "tasks_scanned": scan["task_count"],
            "patterns": [
                "byte-identical initializer deduplication",
                "exact disjoint initializer fusion inside Einsum",
                "exact +/-1 gauge/sign absorption with compensation",
                "strictly re-inferred overdeclared value_info",
            ],
            "private_zero_excluded": scan["private_zero_excluded"],
        },
        "scan_results": {
            "initializer_dedup_opportunities": len(scan["initializer_dedup"]),
            "outer_fusion_opportunities": len(scan["outer_fusion"]),
            "sign_absorption_opportunities": len(scan["sign_absorption"]),
            "metadata": scan["metadata_scan_summary"],
            "scan_errors": scan["errors"],
        },
        "candidate_decisions": [task48, task233, task333],
        "accepted": [],
        "rejected_tasks": [48, 233, 333],
        "projected_gain_accepted": 0.0,
        "zip_merge_performed": False,
        "protected_files_changed": False,
        "notes": [
            "task070 was explicitly excluded as known private-zero lineage.",
            "No independently re-inferable overdeclared value_info reduction was found in the 400 baseline members.",
            "The separate archive lane owns the externally sourced task109 annotation-only candidate; it is not duplicated here.",
        ],
        "verdict": "NO_SAFE_EXACT_CANDIDATE",
    }
    write("FINAL_REPORT.json", report)

    markdown = f"""# Exact resume report (8003.40 baseline)

## Outcome

- Scanned: **{scan['task_count']}/400 tasks**
- Accepted: **0**
- ZIP merge: **not performed**
- Protected root/baseline files: **unchanged**
- Final verdict: **NO_SAFE_EXACT_CANDIDATE**

## Candidate decisions

| Task | SHA-256 | Cost | Evidence | Decision |
|---:|---|---:|---|---|
| 048 | `{task48['candidate_sha256']}` | {task48['cost_before']} -> {task48['cost_after']} | known 270/270; fresh 1818/2000 (90.9%), errors 0 | **REJECT**: below 95%; high-risk floating contraction change |
| 233 | `{task233['candidate_sha256']}` | {task233['cost_before']} -> {task233['cost_after']} | known 266/266, errors 0; exact initializer alias | **REJECT**: highest-risk task and only +{task233['projected_gain']:.9f} |
| 333 | `{task333['candidate_sha256']}` | {task333['cost_before']} -> {task333['cost_after']} | known 265/265; raw differential 2000/2000, errors 0 | **REJECT**: giant Einsum 36 -> 35 is platform-order sensitive |

Every candidate passes full checker, strict shape inference/data propagation, static-shape checks, banned-op checks, and the Conv/ConvTranspose/QLinearConv bias-length gate. Passing those structural gates does not override the risk/fresh policies above.

## Whole-baseline scan

- Byte-identical initializer dedup: {len(scan['initializer_dedup'])} opportunity (task233)
- Exact disjoint outer fusion: {len(scan['outer_fusion'])} variants (task048)
- Exact sign/gauge absorption: {len(scan['sign_absorption'])} opportunity (task333)
- Overdeclared value_info proven by clean strict re-inference: 0 tasks
- Metadata scans completed: {scan['metadata_scan_summary']['scanned']}; strict re-inference unavailable for {scan['metadata_scan_summary']['inference_failures']} baseline models
- task070: explicitly excluded as known private-zero lineage

The separate archive-resume lane is responsible for the externally sourced task109 annotation-only candidate; this lane did not merge or duplicate it.
"""
    (HERE / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"],
        "accepted": 0,
        "rejected_tasks": report["rejected_tasks"],
        "report": str((HERE / "FINAL_REPORT.json").relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()

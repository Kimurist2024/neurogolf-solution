#!/usr/bin/env python3
"""Finalize corrected C40: only cheaper-than-104 SOUND candidates are eligible."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> int:
    baseline = read("baseline_audit.json")
    external = read("baseline_external_known.json")
    history = read("history_audit.json")
    rule = read("generator_rule_audit.json")
    floor = read("sound_floor_audit.json")
    base_cost = baseline["official_like_score"]["cost"]
    assert base_cost == 104
    assert rule["pass"]
    assert not history["eligible"]

    output = {
        "task": 391,
        "corrected_scope": (
            "The authoritative fallback is LB-white. Promote only a candidate that is both "
            "strictly cheaper than cost 104 and SOUND; do not replace it with a higher-cost rebuild."
        ),
        "authority": {
            "zip": "submission_base_8000.46.zip",
            "member": "task391.onnx",
            "sha256": baseline["sha256"],
            "file_bytes": baseline["file_bytes"],
            "cost": baseline["official_like_score"],
            "known_disable_all": baseline["known_disable_all"]["total"],
            "known_default": baseline["known_default"]["total"],
            "runtime_shape_trace": baseline["runtime_shape_trace"],
            "external_known": external["candidate"]["known"],
            "lb_status": "white fallback, per root correction",
            "grandfathered_lookup": True,
        },
        "generator_rule": rule,
        "sub_baseline_history": history["sub_baseline_candidates"],
        "sound_controls": history["sound_controls"],
        "sound_floor": floor,
        "lb_and_quarantine_evidence": {
            "private_zero_log": (
                "docs/golf/private_zero_tasks.md identifies task391's cheap h7901 replacement "
                "as the private-0 regression and records successful rollback."
            ),
            "quarantine_member": (
                "artifacts/quarantine/task391_7801rej_cost102_private0_soloprobe.onnx"
            ),
            "quarantine_sha256": "7ccd0d275038649f5301ab7a19913928eb582c61497b1fb2de681f93be5ad15b",
            "quarantine_cost": 102,
            "quarantine_lookup_nodes": 3,
            "interpretation": (
                "The black evidence applies to cheap lookup replacements, not the restored "
                "authoritative fallback."
            ),
        },
        "candidate_gate": {
            "strictly_cheaper_safe_candidate_found": False,
            "below_baseline_models_examined": len(history["sub_baseline_candidates"]),
            "below_baseline_lookup_rejections": sum(
                bool(row["lookup_nodes"]) for row in history["sub_baseline_candidates"]
            ),
            "known_dual_candidate": {
                "run": False,
                "reason": "No candidate survived the cheaper-plus-safe-structure gate.",
            },
            "fresh_dual_5000_candidate": {
                "run": False,
                "reason": "No candidate survived the cheaper-plus-safe-structure gate.",
            },
            "external_validator_candidate": {
                "run": False,
                "reason": "No candidate survived the cheaper-plus-safe-structure gate.",
            },
        },
        "decision": {
            "winner": None,
            "baseline_preserved": True,
            "verified_score_gain": 0.0,
            "reason": (
                "All discovered cost-85/87/88/102 models are lookup payloads. The smallest "
                "table-free true-rule control costs 139, above the cost-104 authority."
            ),
        },
    }
    (HERE / "final_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    winner = {
        "task": 391,
        "baseline_sha256": baseline["sha256"],
        "baseline_cost": base_cost,
        "winners": [],
        "lane_projected_gain": 0.0,
        "reason": "No strictly cheaper SOUND candidate exists in the audited pool.",
    }
    (HERE / "winner_manifest.json").write_text(
        json.dumps(winner, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

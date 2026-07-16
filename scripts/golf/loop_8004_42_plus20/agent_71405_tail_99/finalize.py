#!/usr/bin/env python3
"""Assemble immutable row evidence into final lane manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
ORDER = (
    "task310_improved_1",
    "task354_improved",
    "task361_cost844",
    "task363_improved",
    "task365_cost1355",
    "task370_improved_1",
    "task378_improved",
    "task396_cost_reduced",
    "task396_improved_v2",
    "task268_improved_cost420",
    "task270_improved",
    "task284_improved",
)

DECISIONS = {
    "task310_improved_1": (
        "LB_PROBE_REQUIRED_HIGH_RISK",
        "exact SHA is unprobed; truthful and known×4, but lookup/31-input Einsum and "
        "one false case in 10,000 fresh make fixed adoption unsafe",
    ),
    "task354_improved": (
        "REJECT_SHAPE_CLOAK",
        "seven runtime/declaration shape mismatches",
    ),
    "task361_cost844": (
        "REJECT_DEFAULT_ORT_RUNTIME",
        "default ORT cannot construct CenterCropPad session; task also has version-dependent LB-black history",
    ),
    "task363_improved": (
        "REJECT_SHAPE_CLOAK_FRESH_FAILURE",
        "seven runtime/declaration shape mismatches and 995/1000 minimum fresh",
    ),
    "task365_cost1355": (
        "REJECT_SHAPE_CLOAK_KNOWN_BLACK_TASK",
        "twelve runtime/declaration shape mismatches; a different SHA for task365 is directly LB-black",
    ),
    "task370_improved_1": (
        "REJECT_DEFAULT_ORT_RUNTIME",
        "default ORT cannot construct Concat session",
    ),
    "task378_improved": (
        "REJECT_RUNTIME_SHAPE_REUSE_FRESH_FAILURE",
        "direct trace raises an ORT buffer shape-reuse error and fresh minimum is 490/500",
    ),
    "task396_cost_reduced": (
        "LB_PROBE_REQUIRED_KNOWN_RISK_LOW",
        "exact SHA is unprobed and truthful, but fresh is only 971/1000 and task396 has cost1026 LB-white then cost982 LB-black history",
    ),
    "task396_improved_v2": (
        "LB_PROBE_REQUIRED_KNOWN_RISK_LOW",
        "exact SHA is unprobed and truthful, but fresh is only 971/1000 and task396 has cost1026 LB-white then cost982 LB-black history",
    ),
    "task268_improved_cost420": (
        "REJECT_DEFAULT_ORT_OVERSIZE_LOOKUP",
        "default ORT CenterCropPad session failure; serialized size exceeds 1,440,000 bytes and contains lookup operators",
    ),
    "task270_improved": (
        "REJECT_SHAPE_CLOAK",
        "four runtime/declaration shape mismatches and giant Einsum",
    ),
    "task284_improved": (
        "REJECT_SHAPE_CLOAK",
        "eleven runtime/declaration shape mismatches",
    ),
}

RELATED_HISTORY = {
    "task310_improved_1": [
        {
            "kind": "related_task_local_risk",
            "detail": "authority cost566 itself had 4993/5000 disable and 4998/5000 default fresh in the C9 audit",
            "source": "scripts/golf/loop_7999_13/lane_c9/REPORT.md",
        }
    ],
    "task361_cost844": [
        {
            "kind": "related_task_lb_version_history",
            "detail": "cost1461 was LB-white, then cheaper cost1445 was LB-black",
            "source": "docs/golf/private_zero_tasks.md",
        }
    ],
    "task365_cost1355": [
        {
            "kind": "related_task_direct_black_other_sha",
            "detail": "task365 is in the direct LB-black set for the 8006.61 second wave",
            "source": "others/71403/lb_verified_8006.61/MANIFEST.json",
        }
    ],
    "task396_cost_reduced": [
        {
            "kind": "related_task_lb_version_history",
            "detail": "cost1026 was LB-white, then cheaper cost982 was LB-black; task396 is a repeated black task",
            "source": "docs/golf/private_zero_tasks.md",
        }
    ],
    "task396_improved_v2": [
        {
            "kind": "related_task_lb_version_history",
            "detail": "cost1026 was LB-white, then cheaper cost982 was LB-black; task396 is a repeated black task",
            "source": "docs/golf/private_zero_tasks.md",
        }
    ],
}


def fresh_min(row: dict) -> dict | None:
    if not row.get("fresh_runs"):
        return None
    modes = []
    for run in row["fresh_runs"]:
        values = next(iter(run["candidates"].values()))
        for mode in ("disable_all", "default"):
            item = values[mode]
            modes.append(
                {
                    "seed": run["seed"],
                    "mode": mode,
                    "right": item["right"],
                    "wrong": item["wrong"],
                    "errors": item["errors"],
                }
            )
    return min(modes, key=lambda item: (item["right"], -item["wrong"], -item["errors"]))


def main() -> int:
    observed_root_sha = hashlib.sha256((ROOT / "submission.zip").read_bytes()).hexdigest()
    # Candidate rows were captured against AUTHORITY_SHA256 before another shared
    # lane changed the root pointer.  Finalization only combines those immutable
    # row files; it must not silently rebase them to the newly observed root.
    got = AUTHORITY_SHA256
    rows = []
    for label in ORDER:
        row = json.loads((HERE / "audit" / "rows" / f"{label}.json").read_text())
        decision, reason = DECISIONS[label]
        row["decision"] = decision
        row["decision_reason"] = reason
        row["fresh_worst_mode"] = fresh_min(row)
        row["exact_sha_lb_status"] = (
            "UNPROBED_NO_EXACT_LB_RECORD"
            if row.get("exact_sha_text_history")
            else "UNPROBED_NO_HISTORY"
        )
        row["related_history"] = RELATED_HISTORY.get(label, [])
        rows.append(row)

    probes = [row for row in rows if row["decision"].startswith("LB_PROBE_REQUIRED")]
    rejected = [row for row in rows if row["decision"].startswith("REJECT")]
    probe_priority = {
        "task310_improved_1": 1,
        "task396_cost_reduced": 2,
        "task396_improved_v2": 3,
    }
    probe_items = []
    for row in probes:
        probe_items.append(
            {
                "priority": probe_priority[row["label"]],
                "label": row["label"],
                "task": row["task"],
                "source": row["source"],
                "sha256": row["sha256"],
                "authority_cost": row["authority_profile"]["cost"],
                "candidate_cost": row["candidate_profile"]["cost"],
                "projected_gain": row["projected_gain"],
                "classification": row["decision"],
                "reason": row["decision_reason"],
                "fresh_worst_mode": row["fresh_worst_mode"],
                "exact_sha_lb_status": row["exact_sha_lb_status"],
                "related_history": row["related_history"],
            }
        )

    best_by_task = {}
    for row in probes:
        current = best_by_task.get(row["task"])
        if current is None or row["projected_gain"] > current["projected_gain"]:
            best_by_task[row["task"]] = row

    deep = {
        "authority_zip": "submission.zip",
        "authority_zip_sha256": got,
        "final_observed_root_sha256": observed_root_sha,
        "authority_drift_observed_after_audit": observed_root_sha != got,
        "candidate_count": len(rows),
        "rows": rows,
    }
    result = {
        "lane": "agent_71405_tail_99",
        "authority_zip": "submission.zip",
        "authority_zip_sha256": got,
        "final_observed_root_sha256": observed_root_sha,
        "authority_drift_observed_after_audit": observed_root_sha != got,
        "candidate_count": len(rows),
        "strict_lower_count": sum(bool(row["strict_lower_correct"]) for row in rows),
        "fixed_winner_count": 0,
        "fixed_projected_gain": 0.0,
        "probe_candidate_count": len(probe_items),
        "probe_candidate_labels": [item["label"] for item in probe_items],
        "best_mutually_exclusive_probe_gain": sum(
            row["projected_gain"] for row in best_by_task.values()
        ),
        "rejected_count": len(rejected),
        "root_modified_by_lane": False,
        "others_modified_by_lane": False,
        "zip_created": False,
        "tail_order_changed": False,
    }
    probe_manifest = {
        "authority_zip": "submission.zip",
        "authority_zip_sha256": got,
        "fixed_adoption_forbidden": True,
        "items": probe_items,
    }
    winner_manifest = {
        "authority_zip": "submission.zip",
        "authority_zip_sha256": got,
        "fixed_winners": [],
        "projected_gain": 0.0,
        "reason": "No candidate has exact LB-white proof; fixed adoption is forbidden.",
    }
    (HERE / "audit" / "deep_audit.json").write_text(json.dumps(deep, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(json.dumps(probe_manifest, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

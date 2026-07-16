#!/usr/bin/env python3
"""Assemble the fail-closed lane decision from independent and prior evidence."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (221, 136, 278, 230, 327, 391, 97, 27)


def main() -> None:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    by_task = {int(row["task"]): row for row in baseline["targets"]}
    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    c40 = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_c40/final_audit.json").read_text()
    )
    one_node = json.loads((HERE / "evidence/task327_one_node_infeasible.json").read_text())

    rules = {
        221: "Count zeros in the 3x3 input, derive the output size and tile counts from that value, then tile the input into the top-left region and zero-fill the rest.",
        136: "Extend the color-1 and color-2 2x2 objects on their prescribed opposite diagonals until the grid boundary.",
        278: "For each of three rotations, detect the specified local 2-pattern and paint its 3-valued neighborhood, then rotate back.",
        230: "For every 2x2 block of color 5, paint colors 1/2 above and 3/4 below at the corresponding corners.",
        327: "Convolve the 3x3 colored seed with a down-right diagonal ray to produce a 6x6 output.",
        391: "Count nonzero colors, remove the strict majority background color, and emit the remaining three colors in descending frequency.",
        97: "Remove colored cells that have no same-color neighbor in their 3x3 neighborhood.",
        27: "Preserve the input and paint the rule-selected region with color 2 using the generator's row/column phase relation.",
    }
    dispositions = {
        221: {
            "best_probe": "cost142 inherited shape-cloak edit",
            "reason": "Fails ORT_DISABLE_ALL on the complete known set through buffer-shape mismatch; correct controls cost at least 151. Latest cost144 graph is already one point below the older audited base and exact Wave2 found no further rewrite.",
        },
        136: {
            "best_probe": "cost135 tie",
            "reason": "Historical alternate only ties and retains a 58-input giant Einsum plus false runtime shapes; conventional controls start at cost1194.",
        },
        278: {
            "best_probe": "cost135 tie Min/Max rewrite",
            "reason": "The tie rewrite fails every known case under ORT_DISABLE_ALL from incompatible hidden allocations. All conventional archived controls exceed current cost135.",
        },
        230: {
            "best_probe": "cost108 tie / cost900 truthful direct Conv",
            "reason": "The cost108 family has a dynamic/external Conv bias, so Conv-bias UB0 cannot be proved; the exact Wave2 structural gate rejects it. The truthful direct Conv is much larger.",
        },
        327: {
            "best_probe": "cost106 archive tie; proposed cost46 one-node ConvTranspose",
            "reason": "Archive minimum ties current cost106. The complete-placement cost46 architecture is infeasible because an identical local patch with the same bias kind requires opposite labels; no ONNX candidate can be emitted.",
        },
        391: {
            "best_probe": "cost85/87/88 lookup models",
            "reason": "Every below-104 model uses three TfIdfVectorizer lookup tables and belongs to the documented private-zero replacement lineage. The smallest table-free truthful TopK engine costs139, already above the current104 floor.",
        },
        97: {
            "best_probe": "cost100 depthwise/local controls",
            "reason": "All valid archived implementations tie cost100 or cost910; no strict cost decrease exists. Latest exact Wave2 found no exact rewrite.",
        },
        27: {
            "best_probe": "cost96 alternate giant-Einsum tie",
            "reason": "The only low archived alternate ties cost96 and retains a 55-input giant Einsum; the truthful conventional implementation costs962 or more. No strict decrease exists.",
        },
    }

    rows = []
    for task in TARGETS:
        current = by_task[task]
        rows.append(
            {
                "task": task,
                "rule_summary": rules[task],
                "baseline_member": current["member"],
                "baseline_sha256": current["sha256"],
                "baseline_cost": current["actual_cost"],
                "unchanged_from_8004_50": current["unchanged_from_8004_50"],
                "structure_observation": {
                    "runtime_shape_cloak": current["structure"]["runtime_shape_trace"].get(
                        "shape_cloak", False
                    ),
                    "runtime_shape_mismatches": current["structure"]["runtime_shape_trace"].get(
                        "mismatch_count"
                    ),
                    "giant_einsum": current["structure"]["giant_einsum"],
                    "lookup_or_scatter": current["structure"]["lookup_or_scatter_nodes"],
                },
                **dispositions[task],
                "candidate": None,
                "decision": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
                "known_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_1_dual": "NOT_RUN_PRE_GATE_FAILED",
                "fresh_seed_2_dual": "NOT_RUN_PRE_GATE_FAILED",
                "conv_bias_ub0": "NO_CANDIDATE",
                "projected_gain": 0.0,
            }
        )

    history = {
        "latest_baseline": baseline["baseline"],
        "all_members_unchanged_from_8004_50": all(
            row["unchanged_from_8004_50"] for row in baseline["targets"]
        ),
        "exact_wave2": baseline["exact_wave2"],
        "harvest_inventory": {
            "baseline": harvest["baseline_zip"],
            "target_rows": {
                str(task): [row for row in harvest["rows"] if row.get("task") == task]
                for task in TARGETS
            },
        },
        "all400_archive_inventory": {
            "baseline": archive["base"],
            "stats": archive["stats"],
            "zip_errors": archive["zip_errors"],
            "retained_below_old_baseline": {
                str(task): archive["retained"].get(str(task), []) for task in TARGETS
            },
        },
        "task327_one_node": one_node,
        "task391_sound_control": c40["sound_floor"],
        "task391_private_zero_evidence": c40["lb_and_quarantine_evidence"],
    }
    result = {
        "lane": "agent_new_low32",
        "baseline": baseline["baseline"],
        "policy": {
            "nonprivate": "strictly cheaper, known100 dual, two independent fresh seeds each >=90%, runtime0, strict/data_prop, truthful shapes, standard domains, no lookup/cloak/giant, Conv-bias UB0",
            "private_risk": "same gates but known/fresh must be 100% dual and the model must be demonstrably SOUND",
            "fail_closed": "known/fresh are not run if price or structural gates fail",
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
    (HERE / "history_audit.json").write_text(json.dumps(history, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text("[]\n")
    print(f"completed={len(rows)} accepted=0 gain=0.0")


if __name__ == "__main__":
    main()

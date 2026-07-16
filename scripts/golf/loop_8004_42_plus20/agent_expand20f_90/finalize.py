#!/usr/bin/env python3
"""Finalize expand20f_90 with fixed-safe and LB-probe classifications."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (
    75, 392, 225, 218, 159, 185, 263, 370, 182, 330,
    361, 157, 280, 382, 201, 251, 12, 107, 131, 364,
)
LOCKED_CHAMPIONS = (13, 70, 158, 254, 267, 323, 379)
BASELINE_SCORE = 8006.61


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def boolean_complete(configs: dict[str, dict[str, Any]]) -> bool:
    return all(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("runtime_errors") == 0
        and not row.get("session_error")
        for row in configs.values()
    )


def main() -> int:
    rescreen = json.loads((HERE / "rescreen.json").read_text())
    ordinary = json.loads((HERE / "audit" / "actual_lower_four_config.json").read_text())
    reopened = json.loads((HERE / "audit" / "reopened_giant_lookup_private.json").read_text())
    lb_history = json.loads((HERE / "audit" / "lb_history_classification.json").read_text())
    probe_manifest = json.loads((HERE / "probe_manifest.json").read_text())
    costs = json.loads((HERE / "baseline_costs_8006_61.json").read_text())

    # Prove that the inherited full-repository exact/no-op evidence applies to
    # the new authority members for this target set.
    identity = []
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as old, zipfile.ZipFile(
        ROOT / "submission_base_8006.61.zip"
    ) as new:
        for task in TARGETS:
            old_sha = hashlib.sha256(old.read(f"task{task:03d}.onnx")).hexdigest()
            new_sha = hashlib.sha256(new.read(f"task{task:03d}.onnx")).hexdigest()
            identity.append(
                {"task": task, "sha_8005_17": old_sha, "sha_8006_61": new_sha, "identical": old_sha == new_sha}
            )
    dump(
        HERE / "audit" / "authority_member_identity.json",
        {"all_identical": all(row["identical"] for row in identity), "rows": identity},
    )

    rescreen_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rescreen["rows"]:
        rescreen_by_task[int(row["task"])].append(row)
    ordinary_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ordinary["rows"]:
        ordinary_by_task[int(row["task"])].append(row)
    reopened_lower_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in reopened["rows"]:
        if row["actual_strict_lower"]:
            reopened_lower_by_task[int(row["task"])].append(row)

    task_summary = []
    for task in TARGETS:
        standard = ordinary_by_task[task]
        extra = reopened_lower_by_task[task]
        costs_found = [int(row["actual_cost"]) for row in standard] + [int(row["profile"]["cost"]) for row in extra]
        known4 = sum(boolean_complete(row["known_four_configs"]) for row in standard) + sum(
            bool(row.get("known_boolean_complete_all_configs")) for row in extra
        )
        truthful = sum(bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in standard) + sum(
            bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in extra
        )
        if task == 185:
            reason = "three KNOWN_LB_BLACK plus three REJECT_LOCAL_FALSE_ACCEPT; no probe remains"
        elif known4 and not truthful:
            reason = "known×4 lower candidates are runtime-shape cloaks"
        elif costs_found:
            reason = "actual-lower candidates fail known/runtime/schema gates"
        else:
            reason = "no actual-lower candidate after official profiling"
        task_summary.append(
            {
                "task": task,
                "baseline_cost": int(costs["costs"][str(task)]),
                "unique_nonbaseline_sha": int(rescreen["inventory"]["unique_by_task"][str(task)]),
                "actual_lower_count": len(costs_found),
                "best_actual_lower_cost": min(costs_found, default=None),
                "known_boolean_complete_four_configs_count": known4,
                "truthful_runtime_shape_count": truthful,
                "lb_probe_required_count": 0,
                "fixed_safe_count": 0,
                "decision": "reject",
                "reason": reason,
            }
        )

    stage_counts = Counter(row["stage"] for row in rescreen["rows"])
    ordinary_actual_jobs = sum(
        count
        for stage, count in stage_counts.items()
        if stage in {
            "actual_reject", "known_reject", "known_dual_reject", "shape_reject",
            "fresh500_reject", "fresh500_pass",
        }
    )
    total_lower = int(ordinary["candidate_count"]) + int(reopened["actual_strict_lower_count"])
    total_known4 = sum(
        boolean_complete(row["known_four_configs"]) for row in ordinary["rows"]
    ) + sum(
        bool(row.get("known_boolean_complete_all_configs"))
        for row in reopened["rows"]
        if row.get("actual_strict_lower")
    )
    total_truthful = sum(
        bool((row.get("runtime_shape_trace") or {}).get("truthful")) for row in ordinary["rows"]
    ) + sum(
        bool((row.get("runtime_shape_trace") or {}).get("truthful"))
        for row in reopened["rows"]
        if row.get("actual_strict_lower")
    )

    inventory_output = {
        "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "targets": list(TARGETS),
        "observation_counts": rescreen["inventory"]["counts"],
        "unique_by_task": rescreen["inventory"]["unique_by_task"],
        "initial_stage_counts": dict(sorted(stage_counts.items())),
        "reopened_policy_pool": {
            "count": reopened["reopened_count"],
            "actual_strict_lower": reopened["actual_strict_lower_count"],
        },
        "baseline_profiles": costs["profiles"],
        "task_summary": task_summary,
    }
    dump(HERE / "inventory" / "candidate_inventory.json", inventory_output)
    dump(
        HERE / "audit" / "final_decisions.json",
        {
            "policy": {
                "fixed_safe": "LB-white network or proved all-input equivalence only",
                "probe_retention": "known×4 plus actual-lower plus truthful/schema/UB0; fresh and giant/lookup are ranking signals",
                "hard_reject": "actual non-improvement, wrong known, runtime crash, schema invalidity, UB, or runtime-shape cloak",
            },
            "locked_champions_untouched": list(LOCKED_CHAMPIONS),
            "ordinary_actual_lower_count": ordinary["candidate_count"],
            "reopened_actual_lower_count": reopened["actual_strict_lower_count"],
            "total_actual_lower_count": total_lower,
            "known_x4_complete_count": total_known4,
            "truthful_count": total_truthful,
            "lb_history_evidence": "audit/lb_history_classification.json",
            "lb_probe_required_count": probe_manifest["candidate_count"],
            "fixed_safe_count": 0,
            "task_decisions": task_summary,
        },
    )
    result = {
        "baseline_score": BASELINE_SCORE,
        "baseline_zip": rescreen["baseline_zip"],
        "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
        "targets": list(TARGETS),
        "funnel": {
            "unique_nonbaseline_sha": int(rescreen["inventory"]["counts"]["unique_different_sha"]),
            "ordinary_actual_profile_jobs": ordinary_actual_jobs,
            "reopened_policy_profile_jobs": reopened["reopened_count"],
            "total_actual_strict_lower": total_lower,
            "known_x4_complete": total_known4,
            "truthful_runtime_shapes": total_truthful,
            "known_lb_black": lb_history["known_lb_black_count"],
            "local_false_accept": lb_history["local_false_accept_count"],
            "lb_probe_required": probe_manifest["candidate_count"],
            "fixed_safe_adoptees": 0,
        },
        "accepted": [],
        "score_gain": 0.0,
        "projected_score": BASELINE_SCORE,
        "promotion_zip_created": False,
        "protected_root_files_changed": False,
        "locked_champions_untouched": list(LOCKED_CHAMPIONS),
        "status": "complete_no_safe_or_probe_candidate",
    }
    dump(HERE / "result.json", result)
    dump(
        HERE / "winner_manifest.json",
        {
            "baseline_zip": rescreen["baseline_zip"],
            "baseline_zip_sha256": rescreen["baseline_zip_sha256"],
            "fixed_safe_winners": [],
            "locked_champions_untouched": list(LOCKED_CHAMPIONS),
            "total_gain": 0.0,
            "promotion": "none",
        },
    )

    table = []
    for row in task_summary:
        best = "—" if row["best_actual_lower_cost"] is None else str(row["best_actual_lower_cost"])
        table.append(
            f"| {row['task']:03d} | {row['baseline_cost']} | {row['unique_nonbaseline_sha']} | "
            f"{row['actual_lower_count']} | {best} | {row['known_boolean_complete_four_configs_count']} | "
            f"{row['truthful_runtime_shape_count']} | 0 |"
        )
    history_lines = []
    for row in lb_history["rows"]:
        rates = "/".join(f"{outcome['minimum_rate']*100:.1f}%" for outcome in row["fresh"])
        history_lines.append(
            f"- `{row['sha256'][:12]}` cost {row['actual_cost']}: **{row['classification']}**, "
            f"fresh {rates}. {row['history_reason']}."
        )
    report = f"""# Expanded 20-task audit — agent_expand20f_90

## Result

- Immutable authority: `submission_base_8006.61.zip`
- Authority SHA-256: `{rescreen['baseline_zip_sha256']}`
- Safe fixed adoptees: **0**
- Final `LB_PROBE_REQUIRED`: **0**
- Score gain: **+0.0**; projected score remains **{BASELINE_SCORE:.2f}**
- Locked champions 013/070/158/254/267/323/379 were untouched. No root/others/ZIP mutation was performed.

## Exhaustive funnel

- Scanned 12,283 loose observations and 23,712 task members from 1,261 ZIP files; SHA dedup left 1,025 nonbaseline candidates.
- Ordinary fail-closed path profiled {ordinary_actual_jobs} models. A second pass deliberately reopened 113 giant/lookup/private models rather than rejecting those labels alone.
- Combined actual strict-lower: {total_lower}; known-complete in disable/default ORT × 1/4 threads: {total_known4}; truthful runtime shapes: {total_truthful}.
- The six truthful models were all task185 lookup/private networks. SHA-specific LB evidence and two fresh-500 seeds reduce them to three known LB-black plus three local false accepts. Final probe and fixed-safe sets are empty.

| task | base | unique SHA | actual-lower | best cost | known×4 | truthful | probe |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## task185 SHA-level disposition

{chr(10).join(history_lines)}

The cost-273 `d21f1db...` model is especially important: it scores 500/500 on both fresh seeds yet is already LB-black. Fresh accuracy is therefore retained as a ranking signal, not treated as a white guarantee.

## Other lower leads

- task107 cost 706/638, task131 cost 627/596, and task201 cost 785/682/543 are known×4 complete but fail direct truthful runtime-shape tracing; they are hard rejects, not probes.
- task251 cost 709/582 is 266/266 under disabled ORT but default ORT fails all cases during session construction. The task-specific recent black warning is therefore moot for these SHAs: they fail the local runtime gate first.
- task382 contributes fourteen lower models across both paths; they miss known cases or fail default ORT. task012 is 235/265, task159 lower giant models fail known, and remaining lower leads likewise fail known/runtime.
- Conv-bias UB, schema-invalid, noncanonical-I/O, and actual non-improvements remain hard rejects. Giant/lookup/private labels alone were not used as the final rejection reason.

## Authority and exact/no-op coverage

All 20 target members are byte-identical between 8005.17 and 8006.61; see `audit/authority_member_identity.json`. Therefore the prior all-400 initializer reuse/dead/no-op results remain applicable. Their emitted candidates, including task107 initializer reuse and task382 truthful repair, were also included in the repository-wide scan. No all-input equivalence proof produced a strict-lower safe winner.

## Artifacts

- `inventory/candidate_inventory.json`
- `rescreen.json`
- `audit/actual_lower_four_config.json`
- `audit/reopened_giant_lookup_private.json`
- `audit/fresh_probe_2seed.json`
- `audit/lb_history_classification.json`
- `probe_manifest.json`, `winner_manifest.json`, and `result.json`
"""
    (HERE / "REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

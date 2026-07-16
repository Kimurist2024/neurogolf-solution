#!/usr/bin/env python3
"""Consolidate existing exhaustive search evidence for the low45 targets."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = {24, 113, 385, 389, 296, 399, 359, 110}


def main() -> None:
    archive = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json").read_text()
    )
    retained = [
        row
        for task, rows in archive["retained"].items()
        if int(task) in TARGETS
        for row in rows
    ]
    quick = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/lower_quick_k20.json").read_text()
    )
    quick_rows = []
    for row in quick:
        if row.get("task") not in TARGETS:
            continue
        result = row.get("result") or {}
        quick_rows.append(
            {
                "task": row["task"],
                "path": row.get("path"),
                "baseline_cost": row.get("baseline_cost"),
                "result": {
                    "cost": result.get("cost"),
                    "lib_gold": result.get("lib_gold"),
                    "official_gold": result.get("official_gold"),
                    "fresh_total": result.get("fresh_total"),
                    "fresh_fails": result.get("fresh_fails"),
                    "fresh_rate": result.get("fresh_rate"),
                    "runtime_exception": row.get("runtime_exception"),
                },
            }
        )

    harvest = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_harvest/scan_results.json").read_text()
    )
    harvest_rows = [row for row in harvest["rows"] if row.get("task") in TARGETS]

    b23 = json.loads(
        (ROOT / "scripts/golf/loop_7999_13/lane_b23/scan_build_manifest.json").read_text()
    )
    b23_rows = [row for row in b23["tasks"] if row.get("task") in TARGETS]

    exact = json.loads(
        (ROOT / "scripts/golf/loop_8004_42_plus20/agent_exact_wave2/static_scan.json").read_text()
    )
    exact_hits = []
    for section in ("baseline_structural_failures", "candidates"):
        for row in exact.get(section, []):
            if row.get("task") in TARGETS:
                exact_hits.append({"section": section, **row})
    for kind, rows in exact.get("opportunities", {}).items():
        for row in rows:
            if row.get("task") in TARGETS:
                exact_hits.append({"section": f"opportunities:{kind}", **row})

    evidence = {
        "targets": sorted(TARGETS),
        "archive_all400": {
            "stats": archive["stats"],
            "strict_lower_retained": retained,
            "strict_lower_count": len(retained),
            "quick_screen": quick_rows,
            "interpretation": "Only task385 (five malformed/incorrect static-cost artifacts) and task389 (one incorrect cost-20 artifact) survive the numeric prefilter; every one is wrong on known and all 20 fresh cases.",
        },
        "focused_harvest": {
            "counts": harvest["inventory"]["counts"],
            "target_rows": harvest_rows,
            "interpretation": "No focused-harvest graph establishes a safe strict decrease. Rows absent here had no unique non-baseline lead after the harvest gates.",
        },
        "exact_initializer_alias_b23": {
            "scope": b23["scope"],
            "target_rows": b23_rows,
            "built_for_targets": sum(row.get("built_candidates", 0) for row in b23_rows),
        },
        "exact_wave2": {
            "summary": exact["summary"],
            "target_hits": exact_hits,
            "interpretation": "The all-400 exact rewrite pass found no opportunity or candidate for these eight current members.",
        },
        "task_specific_proofs": {
            "024": "scripts/golf/scratch_codex/task024/REPORT.md and FAILURE_LOG.md (one-output factor searches; later cost-30 incumbent is below the historical cost-56 floor)",
            "113": "scripts/golf/scratch_codex/task113/REPORT.md and FAILURE_LOG.md (268,968 extra one-node probes plus prior 1,058/13,458/15,528-family screens; no sub-30 survivor)",
            "385": "scripts/golf/scratch_codex/task385/REPORT.md and FAILURE_LOG.md; archive lower_quick_k20.json rejects every apparent sub-30 artifact",
            "389": "scripts/golf/scratch_codex/task389/REPORT.md and FAILURE_LOG.md; cost-20 Einsum approximation is known/fresh false",
            "296": "scripts/golf/scratch_codex/task296/REPORT.md and FAILURE_LOG.md (factored ConvTranspose area <=17 has no solution; area 18 reproduces the cost-28 tie)",
            "399": "scripts/golf/scratch_codex/task399/REPORT.md and FAILURE_LOG.md (exhaustive scalar pattern searches; later cost-25 incumbent is below the historical cost-34 floor)",
            "359": "scripts/golf/loop_8003_40/agent_task359_rebuild/REPORT.md and STRUCTURAL_AUDIT.json (truthful standard-operator rule floor exceeds giant-Einsum cost 24)",
            "110": "scripts/golf/scratch/task110/FAILURE_LOG.md; focused harvest finds only giant-Einsum or cost-dominated alternatives to cost 24",
        },
        "conclusion": "NO_SAFE_STRICTLY_CHEAPER_CANDIDATE",
    }
    (HERE / "history_audit.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()

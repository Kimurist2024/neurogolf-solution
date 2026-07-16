#!/usr/bin/env python3
"""Finalize the A38 no-winner exact-CSE audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"
AUTHORITY_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"


def main() -> None:
    authority_hash = hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()
    if authority_hash != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP changed")
    scan = json.loads((HERE / "scan_build_manifest.json").read_text())
    opportunities = json.loads((HERE / "raw_opportunity_audit.json").read_text())
    if scan["candidate_count"] != 0 or scan["candidates"]:
        raise RuntimeError("unexpected safe candidate; validation is required before finalizing")
    if opportunities["safe_opportunity_tasks"]:
        raise RuntimeError("safe raw opportunities were not resolved")
    exclusion_counts = Counter(
        reason
        for row in scan["excluded"]
        for reason in row["reasons"]
    )
    result = {
        "lane": "a38",
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "authority_zip_modified": False,
        "scope": {
            "tasks_scanned": 400,
            "priority": "task150-400 first in candidate ordering",
            "exact_transforms": [
                "byte-identical initializer alias CSE",
                "normalized identical Constant payload CSE",
                "reachable deterministic single-output subgraph CSE",
            ],
        },
        "safe_models_scanned": 400 - int(scan["excluded_count"]),
        "excluded_models": int(scan["excluded_count"]),
        "exclusion_counts": dict(exclusion_counts.most_common()),
        "raw_opportunities": {
            "task_count": opportunities["opportunity_task_count"],
            "safe_tasks": opportunities["safe_opportunity_tasks"],
            "excluded_tasks": opportunities["excluded_opportunity_tasks"],
            "initializer_aliases": sum(int(row["initializer_aliases"]) for row in opportunities["rows"]),
            "constant_payload_cse": sum(int(row["constant_payload_cse"]) for row in opportunities["rows"]),
            "deterministic_node_cse": sum(int(row["deterministic_node_cse"]) for row in opportunities["rows"]),
        },
        "excluded_opportunity_details": [
            {
                "task": row["task"],
                "reasons": row["exclusion_reasons"],
                "initializer_aliases": row["initializer_aliases"],
                "constant_payload_cse": row["constant_payload_cse"],
                "deterministic_node_cse": row["deterministic_node_cse"],
            }
            for row in opportunities["rows"]
        ],
        "decision": "NO_ADOPTABLE_CANDIDATE",
        "candidate_count": 0,
        "known_dual": "not_run_no_lower_cost_safe_candidate",
        "fresh_dual": "not_run_no_lower_cost_safe_candidate",
        "external500": "not_run_no_lower_cost_safe_candidate",
        "score_gain": 0.0,
        "evidence": [
            "scan_build_manifest.json",
            "raw_opportunity_audit.json",
            "scan_build_exact_cse.py",
            "audit_raw_opportunities.py",
        ],
    }
    (HERE / "A38_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    excluded_tasks = ", ".join(f"task{task:03d}" for task in opportunities["excluded_opportunity_tasks"])
    report = f"""# A38 exact CSE result

All 400 members of `submission_base_8000.46.zip` were scanned for byte-identical initializer aliases, normalized identical Constant payloads, and duplicate reachable deterministic subgraphs.

- Shape/lineage-safe models scanned: **{result['safe_models_scanned']}**
- Explicitly excluded unsafe-lineage models: **{result['excluded_models']}**
- Lower-cost safe candidates: **0**
- Decision: **NO_ADOPTABLE_CANDIDATE**
- Score gain: **0**

The raw inventory found exact duplication only in {excluded_tasks}. All four are excluded by the requested policy: CenterCropPad lineage in every case, with lookup lineage additionally present in task165/task233. task162's 60 duplicate nodes are themselves CenterCropPad nodes. No identical Constant payload opportunity exists in any of the 400 authority models, and no safe model contains an initializer or reachable deterministic-subgraph CSE opportunity.

Because no lower-cost shape-truthful safe candidate exists, known-dual, fresh-dual, and external500 gates were not run. The authority ZIP SHA-256 remains `{AUTHORITY_SHA256}` and no shared submission was modified.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "decision": result["decision"],
        "safe_models_scanned": result["safe_models_scanned"],
        "excluded_models": result["excluded_models"],
        "excluded_opportunity_tasks": opportunities["excluded_opportunity_tasks"],
        "score_gain": 0.0,
    }, indent=2))


if __name__ == "__main__":
    main()

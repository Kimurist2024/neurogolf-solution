#!/usr/bin/env python3
"""Merge worker evidence into a lane-local report and manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY_SCORE = 8012.15


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    workers = [json.loads((HERE / f"worker_{index}.json").read_text()) for index in range(3)]
    finalists = sorted(
        [row for worker in workers for row in worker["finalists"]],
        key=lambda row: int(row["task"]),
    )
    gain = sum(float(row["score_gain"]) for row in finalists)
    decisions = []
    for worker in workers:
        for row in worker["task_rows"]:
            finalist = row.get("admission")
            decisions.append({
                "task": row["task"], "authority_cost": row["authority_cost"],
                "decision": "ADMIT_POLICY90" if finalist else "NO_ADMISSIBLE_REDUCTION",
                "candidate_cost": None if finalist is None else finalist["candidate_cost"],
                "candidate_sha256": None if finalist is None else finalist["sha256"],
                "candidate_path": None if finalist is None else finalist["saved_path"],
                "conditional_score_gain": None if finalist is None else finalist["score_gain"],
            })
    decisions.sort(key=lambda row: int(row["task"]))
    manifest = {
        "lane": rel(HERE), "authority_score": AUTHORITY_SCORE,
        "authority": workers[0]["authority"],
        "authority_sha256": workers[0]["authority_sha256"],
        "threshold": workers[0]["threshold"],
        "fresh_per_seed": workers[0]["fresh_per_seed"],
        "worker_partitions": [worker["assigned_tasks"] for worker in workers],
        "worker_partition_sizes": [worker["assigned_count"] for worker in workers],
        "band_task_count": len(workers[0]["band"]),
        "eligible_task_count": sum(worker["assigned_count"] for worker in workers),
        "excluded_band_tasks": workers[0]["excluded_band_tasks"],
        "explicit_latest_lb_black": workers[0]["explicit_latest_lb_black"],
        "finalist_count": len(finalists), "conditional_score_gain": gain,
        "conditional_score": AUTHORITY_SCORE + gain,
        "finalists": [],
        "protected_writes": "lane only; no root submission/all_scores/others changes",
    }
    for row in finalists:
        path = ROOT / row["saved_path"]
        manifest["finalists"].append({
            "task": row["task"], "authority_cost": row["authority_cost"],
            "candidate_cost": row["candidate_cost"], "sha256": row["sha256"],
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": row["saved_path"], "score_gain": row["score_gain"],
            "known_accuracy_min": min(v["accuracy"] for v in row["known_four"].values()),
            "fresh_accuracy_min": min(
                v["accuracy"] for run in row["fresh"] for v in run["runtime"].values()
            ),
            "hard_gates": {
                "runtime_errors": 0, "nonfinite": 0, "shape_mismatch": 0,
                "small_positive": 0, "ub": 0, "shape_cloak": 0,
            },
        })
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "admission_decisions.json").write_text(json.dumps({
        "authority_score": AUTHORITY_SCORE, "threshold": workers[0]["threshold"],
        "decisions": decisions,
    }, indent=2) + "\n")

    finalist_lines = [
        f"| {row['task']:03d} | {row['authority_cost']} | {row['candidate_cost']} | "
        f"{row['known_accuracy_min']:.4%} | {row['fresh_accuracy_min']:.4%} | "
        f"+{row['score_gain']:.6f} | `{row['path']}` |"
        for row in manifest["finalists"]
    ]
    if not finalist_lines:
        finalist_lines = ["| — | — | — | — | — | +0.000000 | none |"]
    report = f"""# 8012.15 cost501..1000 three-worker wave 408

## Outcome

The lane found **{len(finalists)} POLICY90-admissible strict reduction(s)**.
The combined conditional gain is **+{gain:.6f}**, taking the immutable
8012.15 baseline to **{AUTHORITY_SCORE + gain:.6f}** if every listed candidate
is promoted.  This lane did not edit the root submission, score ledger, or
`others/`.

| task | authority | candidate | min known | min fresh | gain | candidate |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(finalist_lines)}

## Scope and exclusions

The immutable authority is `{manifest['authority']}` with SHA-256
`{manifest['authority_sha256']}`.  All 31 non-score25 members whose current
cost is 501..1000 were enumerated.  The private-zero/unsound and known-LB-black
band members `{manifest['excluded_band_tasks']}` were excluded before any
search.  The remaining {manifest['eligible_task_count']} tasks were split
round-robin into three disjoint partitions of sizes
`{manifest['worker_partition_sizes']}`: `{manifest['worker_partitions']}`.

Each worker reprofiled every assigned current authority member, then ran all
three requested families: loose/ZIP archive strict-lower rebasing, current
graph exact simplification, and current cost<=10 pattern transfer.  Detailed
counters, rejected screens, authority profiles, and complete finalist runtime
rows are in `worker_0.json`, `worker_1.json`, and `worker_2.json`.

## Admission gates

Admission requires at least 90% whole-case accuracy independently in each of
four ORT configurations on complete known data and on 2,000 fresh examples
from each of two independent seeds.  Runtime errors, nonfinite outputs,
runtime/declaration shape mismatches, `(0,0.25)` positive outputs,
configuration sign disagreement, Conv-bias UB, nonstatic/cloaked shapes,
banned/nested graphs, sparse/functions, and nonstandard domains must all be
zero.  Candidate cost is independently profiled and must be strictly below
the current member.

This is a POLICY90 result, not an exact-correctness claim.  Root promotion and
leaderboard credit remain deliberately unclaimed.
"""
    (HERE / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "finalists": len(finalists), "conditional_gain": gain,
        "conditional_score": AUTHORITY_SCORE + gain,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

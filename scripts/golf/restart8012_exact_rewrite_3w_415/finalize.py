#!/usr/bin/env python3
"""Finalize the evidence-only Wave415 handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    scan = json.loads((HERE / "scan_evidence.json").read_text())
    history = json.loads((HERE / "history_evidence.json").read_text())
    task268_path = ROOT / "scripts/golf/agent_cost251_500_half_307/task268_cost327_rebase8012_policy95_audit.json"
    task268 = json.loads(task268_path.read_text())
    rows = []
    baseline_failures = 0
    for worker_id in range(3):
        worker = json.loads((HERE / f"worker_{worker_id}.json").read_text())
        baseline_failures += len(worker["baseline_failures"])
        rows.extend(worker["rows"])

    if scan["finalists"]:
        raise RuntimeError("mechanical finalist unexpectedly present")
    if len(history["finalists"]) != 1 or int(history["finalists"][0]["task"]) != 268:
        raise RuntimeError("unexpected history finalist set")
    if task268["classification"] != "REJECT" or task268["fresh_pass"]:
        raise RuntimeError("task268 prior rejection invariant failed")

    protected = {
        "submission.zip": sha256(ROOT / "submission.zip"),
        "submission_base_8012.15.zip": sha256(ROOT / "submission_base_8012.15.zip"),
        "all_scores.csv": sha256(ROOT / "all_scores.csv"),
    }
    authority_sha = scan["authority"]["sha256"]
    if protected["submission.zip"] != authority_sha or protected["submission_base_8012.15.zip"] != authority_sha:
        raise RuntimeError("root authority changed")

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    rejected_history = history["finalists"][0]
    evidence = {
        "wave": "restart8012_exact_rewrite_3w_415",
        "decision": "NO_NEW_ADMISSION",
        "authority": scan["authority"],
        "worker_contract": {
            "internal_workers": 3,
            "partitions": [len(partition) for partition in scan["partitions"]],
            "task_count": len(scan["tasks"]),
        },
        "mechanical_scan": {
            "cost_range": [100, 500],
            "excluded_private_zero": scan["policy"]["excluded_private_zero"],
            "excluded_known_black": scan["policy"]["excluded_known_black"],
            "excluded_already_admitted": scan["policy"]["excluded_already_admitted"],
            "baseline_strict_or_canonical_failures": baseline_failures,
            "strict_canonical_members_processed": len(scan["tasks"]) - baseline_failures,
            "candidate_variants": len(rows),
            "status_counts": status_counts,
            "admissions": [],
        },
        "history_raw_rebase": {
            "actual_lower_rows_rechecked": history["source_rows"],
            "known_raw_exact_hits": 1,
            "admissions": [],
            "rejected": [{
                "task": 268,
                "authority_cost": rejected_history["current_authority_cost"],
                "candidate_cost": rejected_history["official_profile"]["cost"],
                "candidate_sha256": rejected_history["sha256"],
                "candidate_path": rejected_history["path"],
                "known_raw_cases": rejected_history["known_raw_comparison"]["checked"],
                "fresh_seed_rates": [
                    row["accuracy"] for row in task268["fresh_four_configs_two_seeds"]
                    if "accuracy" in row and row["optimization"] == "disabled" and row["threads"] == 1
                ],
                "default_ort_session_errors": sum(
                    "session_error" in row for row in task268["fresh_four_configs_two_seeds"]
                ),
                "reason": "fresh below POLICY90 and default ORT session construction fails",
                "prior_audit": relative(task268_path),
            }],
        },
        "near_miss_shape_proofs": [
            {"task": row["task"], "label": row["label"], "status": row["status"]}
            for row in rows if row["status"] == "known_raw_mismatch"
        ],
        "protected_root_sha256": protected,
        "root_or_others_modified": False,
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    candidate = HERE / "candidates/history_worker_1/task268_22ea97ffce8b.onnx"
    manifest = {
        "authority_sha256": authority_sha,
        "decision": "NO_NEW_ADMISSION",
        "admitted": [],
        "rejected_artifacts": [{
            "task": 268,
            "path": relative(candidate),
            "sha256": sha256(candidate),
            "cost": 327,
            "classification": "REJECT_POLICY90_AND_DEFAULT_ORT",
        }],
        "evidence": {
            name: {"path": relative(HERE / name), "sha256": sha256(HERE / name)}
            for name in ("scan_evidence.json", "history_evidence.json", "evidence.json")
        },
        "protected_root_sha256": protected,
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

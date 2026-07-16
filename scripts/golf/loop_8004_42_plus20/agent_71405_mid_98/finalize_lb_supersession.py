#!/usr/bin/env python3
"""Finalize lane 98 using the exact-SHA LB-verified 8008.14 authority."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LB_ZIP = ROOT / "submission_base_8008.14.zip"
LB_SHA256 = "50b3215030cf506f692af50e41203d805256a250bd29882cc777749767a350c6"
LB_MD5 = "db4da5cc59186b26572a380725bc2fdf"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    raw = LB_ZIP.read_bytes()
    if sha256(raw) != LB_SHA256 or hashlib.md5(raw).hexdigest() != LB_MD5:  # noqa: S324 - integrity identifier
        raise RuntimeError("8008.14 LB authority hash mismatch")
    pre = json.loads((HERE / "audit/pre_fresh_screen.json").read_text())
    rows = pre["rows"]
    memberships = []
    winners = []
    with zipfile.ZipFile(LB_ZIP) as archive:
        if len(archive.namelist()) != 400:
            raise RuntimeError("8008.14 archive does not have 400 members")
        for row in rows:
            member = f"task{int(row['task']):03d}.onnx"
            member_data = archive.read(member)
            member_sha = sha256(member_data)
            exact = member_sha == row["sha256"]
            item = {
                "task": int(row["task"]),
                "candidate_source": row["source"],
                "candidate_sha256": row["sha256"],
                "lb_member": member,
                "lb_member_sha256": member_sha,
                "exact_member_match": exact,
                "local_classification_before_lb": row["classification"],
                "local_diagnostics_retained": {
                    "structural_hard_failures": row["structural_audit"].get("hard_failures", []),
                    "known_four": row.get("known_four"),
                    "runtime_shape_mismatches": row.get("runtime_shape_trace", {}).get("declared_actual_mismatches", []),
                },
                "final_classification": "LB_WHITE_FIXED" if exact else row["classification"],
            }
            memberships.append(item)
            if exact:
                winners.append({
                    "task": int(row["task"]),
                    "sha256": row["sha256"],
                    "source": row["source"],
                    "lb_path": f"submission_base_8008.14.zip::{member}",
                    "old_authority_cost": int(row["authority_profile"]["cost"]),
                    "candidate_cost": int(row["candidate_profile"]["cost"]),
                    "gain_from_8006_61_member": row["gain"],
                    "private_high_risk": bool(row["private_high_risk"]),
                    "local_classification_before_lb": row["classification"],
                    "classification": "LB_WHITE_FIXED",
                    "evidence": "exact SHA is present in LB-verified submission_base_8008.14.zip",
                })
    # Two task125 files were inspected; only v2 is the exact LB member.
    if len(winners) != 10 or len({item["task"] for item in winners}) != 10:
        raise RuntimeError(f"expected ten exact LB-white task members, got {len(winners)}")
    winners.sort(key=lambda item: item["task"])
    audit = {
        "rule": "exact LB-verified ZIP membership supersedes local false rejection",
        "lb_score": 8008.14,
        "lb_zip": "submission_base_8008.14.zip",
        "lb_zip_sha256": LB_SHA256,
        "lb_zip_md5": LB_MD5,
        "member_count": 400,
        "inspected_file_count": len(rows),
        "exact_lb_white_file_count": len(winners),
        "exact_lb_white_task_count": len({item["task"] for item in winners}),
        "memberships": memberships,
    }
    (HERE / "audit/lb_8008_14_supersession.json").write_text(json.dumps(audit, indent=2) + "\n")

    winner_manifest = {
        "status": "LB_WHITE_FIXED_WINNERS",
        "lb_score": 8008.14,
        "authority_zip": "submission_base_8008.14.zip",
        "authority_zip_sha256": LB_SHA256,
        "authority_zip_md5": LB_MD5,
        "count": len(winners),
        "task_count": len({item["task"] for item in winners}),
        "projected_gain_from_8006_61_members": sum(item["gain_from_8006_61_member"] for item in winners),
        "candidates": winners,
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(json.dumps({
        "status": "NO_PENDING_LB_PROBE",
        "authority_zip_sha256": LB_SHA256,
        "count": 0,
        "candidates": [],
        "reason": "Ten candidate SHAs are already LB-white by exact 8008.14 membership; the remaining task125 v1 file retains its local runtime rejection.",
    }, indent=2) + "\n")

    result = {
        "status": "LB_SUPERSEDED_LOCAL_REJECTIONS",
        "lb_score": 8008.14,
        "authority_zip": "submission_base_8008.14.zip",
        "authority_zip_sha256": LB_SHA256,
        "authority_zip_md5": LB_MD5,
        "file_count": len(rows),
        "task_count": len({int(row["task"]) for row in rows}),
        "local_pre_fresh_classification_counts": pre["classification_counts"],
        "exact_lb_white_count": len(winners),
        "exact_lb_white_tasks": [item["task"] for item in winners],
        "remaining_local_reject_count": len(rows) - len(winners),
        "remaining_local_rejects": [item for item in memberships if not item["exact_member_match"]],
        "fixed_winner_count": len(winners),
        "lb_probe_required_count": 0,
        "gain_from_8006_61_members": winner_manifest["projected_gain_from_8006_61_members"],
        "lb_score_gain_8006_61_to_8008_14": 1.53,
        "fresh_task157": {
            "status": "NOT_RESUMED_LB_SUPERSEDED",
            "reason": "task157 SHA a1254f... is exact LB-white in 8008.14; local fresh sampling is no longer an admission gate",
        },
        "private_high_risk_fixed_by_exact_lb": [item["task"] for item in winners if item["private_high_risk"]],
        "protected_root_artifacts_modified": [],
        "others_modified": False,
        "evidence": {
            "pre_fresh_screen": str((HERE / "audit/pre_fresh_screen.json").relative_to(ROOT)),
            "authority_profiles_2x": str((HERE / "audit/authority_profiles_2x.json").relative_to(ROOT)),
            "lb_supersession": str((HERE / "audit/lb_8008_14_supersession.json").relative_to(ROOT)),
            "winner_manifest": str((HERE / "winner_manifest.json").relative_to(ROOT)),
            "probe_manifest": str((HERE / "probe_manifest.json").relative_to(ROOT)),
        },
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    table = []
    for item in memberships:
        row = next(row for row in rows if row["sha256"] == item["candidate_sha256"])
        table.append(
            f"| {item['task']:03d} | `{item['candidate_sha256'][:12]}` | "
            f"{row['authority_profile']['cost']}→{row['candidate_profile']['cost']} | "
            f"{row['classification']} | {'LB_WHITE_FIXED' if item['exact_member_match'] else 'REJECT'} |"
        )
    report = f"""# others/71405 mid-cost lane 98 — LB supersession final

## Outcome

- New immutable authority: `submission_base_8008.14.zip`
- SHA-256: `{LB_SHA256}`
- MD5: `{LB_MD5}`
- Verified LB: **8008.14**
- Inspected: 11 files / 10 tasks
- Exact LB-white fixed: **10 files / 10 tasks**
- Pending probes: **0**

The exact candidate SHA for tasks 089/096/107/117/125-v2/138/156/157/165/209 is present in the 400-member LB-verified ZIP. Exact LB evidence supersedes the earlier local runtime, schema, and shape-cloak rejections. Those local diagnostics remain recorded because they explain why local policy alone was an unreliable LB oracle.

## Per-file disposition

| task | candidate SHA | old→candidate cost | local result | final result |
|---:|:---|:---:|:---|:---|
{chr(10).join(table)}

The only nonmatching file is task125 `d9af550b...` (v1, cost1048). The verified ZIP contains task125 v2 `c30ac7a...` (cost1045), so v1 retains its local default-ORT rejection and is not separately probed.

## Local evidence retained

- Runtime-config rejects: tasks 089, 096, 125-v1, 125-v2, 165.
- Structural rejects: task107 negative Conv padding; task117 strict AffineGrid shape inference.
- Runtime-shape mismatches: task138 (36), task156 (1), task209 (16).
- task157 alone cleared the full local strict/UB, known265×4, and truthful-shape gates. Its expensive fresh2×500 run was stopped and not resumed because exact LB-white membership is stronger evidence.
- Private-high-risk tasks 096/138/157/209 are now fixed only because their exact SHAs are LB-white; this is not a task-level exemption for future versions.

## Score accounting

The ten member-level reductions sum to `{winner_manifest['projected_gain_from_8006_61_members']:.12f}`. The whole champion moved 8006.61→8008.14 (+1.53) because the verified ZIP contains 37 white changes in total, beyond this lane's ten.

## Artifacts

- `audit/pre_fresh_screen.json`: original competition profiles and local gates.
- `audit/authority_profiles_2x.json`: repeated 8006.61 authority profiles.
- `audit/lb_8008_14_supersession.json`: exact member comparison.
- `result.json`, `winner_manifest.json`, `probe_manifest.json`: final disposition.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "status": result["status"],
        "exact_lb_white_count": result["exact_lb_white_count"],
        "exact_lb_white_tasks": result["exact_lb_white_tasks"],
        "remaining_local_reject_count": result["remaining_local_reject_count"],
        "fixed_winner_count": result["fixed_winner_count"],
        "lb_probe_required_count": result["lb_probe_required_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

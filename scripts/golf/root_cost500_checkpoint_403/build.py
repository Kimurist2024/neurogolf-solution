#!/usr/bin/env python3
"""Build a separated cost<=500 POLICY95 checkpoint from the 8012.15 authority."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
OUT = ROOT / "others/71407/cost_le500_8012_15"
AUTHORITY_LB = 8012.15
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
ROOT_SUBMISSION_SHA256 = AUTHORITY_SHA256
PRIOR_AUTHORITY = ROOT / "submission_base_8011.05.zip"
PRIOR_AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"


SPECS = [
    {
        "task": 70,
        "authority_cost": 66,
        "candidate_cost": 52,
        "source": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/candidates/task070_policy95_cost52.onnx",
        "sha256": "a4c8818ae04ee8445e42907383d5d1fd003eb0537ff54d48278534e173297b60",
        "evidence": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/task070_policy95_audit_reused.json",
        "fresh_rates": [0.99, 0.9845],
        "known_rate": 1.0,
        "classification": "POLICY95_PRIVATE_ZERO_RISK",
        "known_lb_zero": True,
    },
    {
        "task": 134,
        "authority_cost": 422,
        "candidate_cost": 320,
        "source": ROOT / "scripts/golf/agent_cost251_500_half_307/approved_policy95/task134_cost320.onnx",
        "sha256": "a610dcc58d2715ea4c39e000bfc83bb39ee69b69b95ee4a7ead252f3b126880b",
        "evidence": ROOT / "scripts/golf/agent_cost251_500_half_307/task134_cost320_rebase8012_policy95_audit.json",
        "fresh_rates": [0.9685, 0.963],
        "known_rate": 1.0,
        "classification": "POLICY95_PRIVATE_ZERO_RISK",
        "known_lb_zero": True,
    },
    {
        "task": 202,
        "authority_cost": 48,
        "candidate_cost": 20,
        "source": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/candidates/task202_policy95_cost20.onnx",
        "sha256": "06a945b16a5682a14458a54463f23634cf6963906ff7f9370e9d64084ff71073",
        "evidence": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/task202_policy95_audit_reused.json",
        "fresh_rates": [0.974, 0.9665],
        "known_rate": 1.0,
        "classification": "POLICY95_PRIVATE_ZERO_LINEAGE_NON_GIANT",
        "known_lb_zero": True,
    },
    {
        "task": 343,
        "authority_cost": 173,
        "candidate_cost": 172,
        "source": ROOT / "scripts/golf/cost101_250_half_307/policy95_candidates/task343_cost172_POLICY95_KNOWN_LB_ZERO.onnx",
        "sha256": "c1047d40b875d37a7a9e28a52a47e2c569f5156924691118082aaca4ed5198e6",
        "evidence": ROOT / "scripts/golf/cost101_250_half_307/task343_policy95_evidence.json",
        "fresh_rates": [0.9935, 0.996],
        "known_rate": 1.0,
        "classification": "POLICY95_KNOWN_LB_ZERO_NOT_GUARANTEED",
        "known_lb_zero": True,
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def task_score(cost: int) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def build_zip(path: Path, replacements: dict[str, bytes]) -> dict[str, object]:
    if path.exists():
        path.unlink()
    with zipfile.ZipFile(AUTHORITY) as source, zipfile.ZipFile(path, "w") as target:
        target.comment = source.comment
        source_infos = source.infolist()
        if len(source_infos) != 400:
            raise RuntimeError(f"authority member count {len(source_infos)} != 400")
        for info in source_infos:
            target.writestr(info, replacements.get(info.filename, source.read(info.filename)))

    with zipfile.ZipFile(AUTHORITY) as source, zipfile.ZipFile(path) as candidate:
        if candidate.namelist() != source.namelist():
            raise RuntimeError("member order changed")
        changed = []
        for name in source.namelist():
            before = digest(source.read(name))
            after = digest(candidate.read(name))
            if before != after:
                changed.append(name)
        if sorted(changed) != sorted(replacements):
            raise RuntimeError({"changed": changed, "expected": sorted(replacements)})
        for name, data in replacements.items():
            if candidate.read(name) != data:
                raise RuntimeError(f"replacement mismatch: {name}")
    data = path.read_bytes()
    return {
        "file": path.name,
        "sha256": digest(data),
        "md5": hashlib.md5(data).hexdigest(),
        "member_count": 400,
        "changed_members_only": sorted(replacements),
        "member_order_preserved": True,
    }


def main() -> int:
    authority_data = AUTHORITY.read_bytes()
    if digest(authority_data) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    root_before = digest((ROOT / "submission.zip").read_bytes())
    if root_before != ROOT_SUBMISSION_SHA256:
        raise RuntimeError("root submission is not the pinned authority")
    if digest(PRIOR_AUTHORITY.read_bytes()) != PRIOR_AUTHORITY_SHA256:
        raise RuntimeError("prior audited authority SHA mismatch")
    rebased_members = {}
    with zipfile.ZipFile(PRIOR_AUTHORITY) as prior, zipfile.ZipFile(AUTHORITY) as current:
        for spec in SPECS:
            member = f"task{spec['task']:03d}.onnx"
            same = prior.read(member) == current.read(member)
            rebased_members[member] = same
            if not same:
                raise RuntimeError(f"audited authority member changed: {member}")

    policy_dir = OUT / "POLICY95_NOT_LB_GUARANTEED"
    evidence_dir = OUT / "evidence"
    reports_dir = OUT / "search_reports"
    for directory in (OUT, policy_dir, evidence_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    replacements = {}
    manifest_candidates = []
    for source_spec in SPECS:
        spec = dict(source_spec)
        data = spec.pop("source").read_bytes()
        expected = spec["sha256"]
        if digest(data) != expected:
            raise RuntimeError(f"task{spec['task']:03d} SHA mismatch")
        member = f"task{spec['task']:03d}.onnx"
        replacements[member] = data
        output_model = policy_dir / member
        output_model.write_bytes(data)
        evidence_source = spec.pop("evidence")
        evidence_name = f"task{spec['task']:03d}_audit.json"
        shutil.copy2(evidence_source, evidence_dir / evidence_name)
        spec.update({
            "file": str(output_model.relative_to(OUT)),
            "evidence": f"evidence/{evidence_name}",
            "half_target_met": spec["candidate_cost"] * 2 <= spec["authority_cost"],
            "authority_task_score": task_score(spec["authority_cost"]),
            "candidate_task_score": task_score(spec["candidate_cost"]),
            "score_gain": task_score(spec["candidate_cost"]) - task_score(spec["authority_cost"]),
            "fresh_threshold": 0.95,
            "fresh_per_seed": 2000,
            "ort_configs": 4,
            "runtime_errors": 0,
            "nonfinite_cases": 0,
            "shape_mismatches": 0,
            "small_positive_elements": 0,
            "guaranteed_safe": False,
        })
        manifest_candidates.append(spec)

    all_zip = build_zip(
        OUT / "submission_POLICY95_CONFIRMED_CHECKPOINT_NOT_LB_GUARANTEED.zip",
        replacements,
    )
    half_replacements = {
        name: data
        for name, data in replacements.items()
        if next(row for row in manifest_candidates if row["task"] == int(name[4:7]))["half_target_met"]
    }
    half_zip = build_zip(
        OUT / "submission_POLICY95_HALF_ONLY_NOT_LB_GUARANTEED.zip",
        half_replacements,
    )

    scores_path = OUT / "all_scores_POLICY95_projection.csv"
    candidate_by_task = {row["task"]: row for row in manifest_candidates}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        raise RuntimeError("all_scores.csv has no header")
    for row in rows:
        task = int(row["task"].removeprefix("task"))
        if task in candidate_by_task:
            candidate = candidate_by_task[task]
            if int(row["cost"]) != candidate["authority_cost"]:
                raise RuntimeError(f"task{task:03d} authority CSV cost mismatch")
            row["cost"] = str(candidate["candidate_cost"])
            row["score"] = f"{candidate['candidate_task_score']:.4f}"
    with scores_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_sources = {
        "cost11_100_REPORT.md": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/REPORT.md",
        "cost11_100_MANIFEST.json": ROOT / "scripts/golf/agent_cost11_100_lowcost_patterns_401/MANIFEST.json",
        "root_half_history_evidence.json": ROOT / "scripts/golf/root_cost500_history_scan_308/evidence.json",
        "cost101_250_pattern_evidence.json": ROOT / "scripts/golf/cost101_250_half_307/pattern_evidence.json",
        "cost251_500_lowcost_transfer_evidence.json": ROOT / "scripts/golf/agent_cost251_500_half_307/lowcost_transfer_evidence.json",
        "task048_rejection_fresh.json": ROOT / "scripts/golf/agent_cost251_500_half_307/task048_fresh1000.json",
        "task168_rejection_fresh.json": ROOT / "scripts/golf/agent_cost251_500_half_307/task168_fresh2000.json",
    }
    for name, source in report_sources.items():
        shutil.copy2(source, reports_dir / name)

    total_gain = sum(row["score_gain"] for row in manifest_candidates)
    half_gain = sum(row["score_gain"] for row in manifest_candidates if row["half_target_met"])
    authority_score_at_risk = sum(row["authority_task_score"] for row in manifest_candidates)
    authority_le500_rows = [row for row in rows if int(row["cost"]) <= 500]
    score25_tasks = sorted(
        int(row["task"].removeprefix("task"))
        for row in authority_le500_rows
        if float(row["score"]) >= 25.0
    )
    manifest = {
        "checkpoint": True,
        "checkpoint_note": "Confirmed candidates only; task161/task175 and remaining POLICY95 scans are still pending.",
        "authority": {
            "lb": AUTHORITY_LB,
            "zip": AUTHORITY.name,
            "sha256": AUTHORITY_SHA256,
            "md5": hashlib.md5(authority_data).hexdigest(),
            "rebase_proof": {
                "audits_originally_pinned_to": PRIOR_AUTHORITY.name,
                "prior_sha256": PRIOR_AUTHORITY_SHA256,
                "candidate_task_members_byte_identical_in_8012_15": rebased_members,
            },
        },
        "scope": {
            "authority_tasks_at_cost_le_500": len(authority_le500_rows),
            "existing_score25_tasks": score25_tasks,
            "eligible_non_score25_tasks": len(authority_le500_rows) - len(score25_tasks),
            "cost_le_10_reference_members": 21,
            "target": "candidate cost <= half authority cost",
        },
        "acceptance_policy": {
            "known_or_fresh_minimum": 0.95,
            "known_private_zero_lineage_allowed": True,
            "runtime_errors_forbidden": True,
            "undefined_behavior_forbidden": True,
            "nonfinite_forbidden": True,
            "shape_cloak_forbidden": True,
            "small_positive_margin_forbidden": True,
        },
        "guaranteed_safe_candidates": [],
        "policy95_candidates": manifest_candidates,
        "confirmed_summary": {
            "candidate_count": len(manifest_candidates),
            "half_target_count": sum(row["half_target_met"] for row in manifest_candidates),
            "cost_reduction": sum(row["authority_cost"] - row["candidate_cost"] for row in manifest_candidates),
            "conditional_score_gain": total_gain,
            "conditional_lb": AUTHORITY_LB + total_gain,
            "half_only_conditional_gain": half_gain,
            "half_only_conditional_lb": AUTHORITY_LB + half_gain,
            "all_fail_worst_case_delta": -authority_score_at_risk,
            "all_fail_worst_case_lb": AUTHORITY_LB - authority_score_at_risk,
        },
        "search_snapshot": {
            "cost11_100_candidate_task_evaluations": 22280,
            "cost11_100_new_safe_finalists": 0,
            "cost251_500_lowcost_transfer_evaluations": 9044,
            "cost251_500_lowcost_transfer_finalists": 0,
            "all_band_historical_half_candidates_profiled": 518,
            "historical_half_tasks_found": [48, 168, 202],
            "historical_half_tasks_passing_policy95": [202],
            "pending_policy95_tasks": [161, 175],
        },
        "submission_patches": {
            "all_confirmed": all_zip,
            "half_only": half_zip,
            "root_submission_modified": False,
        },
        "root_integrity": {
            "submission_sha256_before": root_before,
            "submission_sha256_after": digest((ROOT / "submission.zip").read_bytes()),
            "unchanged": root_before == digest((ROOT / "submission.zip").read_bytes()),
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    table_rows = "\n".join(
        f"| {row['task']:03d} | {row['authority_cost']} -> {row['candidate_cost']} | "
        f"{'yes' if row['half_target_met'] else 'no'} | "
        f"{row['fresh_rates'][0]*100:.2f}% / {row['fresh_rates'][1]*100:.2f}% | "
        f"+{row['score_gain']:.6f} | {row['classification']} |"
        for row in manifest_candidates
    )
    readme = "# cost<=500 optimization checkpoint (8012.15 authority)\n\n"
    readme += "This checkpoint contains only candidates whose full audit is already complete. "
    readme += "The root champion is unchanged. task161/task175 and the remaining history scan are pending.\n\n"
    readme += "| task | cost | half | fresh (2 seeds) | conditional gain | class |\n"
    readme += "|---:|---:|:---:|---:|---:|---|\n" + table_rows + "\n\n"
    readme += f"Confirmed conditional total: **+{total_gain:.6f}** -> **{AUTHORITY_LB + total_gain:.6f}**. "
    readme += f"Only task202 reaches the half-cost target; half-only conditional gain is **+{half_gain:.6f}**.\n\n"
    readme += "All four entries are POLICY95 and are not leaderboard-guaranteed. "
    readme += "Use `submission_POLICY95_CONFIRMED_CHECKPOINT_NOT_LB_GUARANTEED.zip` for the four-task checkpoint, "
    readme += "or `submission_POLICY95_HALF_ONLY_NOT_LB_GUARANTEED.zip` for task202 only.\n"
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    if digest((ROOT / "submission.zip").read_bytes()) != root_before:
        raise RuntimeError("root submission changed during build")
    print(json.dumps({
        "output": str(OUT),
        "candidates": [row["task"] for row in manifest_candidates],
        "conditional_gain": total_gain,
        "conditional_lb": AUTHORITY_LB + total_gain,
        "all_zip": all_zip,
        "half_zip": half_zip,
        "root_unchanged": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

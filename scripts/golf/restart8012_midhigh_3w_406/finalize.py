#!/usr/bin/env python3
"""Build the fail-closed handoff for the 8012.15 cost167..500 restart lane."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def candidate(task: int, cost: int, prefix: str) -> dict[str, object]:
    matches = sorted((HERE / "history_candidates").glob(f"task{task:03d}_cost{cost}_{prefix}*.onnx"))
    if len(matches) != 1:
        raise RuntimeError((task, cost, prefix, matches))
    path = matches[0]
    return {"task": task, "candidate_cost": cost, "path": rel(path), "sha256": sha(path)}


def main() -> int:
    authority = json.loads((HERE / "authority.json").read_text())
    if authority["authority_sha256"] != AUTHORITY_SHA256:
        raise RuntimeError("authority metadata mismatch")
    if sha(ROOT / authority["authority"]) != AUTHORITY_SHA256:
        raise RuntimeError("authority bytes changed")
    transfer = json.loads((HERE / "transfer_evidence.json").read_text())
    simplify = json.loads((HERE / "simplify_evidence.json").read_text())

    dispositions = []
    rows = [
        {
            **candidate(48, 142, "a4b9ff22291f"), "authority_cost": 379,
            "classification": "REJECT_FRESH_BELOW_POLICY90",
            "known_accuracy": 1.0, "fresh_accuracies": [0.611],
            "reason": "independent fresh screen was 611/1000; prior LB-black/private-zero lineage",
            "evidence": "scripts/golf/agent_cost251_500_half_307/task048_fresh1000.json",
        },
        {
            **candidate(143, 148, "d3b05e52036a"), "authority_cost": 212,
            "classification": "REJECT_FRESH_BELOW_POLICY90",
            "known_accuracy": 1.0, "fresh_accuracies": [0.0004, 0.0006],
            "reason": "lookup carrier; independent seeds/configurations produced 2/5000 and 3/5000",
            "evidence": "scripts/golf/loop_7999_13/lane_c11/fresh_audit.json",
        },
        {
            **candidate(161, 186, "57487cce1b40"), "authority_cost": 190,
            "classification": "POLICY90_DUPLICATE_LANE404",
            "known_accuracy": 265 / 266, "fresh_accuracies": [0.9924, 0.9935],
            "score_gain": math.log(190 / 186),
            "reason": "nonexact normal-POLICY90 candidate; independently margin-repaired and already handed off by lane404",
            "evidence": "scripts/golf/agent_review_task161_margin8_280/evidence.json",
            "admit_policy90": True, "unique_to_lane406": False,
        },
        {
            **candidate(168, 166, "e27c909a920a"), "authority_cost": 414,
            "classification": "REJECT_FRESH_BELOW_POLICY90",
            "known_accuracy": 1.0, "fresh_accuracies": [0.3035],
            "reason": "independent fresh screen was 607/2000; prior LB-black/private-zero lineage",
            "evidence": "scripts/golf/agent_cost251_500_half_307/task168_fresh2000.json",
        },
        {
            **candidate(185, 185, "819bc2f00d96"), "authority_cost": 279,
            "classification": "EXCLUDED_KNOWN_BLACK_CATALOG",
            "known_accuracy": 1.0, "fresh_accuracies": [0.002, 0.002],
            "reason": "maintained private-zero/black catalog exclusion; independent cost185 fresh was 1/500 on both seeds/configurations",
            "evidence": "scripts/golf/loop_8004_42_plus20/agent_expand20f_90/audit/fresh_probe_2seed.json",
        },
        {
            **candidate(355, 249, "7ca617858a19"), "authority_cost": 250,
            "classification": "POLICY90_DUPLICATE_LANE404_PUBLIC_OVERFIT_RISK",
            "known_accuracy": 264 / 267, "fresh_accuracies": [0.9871, 0.9860],
            "score_gain": math.log(250 / 249),
            "reason": "normal-POLICY90/public-overfit-risk candidate already handed off by lane404",
            "evidence": "scripts/golf/agent_review_task355_policy90_284/evidence.json",
            "admit_policy90": True, "unique_to_lane406": False,
        },
        {
            **candidate(384, 179, "d4f13184877f"), "authority_cost": 180,
            "classification": "REJECT_RUNTIME_SHAPE_CLOAK",
            "known_accuracy": 265 / 266,
            "reason": "ych/hid declared [1,1,1,1] but execute [1,1,30,30]; real intermediate footprint destroys strict reduction",
            "evidence": "scripts/golf/agent_policy90_backlog_281/candidates.json",
        },
    ]
    for row in rows:
        row.setdefault("admit_policy90", False)
        row.setdefault("unique_to_lane406", False)
        row["half"] = 2 * int(row["candidate_cost"]) <= int(row["authority_cost"])
        dispositions.append(row)

    duplicate_admissions = [row for row in dispositions if row["admit_policy90"]]
    unique_admissions = [row for row in duplicate_admissions if row["unique_to_lane406"]]
    black4 = [70, 134, 202, 343]
    payload = {
        "authority": authority["authority"], "authority_sha256": AUTHORITY_SHA256,
        "authority_lb": 8012.15,
        "scope": "101 non-score25 tasks with current cost167..500",
        "parallelism": {
            "requested": 10, "history_threads": 10,
            "pipelines": ["loose+ZIP history", "cost<=10 transfer", "current graph simplification"],
        },
        "search_coverage": {
            "cost_le10_transfer_variants_per_task": 144,
            "cost_le10_transfer_finalists": len(transfer.get("finalists", [])),
            "current_graph_simplifier_finalists": len(simplify.get("finalists", [])),
            "history_loose_paths": 9792, "history_zip_paths": 378,
            "history_unique_task_sha": 2419,
            "history_theoretical_strict_lower": 1339,
            "history_completed_checkpoint": 1000,
            "history_residual_disposition": "bounded-wait timeout/reject; no unverified candidate admitted",
            "prior_isolated_exhaustive_crosscheck": [
                "scripts/golf/cost101_250_half_307/policy95_history_evidence.json",
                "scripts/golf/agent_cost251_500_half_307/strict_inventory.json",
                "scripts/golf/agent_cost251_500_half_307/cost_first_rank.json",
                "scripts/golf/agent_cost251_500_half_307/known_first_rank.json",
            ],
        },
        "policy": {
            "threshold": 0.90,
            "each_seed_each_config": True,
            "runtime_errors_nonfinite_shape_smallpositive_allowed": 0,
            "known_lb_black_unconditional_exclusion": black4,
            "guaranteed_safe_separate_from_policy90": True,
        },
        "dispositions": dispositions,
        "unique_new_admissions": unique_admissions,
        "duplicate_policy90_admissions": duplicate_admissions,
        "unique_new_gain": sum(float(row.get("score_gain", 0.0)) for row in unique_admissions),
        "duplicate_gain_already_in_lane404": sum(float(row.get("score_gain", 0.0)) for row in duplicate_admissions),
        "guaranteed_safe": [],
        "protected_writes": "root/others untouched; only scripts/golf/restart8012_midhigh_3w_406",
    }
    (HERE / "history_evidence.json").write_text(json.dumps(payload, indent=2) + "\n")

    manifest = {
        "authority": payload["authority"], "authority_sha256": AUTHORITY_SHA256,
        "authority_lb": 8012.15, "scope_count": authority["scope_count"],
        "unique_new_admission_count": len(unique_admissions),
        "duplicate_policy90_admission_count": len(duplicate_admissions),
        "unique_new_score_gain": payload["unique_new_gain"],
        "duplicate_score_gain_already_in_lane404": payload["duplicate_gain_already_in_lane404"],
        "duplicate_candidates": duplicate_admissions,
        "rejections": [row for row in dispositions if not row["admit_policy90"]],
        "evidence": [
            rel(HERE / "authority.json"), rel(HERE / "transfer_evidence.json"),
            rel(HERE / "simplify_evidence.json"), rel(HERE / "history_evidence.json"),
        ],
        "root_or_others_modified": False,
    }
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = f"""# 8012.15 cost167..500 restart — 10-parallel handoff

Authority: `submission_base_8012.15.zip` (`{AUTHORITY_SHA256}`), LB **8012.15**.

## Outcome

No unique new admissible candidate remains in this lane. Two valid normal-POLICY90
candidates were rediscovered, but both are byte-identical duplicates of lane404:

| task | authority -> candidate | known | fresh (two seeds) | gain | disposition |
|---:|---:|---:|---:|---:|---|
| 161 | 190 -> 186 | 265/266 | 99.24%, 99.35% | +{math.log(190/186):.6f} | POLICY90 duplicate lane404 |
| 355 | 250 -> 249 | 264/267 | 98.71%, 98.60% | +{math.log(250/249):.6f} | POLICY90 duplicate lane404 / public-overfit risk |

Duplicate conditional gain already represented by lane404: **+{payload['duplicate_gain_already_in_lane404']:.6f}**.
This lane contributes **+0.000000 unique gain** and must not cause the two models
to be merged twice.

## Decisive rejections

- task048 379->142: fresh 61.10%; reject.
- task143 212->148: lookup carrier, fresh 2/5000 and 3/5000; reject.
- task168 414->166: fresh 30.35%; reject.
- task185 279->185: catalog black/private-zero and fresh 1/500 per seed; reject.
- task384 180->179: runtime shape cloak; reject regardless of 99.62% known.
- task070/task134/task202/task343: latest explicit LB-black list; unconditionally excluded.

## Coverage

- 101 current tasks with cost167..500.
- 144 finite low-cost/generic variants per task; no finalist.
- Exact current-graph initializer/Einsum/Gather/lookup/ConvTranspose-oriented shaves; no finalist.
- Loose+ZIP history: 9,792 ONNX paths, 378 ZIPs, 2,419 unique task/hash pairs,
  1,339 theoretical strict-lower profiles. Ten workers reached the 1,000-result
  checkpoint; pathological residual calls exceeded the bounded wait and are
  fail-closed, never admitted. Prior isolated exhaustive 101..500 inventories
  were cross-checked for the residual candidate families.

Every admission requires >=90% independently in each seed/config and zero
runtime errors, nonfinite values, output-shape mismatches, small-positive values,
UB, or runtime-shape cloak. `GUARANTEED_SAFE` remains empty and separate from POLICY90.

No root artifact or `others/` file was modified.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "unique_new": len(unique_admissions), "duplicates": [row["task"] for row in duplicate_admissions],
        "rejections": [row["task"] for row in dispositions if not row["admit_policy90"]],
        "duplicate_gain": payload["duplicate_gain_already_in_lane404"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Assemble the durable A37 adoption manifest after every gate completes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8000.46.zip"
AUTHORITY_SHA256 = "74cb9c4a96dcf9d1d1ca4ad1e0afdb67f5218c0a06ad8e896ad1654b6548f534"


def load(name: str):
    return json.loads((HERE / name).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("authority ZIP changed during A37")
    build13 = load("task013_qch_from_qor_shared_reduction_build.json")
    audit13 = load("task013_candidate_audit.json")
    fresh13 = load("task013_dual_fresh5000.json")[0]
    external13 = load("task013_external500.json")
    build105 = load("task105_remove_output_one_build.json")
    audit105 = load("task105_candidate_audit.json")
    fresh105 = load("task105_dual_fresh5000.json")[0]
    explicit105 = {
        mode: load(f"task105_fresh5000_{mode}_explicit.json")
        for mode in ("disable_all", "default")
    }
    external105 = load("task105_external500.json")

    if not audit13["eligible"]:
        raise RuntimeError("task013 structural/known audit failed")
    if build13["candidate_sha256"] != audit13["candidate_sha256"]:
        raise RuntimeError("task013 candidate hash mismatch")
    if not fresh13["perfect"]:
        raise RuntimeError("task013 fresh5000 failed")
    if external13["decision"]["verdict"] != "ACCEPT_STRICT":
        raise RuntimeError("task013 external validator failed")
    if external13["candidate"]["sha256"] != build13["candidate_sha256"]:
        raise RuntimeError("task013 external hash mismatch")

    if not audit105["eligible_for_user_95_percent_gate"]:
        raise RuntimeError("task105 structural/known audit failed")
    if build105["candidate_sha256"] != audit105["candidate_sha256"]:
        raise RuntimeError("task105 candidate hash mismatch")
    if external105["decision"]["verdict"] != "ACCEPT_STRICT":
        raise RuntimeError("task105 external validator failed")
    if external105["candidate"]["sha256"] != build105["candidate_sha256"]:
        raise RuntimeError("task105 external hash mismatch")
    for mode, row in explicit105.items():
        if row["requested"] != 5000 or row["runtime_errors"] != 0:
            raise RuntimeError(f"task105 {mode} fresh runtime gate failed")
        if not row["passes_user_95_percent_gate"]:
            raise RuntimeError(f"task105 {mode} fresh accuracy below user gate")

    incremental_gain = math.log(739 / 731) + math.log(195 / 194)
    authority_gain = math.log(743 / 731) + math.log(199 / 194)
    manifest = {
        "lane": "a37",
        "authority_zip": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "authority_zip_modified": False,
        "decisions": [
            {
                "task": 13,
                "decision": "ACCEPT_STRICT",
                "candidate": str(Path(build13["candidate"]).relative_to(ROOT)),
                "sha256": build13["candidate_sha256"],
                "safe_baseline_cost": 739,
                "authority_cost": external13["baseline"]["cost"],
                "candidate_cost": build13["candidate_cost"],
                "safe_incremental_gain": math.log(739 / 731),
                "authority_gain": math.log(743 / 731),
                "known_dual": audit13["known_dual_and_raw_differential"],
                "fresh_dual": fresh13,
                "runtime_shapes": audit13["structure"]["runtime_shapes"],
                "external500": external13["decision"],
                "evidence": [
                    "task013_qch_from_qor_shared_reduction_build.json",
                    "task013_candidate_audit.json",
                    "task013_dual_fresh5000.json",
                    "task013_external500.json",
                ],
            },
            {
                "task": 105,
                "decision": "ACCEPT_USER_95_PERCENT_GATE",
                "candidate": str(Path(build105["candidate"]).relative_to(ROOT)),
                "sha256": build105["candidate_sha256"],
                "safe_baseline_cost": 195,
                "authority_cost": external105["baseline"]["cost"],
                "candidate_cost": build105["candidate_cost"],
                "safe_incremental_gain": math.log(195 / 194),
                "authority_gain": math.log(199 / 194),
                "known_dual": audit105["known_dual_and_raw_differential"],
                "fresh_dual_summary": fresh105,
                "fresh_dual_explicit_errors": explicit105,
                "runtime_shapes": audit105["structure"]["runtime_shapes"],
                "external500": external105["decision"],
                "evidence": [
                    "task105_remove_output_one_build.json",
                    "task105_candidate_audit.json",
                    "task105_dual_fresh5000.json",
                    "task105_fresh5000_disable_all_explicit.json",
                    "task105_fresh5000_default_explicit.json",
                    "task105_external500.json",
                    "task105_affine_audit.json",
                ],
            },
        ],
        "safe_incremental_gain": incremental_gain,
        "authority_relative_gain": authority_gain,
        "projected_score_from_8000_46": 8000.46 + authority_gain,
        "rejected_deeper_paths": [
            {
                "task": 105,
                "idea": "delete root_2 by linear absorption into v_pair",
                "reason": "reachable v_pair states do not admit an exact linear affine-add replacement",
                "evidence": "task105_affine_audit.json",
            },
            {
                "task": 13,
                "idea": "unnamed reduction ellipsis inside Qor replacement",
                "reason": "ORT requires an input ellipsis to appear in the output; replaced by a standard shared explicit label",
                "evidence": "task013_qch_from_qor_ellipsis.onnx (rejected probe only)",
            },
        ],
    }
    (HERE / "A37_RESULT.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = f"""# A37 exact-shrink result

- task013: **ACCEPT_STRICT**, cost 739→731 (authority 743→731), SHA `{build13['candidate_sha256']}`.
- task105: **ACCEPT_USER_95_PERCENT_GATE**, cost 195→194 (authority 199→194), SHA `{build105['candidate_sha256']}`.
- Increment over the two already-safe lane candidates: **+{incremental_gain:.12f}**.
- Increment against `submission_base_8000.46.zip`: **+{authority_gain:.12f}**, projected total **{8000.46 + authority_gain:.6f}**.
- The authority ZIP was hash-checked and not modified.

Both candidates pass full checker, strict static inference, complete dual-ORT known sets, truthful all-node runtime shapes, standard-domain and safety checks, and external differential 500/500 with `ACCEPT_STRICT`. task013 is fresh 5000/5000 on both ORT modes. task105 is fresh {explicit105['disable_all']['right']}/5000 and {explicit105['default']['right']}/5000, with zero runtime errors in both modes; this is above the user-authorized 95% gate and is raw-identical to the authority baseline on the external 500-case differential.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({
        "decisions": [(row["task"], row["decision"], row["candidate_cost"]) for row in manifest["decisions"]],
        "safe_incremental_gain": incremental_gain,
        "authority_relative_gain": authority_gain,
        "projected_score_from_8000_46": 8000.46 + authority_gain,
    }, indent=2))


if __name__ == "__main__":
    main()

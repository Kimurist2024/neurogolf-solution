#!/usr/bin/env python3
"""Fail-closed finalizer for the task101 exact-regolf winner."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8009.46.zip"
BASE_ZIP_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
BASE_MEMBER_SHA = "ad535c519c18681700f956262c48ca5990f15ff4b58fd94de95ef3beff69a84b"
CANDIDATE_SHA = "a57a944d958be1945563a7d55320239bb0f36b4ba25af1041f589a904cc7b81e"
CANDIDATE = HERE / "candidates/task101_exact_broadcast_expand_a57a944d958b.onnx"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from golf.check_conv_bias import check_model as check_conv_bias  # noqa: PLC0415

    require(sha256(BASE_ZIP) == BASE_ZIP_SHA, "authority zip changed")
    require(sha256(HERE / "baseline/task101.onnx") == BASE_MEMBER_SHA, "authority member changed")
    require(sha256(CANDIDATE) == CANDIDATE_SHA, "candidate changed")

    mechanical = json.loads((HERE / "mechanical_audit.json").read_text())
    fresh = json.loads((HERE / "fresh_dual_raw_2x5000.json").read_text())
    rows = mechanical["tasks"]["101"]["rewrites"]
    row = next(item for item in rows if item["sha256"] == CANDIDATE_SHA)
    audit = row["audit"]
    require(row["actual_cost"] == 5641, "candidate cost is not 5641")
    require(row["strictly_lower"], "candidate is not strictly lower")
    require(audit["full_check"] and audit["strict_shape_data_prop"], "checker/strict failed")
    require(not audit["runtime_shape_trace"]["declared_actual_mismatches"], "runtime shape mismatch")
    require(audit["official_like_score"] == {
        "memory": 5541,
        "params": 100,
        "cost": 5641,
        "score": audit["official_like_score"]["score"],
        "correct": True,
    }, "official-like score mismatch")
    for mode in ("known_disable_all", "known_default"):
        require(audit[mode]["total"] == {"right": 266, "wrong": 0, "errors": 0}, f"{mode} failed")
    require(check_conv_bias(onnx.load(CANDIDATE)) == [], "official Conv-bias checker failed")
    require(audit["nonstandard_domains"] == [], "nonstandard domain")
    require(audit["banned_ops"] == [], "banned op")
    require(audit["function_count"] == audit["sparse_initializer_count"] == 0, "function/sparse initializer")
    require(not any(audit["lookup_red_flags"].values()), "lookup red flag")

    seed_rows: dict[str, object] = {}
    require(set(fresh["seeds"]) == {"1011181", "2011181"}, "fresh ranges are not the disjoint approved ranges")
    for seed, evidence in fresh["seeds"].items():
        require(evidence["valid"] == 5000 and evidence["skipped_over_30"] == 0, f"fresh count failed {seed}")
        for mode in ("candidate_disabled", "candidate_default"):
            counter = evidence["counters"][mode]
            require(counter["runtime_errors"] == counter["nonfinite"] == 0, f"runtime/nonfinite {seed} {mode}")
            require(counter["right"] / 5000 >= 0.90, f"fresh accuracy below policy {seed} {mode}")
        for key in ("disabled_raw_equal", "default_raw_equal", "candidate_dual_raw_equal"):
            require(evidence["comparisons"][key] == 5000, f"raw equality failed {seed} {key}")
            require(evidence["max_raw_delta"][key] == 0.0, f"raw delta failed {seed} {key}")
        seed_rows[seed] = {
            "candidate_disabled": evidence["counters"]["candidate_disabled"],
            "candidate_default": evidence["counters"]["candidate_default"],
            "raw_equal_to_authority_each_mode": 5000,
            "dual_ort_raw_equal": 5000,
            "max_raw_delta": 0.0,
        }

    gain = math.log(5655 / 5641)
    winner_dir = HERE / "winner"
    winner_dir.mkdir(exist_ok=True)
    winner = winner_dir / "task101.onnx"
    winner.write_bytes(CANDIDATE.read_bytes())
    require(sha256(winner) == CANDIDATE_SHA, "winner copy mismatch")

    manifest = {
        "authority_zip": "submission_base_8009.46.zip",
        "authority_zip_sha256": BASE_ZIP_SHA,
        "winner_count": 1,
        "projected_gain": gain,
        "winners": [{
            "task": 101,
            "path": str(winner.relative_to(ROOT)),
            "sha256": CANDIDATE_SHA,
            "authority_member_sha256": BASE_MEMBER_SHA,
            "base_cost": 5655,
            "candidate_cost": 5641,
            "memory": 5541,
            "params": 100,
            "gain": gain,
            "rewrite": "And(all-true[1,1,3,6], scalar_bool) -> Expand(scalar_bool, [1,1,3,6])",
            "formal_equivalence": "Boolean identity plus shape-preserving broadcast; candidate equals the immutable 8009.46 payload for every valid input under ONNX semantics.",
            "known_dual_ort": {"right": 266, "wrong": 0, "errors": 0},
            "fresh_disjoint_2x5000": seed_rows,
            "gate_basis": "strictly-lower exact LB-white raw-equivalence; user >=90% fresh threshold; no-regression to accepted 8009.46 payload",
            "checker": True,
            "strict_data_prop": True,
            "runtime_shape_mismatches": 0,
            "official_conv_bias_findings": [],
            "nonstandard_domains": [],
            "banned_ops": [],
            "functions": 0,
            "sparse_initializers": 0,
            "lookup_red_flags": audit["lookup_red_flags"],
        }],
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rejected = {
        "task101": [
            {
                "sha256": "3e60ea7e630360a106c20630cf168dbd6d0c139554cfa12bb8bc6507e36afa6a",
                "rewrite": "generic scalar substitution",
                "reason": "not shape preserving; Concat infers 15 instead of 17 and ORT/checker reject",
            },
            {
                "sha256": "788c809c98869afded6dc6dc553de139d0149da53cf32acc7a1156431bc47255",
                "rewrite": "bool Resize to [3,6]",
                "reason": "ORT 1.24 has no Resize(bool) implementation",
            },
        ],
        "task133": [
            {
                "authority_member_sha256": "6c5dc3a593b0900e16966b9d4c40af509a34c1dd1f0264c31cd30eaf9b4570e5",
                "base_cost": 4393,
                "reason": "no semantic cleanup/dedupe/no-op/CSE/fold/absorb opportunity; current graph has 30 runtime/declaration shape mismatches",
            },
            {
                "sha256": "f7dd6c37e74f1d6e6cc88b2f3311fb1ce667a7fc62717f7eac4c75d53bedf24a",
                "rewrite": "clear 238 stale value_info annotations",
                "reason": "not statically fully shaped and not scorer-profileable; not a strictly-lower admissible candidate",
            },
            {
                "sha256": "867f1f1c7d4e8ea09f5d01de4fcce619640d364827c73c3c515ba0bcd4ad1731",
                "rewrite": "independent truthful generator-rule control",
                "cost": 5570,
                "reason": "truthful but 1177 cost above incumbent 4393",
            },
        ],
    }
    (HERE / "rejected_manifest.json").write_text(json.dumps(rejected, indent=2) + "\n")
    result = {
        "status": "ONE_EXACT_REGOLF_WINNER",
        "targets": [101, 133],
        "winner_tasks": [101],
        "rejected_tasks": [133],
        "projected_gain": gain,
        "root_or_submission_modified": False,
        "evidence": [
            "mechanical_audit.json",
            "fresh_dual_raw_2x5000.json",
            "winner_manifest.json",
            "rejected_manifest.json",
        ],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

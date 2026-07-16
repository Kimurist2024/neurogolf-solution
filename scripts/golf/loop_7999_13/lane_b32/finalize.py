#!/usr/bin/env python3
"""Fail-closed integrity check for the lane B32 task219 candidate."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LANE = Path(__file__).resolve().parent
AUTHORITY = ROOT / "submission_base_8002.63.zip"
CANDIDATE = LANE / "task219_b32_winner.onnx"

EXPECTED_AUTHORITY_SHA256 = (
    "a2da30657f3798e861f369ac896f36722ff658ed3e468c4d55db9a04eefbccfc"
)
EXPECTED_BASELINE_SHA256 = (
    "7a2ead58107803948d316fb8e00c4fd3ff601769309f9ad99661976f1a51bd67"
)
EXPECTED_CANDIDATE_SHA256 = (
    "e6b9793c6c54db7c795c355d82853b6d2100c2992f5c887e7ee34a2ae07a172c"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(name: str) -> dict:
    return json.loads((LANE / name).read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    require(sha256_file(AUTHORITY) == EXPECTED_AUTHORITY_SHA256, "authority ZIP changed")
    with zipfile.ZipFile(AUTHORITY) as archive:
        baseline = archive.read("task219.onnx")
    require(sha256_bytes(baseline) == EXPECTED_BASELINE_SHA256, "authority member changed")
    require(sha256_file(CANDIDATE) == EXPECTED_CANDIDATE_SHA256, "candidate changed")

    audit = read_json("winner_audit.json")
    external = read_json("external500_summary.json")
    disabled = read_json("fresh5000_disable_all_explicit.json")
    default = read_json("fresh5000_default_explicit.json")

    require(audit["pass"] is True, "winner audit did not pass")
    require(audit["baseline"]["cost"] == 1479, "unexpected baseline cost")
    require(audit["candidate"]["cost"] == 1445, "unexpected candidate cost")
    require(audit["structure"]["full_checker"] is True, "full checker failed")
    require(audit["structure"]["runtime_shapes"]["truthful"] is True, "shape cloak")
    require(audit["structure"]["banned_ops"] == [], "banned operation present")
    require(audit["structure"]["conv_bias_issues"] == [], "Conv bias issue present")
    for mode in ("disable_all", "default"):
        known = audit["known_dual_raw_equivalence"][mode]
        require(known["perfect"] is True, f"known corpus failed in {mode}")
        require(known["runtime_errors"] == 0, f"known runtime error in {mode}")
        require(known["raw_equal"] == known["total"] == 265, f"known raw mismatch in {mode}")

    decision = external["decision"]
    differential = external["differential"]
    require(external["returncode"] == 0, "external validator returned nonzero")
    require(decision["verdict"] == "ACCEPT_STRICT", "external validator rejected candidate")
    require(differential["requested"] == differential["raw_equal"] == 500, "external raw mismatch")
    require(differential["mismatches"] == 0, "external mismatch present")
    require(differential["max_abs_difference"] == 0, "external raw output changed")

    for result in (disabled, default):
        require(result["generated"] == 5000, "incomplete fresh generation")
        require(result["generation_errors"] == 0, "fresh generation error")
        require(result["runtime_errors"] == 0, "fresh runtime error")
        require(result["right"] == 4327 and result["wrong"] == 673, "fresh result drift")

    gain = math.log(1479 / 1445)
    require(abs(gain - audit["projected_gain"]) < 1e-15, "projected gain mismatch")
    print(
        json.dumps(
            {
                "status": "PASS_STRICT_CURRENT_RAW_EQUIVALENT",
                "task": 219,
                "baseline_cost": 1479,
                "candidate_cost": 1445,
                "cost_reduction": 34,
                "projected_gain": gain,
                "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
                "shared_files_written": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

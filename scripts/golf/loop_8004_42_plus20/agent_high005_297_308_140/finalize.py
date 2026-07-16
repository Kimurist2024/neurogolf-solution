#!/usr/bin/env python3
"""Fail-closed integrity check for the task005/297/308 lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPECTED_AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
EXPECTED_BASE = {
    "5": ("77eb35fdcf2dbbacaa1c63d2dfef4f3b50ecbfbc8178da3bc2e7883ee8275c57", 2325),
    "297": ("cdba3d03bf43853742508f284bf98ca5341fdb2ab50042ec895afb0069296537", 371),
    "308": ("fc845e9edee06830a880be6f385f2601d1d0ff7f017cb54b64b36cb84da7785d", 433),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    authority = ROOT / "submission_base_8009.46.zip"
    assert digest(authority) == EXPECTED_AUTHORITY_SHA

    baseline = read_json("baseline_audit.json")
    assert baseline["authority_zip_sha256"] == EXPECTED_AUTHORITY_SHA
    for task, (expected_sha, expected_cost) in EXPECTED_BASE.items():
        row = baseline["tasks"][task]
        assert row["sha256"] == expected_sha
        assert row["official_like_score"]["cost"] == expected_cost
        assert row["full_check"] is True
        assert row["strict_shape_data_prop"] is True

    candidates = read_json("candidate_audit.json")["candidates"]
    qscale = candidates["task297_shared_c_scale"]
    assert digest(ROOT / qscale["path"]) == qscale["sha256"]
    assert qscale["actual_cost"] == 370
    assert qscale["proof"]["hypothesis_equivalent_over_reachable_codes"] is False
    for mode in ("known_disable_all", "known_default"):
        assert qscale["audit"][mode]["total"] == {"right": 1, "wrong": 264, "errors": 0}

    bypass = candidates["task308_bypass_constant_shape_copy"]
    assert digest(ROOT / bypass["path"]) == bypass["sha256"]
    assert bypass["actual_cost"] is None
    assert bypass["audit"]["full_check"] is False
    assert "session_error" in bypass["audit"]["known_disable_all"]
    assert "session_error" in bypass["audit"]["known_default"]

    winners = read_json("winner_manifest.json")
    result = read_json("result.json")
    rejected = read_json("rejected_manifest.json")
    assert winners["winner_count"] == 0 and winners["winners"] == []
    assert result["winner_tasks"] == [] and result["projected_gain"] == 0.0
    assert set(rejected) == {"task005", "task297", "task308"}
    print("PASS: no candidate crossed every fail-closed admission gate")


if __name__ == "__main__":
    main()

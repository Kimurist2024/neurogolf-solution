#!/usr/bin/env python3
"""Fail-closed integrity verification for high153."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY_SHA = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
MEMBERS = {
    "44": ("12b6414193b8716e15c4129f02bf7a8cddf31159f609427824345239de080492", 1076, 2),
    "117": ("042e3ee0976af0c684fb98064800ab0b84e8bf53273a0c4121315ab7a0bfaac2", 605, 10),
    "330": ("af2a81db8b4b16f913ec05c689cb04e2894e288b6f124c2424c7aa438b9bfd0e", 896, 38),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def main() -> None:
    assert digest(ROOT / "submission_base_8009.46.zip") == AUTHORITY_SHA
    baseline = read("baseline_audit.json")
    assert baseline["authority_zip_sha256"] == AUTHORITY_SHA
    for task, (sha, cost, mismatches) in MEMBERS.items():
        row = baseline["tasks"][task]
        assert digest(ROOT / row["path"]) == sha == row["sha256"]
        assert row["official_like_score"]["cost"] == cost
        assert len(row["runtime_shape_trace"]["declared_actual_mismatches"]) == mismatches
        assert all(item["safe"] for item in row["conv_bias_findings"])

    mechanical = read("mechanical_audit.json")
    assert all(mechanical["tasks"][task]["strict_lower_count"] == 0 for task in MEMBERS)
    controls = read("control_audit.json")
    assert controls["task117_truthful_copy_hist"]["official_like_score"]["cost"] == 6762
    assert controls["task330_truthful_component_rect"]["official_like_score"]["cost"] == 5525
    for label in ("task117_truthful_copy_hist", "task330_truthful_component_rect"):
        assert not controls[label]["runtime_shape_trace"]["declared_actual_mismatches"]

    manifest = read("manifest.json")
    winners = read("winner_manifest.json")
    assert manifest["strict_lower_finalist_count"] == 0
    assert all(not row["strict_lower_candidates"] for row in manifest["tasks"].values())
    assert winners["winner_count"] == 0 and winners["winners"] == []
    print("PASS: no task044/117/330 graph crossed every strict admission gate")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate the pinned-authority 71407 candidate stage and projected score."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from golf.rank_dir import cost_of  # noqa: E402

STAGE = REPO / "others" / "71407"
ROOT_GUARDS = {
    "submission_base_8009.46.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
}
PINNED_ROOT_EXPECTED = {
    "submission.zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores.csv": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((STAGE / "MANIFEST.json").read_text())
    rebase = json.loads((STAGE / "REBASE_8009_46.json").read_text())
    active = manifest["active_candidates"]
    errors: list[str] = []
    warnings: list[str] = []
    rows = []
    with zipfile.ZipFile(REPO / "submission_base_8009.46.zip") as authority_zip:
        members = {
            int(Path(name).stem.removeprefix("task")): name
            for name in authority_zip.namelist() if name.endswith(".onnx")
        }
        with tempfile.TemporaryDirectory(prefix="validate71407_") as wd:
            for row in active:
                path = STAGE / row["file"]
                if not path.is_file():
                    errors.append(f"missing {path}")
                    continue
                actual_sha = sha256(path)
                memory, params, cost = cost_of(str(path))
                authority_path = Path(wd) / f"task{row['task']:03d}.onnx"
                authority_path.write_bytes(authority_zip.read(members[row["task"]]))
                auth_memory, auth_params, auth_cost = cost_of(str(authority_path))
                gain = math.log(row["authority_cost"] / row["candidate_cost"])
                if actual_sha != row["sha256"]:
                    errors.append(f"SHA mismatch {row['file']}")
                declared_delta = row["authority_cost"] - row["candidate_cost"]
                diagnostic_delta = auth_cost - cost
                # Truthful tasks profile directly at the official candidate
                # cost.  Inherited input-dependent-shape lineages can instead
                # undercount both sides on the zero canvas; their exact delta
                # must still match.  Accept either independently checkable
                # condition, never a bare manifest assertion.
                direct_candidate_matches = cost == row["candidate_cost"]
                diagnostic_delta_matches = diagnostic_delta == declared_delta
                if not (direct_candidate_matches or diagnostic_delta_matches):
                    errors.append(
                        f"cost evidence mismatch {row['file']}: candidate {cost} "
                        f"!= official {row['candidate_cost']}; diagnostic delta "
                        f"{diagnostic_delta} != official {declared_delta}"
                    )
                if abs(gain - row["projected_gain"]) > 1e-12:
                    errors.append(f"gain mismatch {row['file']}")
                rows.append({
                    "task": row["task"],
                    "sha256": actual_sha,
                    "official_authority_cost": row["authority_cost"],
                    "official_candidate_cost": row["candidate_cost"],
                    "diagnostic_authority_profile": {
                        "memory": auth_memory, "params": auth_params, "cost": auth_cost,
                    },
                    "diagnostic_candidate_profile": {
                        "memory": memory, "params": params, "cost": cost,
                    },
                    "cost_delta": diagnostic_delta,
                    "cost_gate": (
                        "direct_candidate_profile"
                        if direct_candidate_matches else "authority_relative_exact_delta"
                    ),
                    "gain": gain,
                })
    direct = sorted(STAGE.glob("task*.onnx"))
    if len(direct) != manifest["active_root_onnx_count"] or len(direct) != len(active):
        errors.append(f"active count mismatch: direct={len(direct)} manifest={len(active)}")
    expected_names = {row["file"] for row in active}
    if {path.name for path in direct} != expected_names:
        errors.append("direct ONNX names differ from manifest")
    total = math.fsum(row["gain"] for row in rows)
    score = manifest["baseline_score"] + total
    for source, label in ((manifest, "manifest"), (rebase, "rebase")):
        if abs(total - source["combined_projected_gain"]) > 1e-12:
            errors.append(f"{label} total mismatch")
        if abs(score - source["combined_projected_score"]) > 1e-12:
            errors.append(f"{label} score mismatch")
    guards = {}
    for name, expected in ROOT_GUARDS.items():
        actual = sha256(REPO / name)
        guards[name] = actual
        if actual != expected:
            errors.append(f"root guard changed: {name}")
    observed_root = {
        "submission.zip": manifest.get("observed_root_submission_sha256"),
        "all_scores.csv": manifest.get("observed_root_all_scores_sha256"),
    }
    for name, expected in observed_root.items():
        actual = sha256(REPO / name)
        guards[name] = actual
        if expected is None:
            errors.append(f"missing recorded concurrent-root hash: {name}")
        elif actual != expected:
            errors.append(f"concurrent root changed again: {name}")
        elif actual != PINNED_ROOT_EXPECTED[name]:
            warnings.append(
                f"{name} differs from pinned authority; preserved as recorded external state"
            )
    result = {
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "active_count": len(rows),
        "combined_projected_gain": total,
        "combined_projected_score": score,
        "remaining_to_8059_46": 8059.46 - score,
        "root_guards": guards,
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

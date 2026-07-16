#!/usr/bin/env python3
"""Write the immutable B16 handoff; no root submission files are touched."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def dump(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    exact = json.loads((HERE / "exact_audit.json").read_text())
    candidates = json.loads((HERE / "candidate_audit.json").read_text())
    collisions = json.loads((HERE / "generator_collision_audit.json").read_text())
    no_lookup = next(
        row
        for row in candidates["rows"]
        if row["path"].endswith("candidate_task157_no_lookup.onnx")
    )
    failures = {
        "baseline": "submission_base_7999.13.zip",
        "tasks": {
            "157": {
                "baseline_sha256": exact["157"]["sha256"],
                "baseline_cost": exact["157"]["actual_score"]["cost"],
                "baseline_issue": (
                    "The graph contains explicit fixk_train/fixk_a117/fixk_a187 "
                    "visible-fixture corrections, prohibited by the no-lookup gate."
                ),
                "best_new_no_lookup": {
                    "path": no_lookup["path"],
                    "sha256": no_lookup["sha256"],
                    "actual_cost": no_lookup["actual_score"]["cost"],
                    "raw_cost_cut": exact["157"]["actual_score"]["cost"]
                    - no_lookup["actual_score"]["cost"],
                    "known_dual_ort": no_lookup["known_dual"],
                    "cached_fresh1000": {
                        "right": 981,
                        "wrong": 19,
                        "ambiguous_wrong": 3,
                        "unique_wrong": 16,
                    },
                    "rejection": "known_wrong_3_and_fresh_unique_wrong_16",
                },
                "generator_collision": collisions["tasks"]["157"],
                "winner": None,
            },
            "319": {
                "baseline_sha256": exact["319"]["sha256"],
                "baseline_cost": exact["319"]["actual_score"]["cost"],
                "baseline_issue": {
                    "shape_cloak": exact["319"]["runtime_shapes"]["shape_cloak"],
                    "runtime_shape_mismatch_count": len(
                        exact["319"]["runtime_shapes"]["mismatches"]
                    ),
                    "lookup": "fixed corr_pattern correction",
                },
                "archive_summary": [
                    {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "actual_cost": (
                            None
                            if row["actual_score"] is None
                            else row["actual_score"]["cost"]
                        ),
                        "shape_cloak": row["runtime_shapes"].get("shape_cloak"),
                        "known_dual_ort": row["known_dual"],
                        "rejections": row["semantic_rejections"],
                    }
                    for row in candidates["rows"]
                    if row["task"] == 319
                ],
                "generator_collision": collisions["tasks"]["319"],
                "winner": None,
            },
        },
        "fresh5000_dual_ort": {
            "status": "not_run",
            "reason": "zero candidates survived actual-cost, known-dual, shape, and provenance gates",
        },
        "external_validator": {
            "status": "not_run",
            "reason": "external validation is required only for a strict cheaper survivor; survivor count was zero",
        },
    }
    winner = {
        "baseline": "7999.13",
        "winner_count": 0,
        "valid_cost_gain": 0,
        "valid_score_gain": 0.0,
        "winners": [],
        "root_files_modified": [],
    }
    dump("failure_manifest.json", failures)
    dump("winner_manifest.json", winner)
    report = f"""# B16 — task157 / task319

## Outcome

No candidate survived the strict soundness gate. Winner count **0**, valid cost
gain **0**, valid score gain **0.0**. No root ZIP, CSV, ledger, artifact, or
handcrafted model was modified.

## Exact 7999.13 baselines

- task157: actual cost **853**, SHA `{exact['157']['sha256']}`. Full checker,
  strict shape inference, static-positive shapes, standard domains, Conv-bias,
  banned-op, and `<15`-input Einsum checks pass. Both ORT modes are 265/265.
  It is nevertheless prohibited because it contains the explicit visible
  fixture keys `fixk_train`, `fixk_a117`, and `fixk_a187`.
- task319: actual cost **1023**, SHA `{exact['319']['sha256']}`. Both ORT modes
  are 267/267, but runtime tracing finds **{len(exact['319']['runtime_shapes']['mismatches'])}**
  declared/runtime shape mismatches, so it is a shape cloak; it also includes a
  fixed correction pattern.

## New sound rebuild attempt

`candidate_task157_no_lookup.onnx` was rebuilt from the exact task157 model by
removing all three fixture-key correction branches and returning to the generic
`bstarts` rule. SHA `{no_lookup['sha256']}`; actual cost **833** (raw cut **20**),
truthful static shapes, no lookup keys. It is rejected: both ORT modes are
262/265, and an independent cached 1000-case audit is 981/1000 with 16 errors
on uniquely solvable inputs. This is a real accuracy regression, not only the
generator's irreducible ambiguity.

## Archive audit

The task157 cost-520 header probe is only 4/265. The prior no-key cost-851 probe
is 262/265. For task319, archive r01 ties actual cost 1023 and is cloak/lookup;
r02 cannot instantiate in either ORT mode; r03/r04/r05 cost 1086/1131/1132 and
are all shape cloaks. Therefore no archive graph is strictly cheaper, truthful,
and known-correct.

## Generator truth

Constructive witnesses in `generator_collision_audit.json` reproduce valid
calls with identical input and distinct outputs for both hashes 6a1e5592 and
ce602527. A deterministic ONNX cannot achieve zero error over every valid call.
This does not excuse the 16 unique-case misses of the compact task157 rebuild.

## Deferred gates

Fresh-5000 dual-ORT and the external team validator were not run because no
candidate passed the earlier actual-cost + known-dual + truthful-shape +
provenance gates. Running them cannot turn a known-invalid graph into a winner.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps(winner, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

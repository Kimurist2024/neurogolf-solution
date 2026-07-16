#!/usr/bin/env python3
"""Finalize the B17 evidence bundle without touching promoted/root state."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str):
    return json.loads((HERE / name).read_text())


def dump(name: str, value) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def main() -> int:
    exact = load("exact_audit.json")["models"]
    candidate = load("candidate_audit.json")
    fresh = load("fresh5000_dual_ort.json")
    control280 = load("task280_control_fresh5000.json")
    control396 = load("task396_control_fresh5000.json")
    row280 = next(
        row for row in candidate["rows"] if row["path"].endswith("candidate_task280_truthful.onnx")
    )
    row396 = next(
        row for row in candidate["rows"] if row["path"].endswith("agent_corner_micro.onnx")
    )
    failure = {
        "baseline": "submission_base_7999.13.zip",
        "tasks": {
            "280": {
                "hash": "b527c5c6",
                "exact": {
                    "sha256": exact["280"]["sha256"],
                    "actual_cost": exact["280"]["actual_score"]["cost"],
                    "known_dual_ort": exact["280"]["known_dual"],
                    "shape_cloak": exact["280"]["runtime_shapes"]["shape_cloak"],
                    "runtime_shape_mismatch_count": len(
                        exact["280"]["runtime_shapes"]["mismatches"]
                    ),
                    "max_einsum_inputs": exact["280"]["max_einsum_inputs"],
                    "rejection": "default_ort_error_plus_shape_cloak_plus_giant_einsum",
                },
                "truthful_sound_control": {
                    "path": row280["path"],
                    "sha256": row280["sha256"],
                    "actual_cost": row280["actual_score"]["cost"],
                    "cost_delta_vs_exact": row280["actual_score"]["cost"]
                    - exact["280"]["actual_score"]["cost"],
                    "known_dual_ort": row280["known_dual"],
                    "fresh5000_dual_ort": control280["rows"],
                    "shape_cloak": row280["runtime_shapes"]["shape_cloak"],
                    "rejection": "sound_but_1333_cost_units_more_expensive",
                },
                "winner": None,
            },
            "396": {
                "hash": "fcb5c309",
                "exact": {
                    "sha256": exact["396"]["sha256"],
                    "actual_cost": exact["396"]["actual_score"]["cost"],
                    "known_dual_ort": exact["396"]["known_dual"],
                    "shape_cloak": exact["396"]["runtime_shapes"]["shape_cloak"],
                    "fresh5000_dual_ort": [
                        row for row in control396["rows"] if row["label"] == "exact_7999_13"
                    ],
                    "rejection": "fresh_wrong_46_of_5000_in_both_modes",
                },
                "strict_cheaper_fresh5000": fresh["rows"],
                "truthful_sound_control": {
                    "path": row396["path"],
                    "sha256": row396["sha256"],
                    "actual_cost": row396["actual_score"]["cost"],
                    "cost_delta_vs_exact": row396["actual_score"]["cost"]
                    - exact["396"]["actual_score"]["cost"],
                    "known_dual_ort": row396["known_dual"],
                    "fresh5000_dual_ort": [
                        row for row in control396["rows"] if row["label"] == "sound_control_1245"
                    ],
                    "shape_cloak": row396["runtime_shapes"]["shape_cloak"],
                    "rejection": "sound_but_226_cost_units_more_expensive",
                },
                "winner": None,
            },
        },
        "external_validator": {
            "status": "not_run",
            "reason": "zero strict-cheaper candidates survived fresh5000 dual ORT",
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
    dump("failure_manifest.json", failure)
    dump("winner_manifest.json", winner)
    dump("external_validator.json", failure["external_validator"])
    cheap396 = [row for row in fresh["rows"] if row["mode"] == "disabled"]
    cheap_lines = "\n".join(
        f"- cost {row['actual_cost']}, SHA `{row['sha256']}`: "
        f"{row['right']}/5000, wrong {row['wrong']}, errors {row['errors']} "
        "(default ORT produced the identical count)."
        for row in cheap396
    )
    report = f"""# B17 — task280 / task396 strict sound audit

## Outcome

Winner count **0**; valid cost gain **0**; valid score gain **0.0**. B17 did
not modify any root ZIP, CSV, best-score state, promoted artifact, or
`artifacts/handcrafted` model.

## Generator truth

- task280 / `b527c5c6`: two red side dots identify the outward directions of
  two green rectangles. Each dot emits a red centreline and a green band of
  radius `short_side-1` to the grid boundary. Flip and transpose preserve the
  rule. This needs exact emitter classification and full-distance rendering.
- task396 / `fcb5c309`: select the uniquely widest-and-tallest box, crop it,
  and recolor its border plus retained interior/static pixels with the other
  nonzero color. Correctness needs same-color box geometry; global frequency
  and all-nonzero-run shortcuts are not equivalent under random static.

## Exact 7999.13 audit

- task280: cost **828**, SHA `{exact['280']['sha256']}`. ORT_DISABLE_ALL is
  267/267, but default ORT cannot create the graph. The model has a 24-input
  Einsum and **{len(exact['280']['runtime_shapes']['mismatches'])}** declared/runtime
  shape mismatches. It is prohibited independently of its low cost.
- task396: cost **1019**, SHA `{exact['396']['sha256']}`. Known examples are
  266/266 in both ORT modes and runtime shapes are truthful, but the same fresh
  5000 cases produce **4954/5000** in each mode. This reproduces the
  private-black processing risk; bitwise-equivalent shaving cannot repair it.

## Task280 truthful rebuild

The B17 rebuild changes only the four carrier declarations of the
generator-derived `cand_pad20` graph to their real `[1,4,30]` runtime shapes.
It is bitwise-equivalent, max Einsum input count 10, both known modes 267/267,
both fresh modes 5000/5000, and has no shape cloak. Its SHA is
`{row280['sha256']}`. Truthful actual cost is **2161**, which is **1333 above**
the cost-828 comparator. The apparent cost-1209 form still had four cloak
tensors; the cost-884 form had 22 cloak tensors and a 21-input Einsum.

## Task396 cheap candidates

Every strictly cheaper known-correct, truthful candidate failed the mandatory
fresh5000 gate:

{cheap_lines}

The best failure rate is the cost-1014 compact occupancy graph at 4963/5000;
it still has 37 deterministic wrong outputs per ORT mode and cannot be adopted.

## Task396 SOUND control

The generator-derived corner parser SHA `{row396['sha256']}` is truthful,
known 266/266 in both modes, and fresh 5000/5000 in both modes. Actual cost is
**1245**, or **226 above** the unsound cost-1019 baseline. This establishes the
measured safe floor reached by the existing same-color geometry formulation.

## Final gates

No strictly cheaper candidate survived fresh5000 dual ORT, so there was no
finalist for the external validator. Full checker, strict shape inference,
standard-domain, no sparse/nested/function, banned-op, finite-initializer,
Conv-bias, and max-Einsum checks are recorded per candidate in
`candidate_audit.json`.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps(winner, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

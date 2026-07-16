#!/usr/bin/env python3
"""Assemble the final B15 evidence and empty winner manifest."""

from __future__ import annotations

import json
from pathlib import Path

import onnx

import audit_candidates


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def exact_structure() -> list[dict[str, object]]:
    rows = []
    for task in (23, 36):
        path = HERE / "baseline" / f"task{task:03d}.onnx"
        model = onnx.load(path)
        structure = audit_candidates.structural(model)
        runtime = audit_candidates.trace_runtime_shapes(model, task)
        semantic = audit_candidates.semantic_rejection(task, model, audit_candidates.sha256(path))
        rows.append(
            {
                "task": task,
                "path": str(path.relative_to(ROOT)),
                "sha256": audit_candidates.sha256(path),
                "structure": structure,
                "runtime_shape": runtime,
                "semantic_rejections": semantic,
                "eligible": structure["pass_before_runtime_shape"] and not runtime["shape_cloak"] and not semantic,
            }
        )
    (HERE / "exact_structure_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> int:
    baseline = json.loads((HERE / "baseline_audit.json").read_text())
    candidates = json.loads((HERE / "candidate_audit.json").read_text())
    generator = json.loads((HERE / "generator_rule_audit.json").read_text())
    controls = json.loads((HERE / "fresh5000_controls_dual_ort.json").read_text())
    structures = exact_structure()

    truthful = next(
        row
        for row in candidates["rows"]
        if row["path"].endswith("candidate_task036_truthful_gather.onnx")
    )
    archive1541 = [row for row in controls if row["label"] == "task023_archive1541"]
    truthful_fresh = [row for row in controls if row["label"] == "task036_truthful_gather"]
    task36_exact_structure = next(row for row in structures if row["task"] == 36)

    winner_manifest = {
        "lane": "b15",
        "base_zip": "submission_base_7999.13.zip",
        "tasks": [23, 36],
        "winners": [],
        "aggregate_gain": 0.0,
        "runtime_session_errors_in_final_controls": sum(row["errors"] for row in controls),
        "root_mutations_by_lane": [],
        "reason": "No candidate is simultaneously sound, no-cloak, complete-known-correct, fresh-perfect, and strictly cheaper than its exact base.",
    }
    failure_manifest = {
        "task023": {
            "exact_cost": baseline["costs"]["23"]["cost"],
            "exact_fresh_dual": [row for row in baseline["fresh"] if row["task"] == 23],
            "generator_non_injective_proof": generator["task023"],
            "archive_candidates": [row for row in candidates["rows"] if row["task"] == 23],
            "archive1541_fresh_dual": archive1541,
            "excluded_1497_evidence": "scripts/golf/loop_7999_13/lane_root21_task023_dual5000.json (13/5000 in both modes)",
            "decision": "NO_SOUND_DETERMINISTIC_REBUILD",
        },
        "task036": {
            "exact_cost": baseline["costs"]["36"]["cost"],
            "exact_fresh_dual": [row for row in baseline["fresh"] if row["task"] == 36],
            "exact_shape_cloak_mismatches": len(task36_exact_structure["runtime_shape"]["declared_runtime_mismatches"]),
            "exact_truthful_one_example_intermediate_bytes": task36_exact_structure["runtime_shape"]["truthful_one_example_intermediate_bytes"],
            "generator_reference": generator["task036"],
            "truthful_gather": truthful,
            "truthful_gather_fresh_dual": truthful_fresh,
            "decision": "SOUND_NO_CLOAK_CONTROL_COSTS_1428_ABOVE_BASE_325",
        },
    }
    manifest = {
        "winner_manifest": winner_manifest,
        "failure_manifest": failure_manifest,
        "exact_structure": structures,
        "eligible_archive_candidates": candidates["eligible_for_fresh5000"],
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(winner_manifest, indent=2) + "\n")
    (HERE / "failure_manifest.json").write_text(json.dumps(failure_manifest, indent=2) + "\n")
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = """# Lane B15 — task023/task036 sound rebuild audit

Base authority: `submission_base_7999.13.zip`. This lane did not modify any
root ZIP, CSV, score ledger, or shared handcrafted model.

## Outcome

- Winners: **0**; projected aggregate gain: **+0.00**.
- Exact measured costs: task023 **1622 = 1245 memory + 377 params**;
  task036 **325 = 255 memory + 70 params**.
- Known dual ORT: task023 266/266 in both modes; task036 265/265 in both
  modes. Runtime/session errors: 0.

## task023

The generator places latent 2x2 boxes and length-3 sticks into an unlabelled
gray union. Seeds 29685 and 120072 reproduce the same 9x10 input but different
legal outputs. Therefore no deterministic input-only ONNX can implement the
full generator relation.

The exact cost-1622 graph scored 4195/5000 (83.90%) in each ORT mode on seed
150799913, errors 0. The parent-excluded cost-1497 family is not reused; its
existing independent evidence is 13/5000 in both modes. The other two archive
models cost 1520 and 1541. They pass all 266 known cases in both modes, but use
ArgMax/GatherND/Scatter or TopK/Scatter lookup/rank pipelines; the 1520 model is
also an explicit PRIVATE0 artifact. The 1541 model independently scored only
4389/5000 (87.78%) in both modes, errors 0. All are rejected before promotion.

## task036

The generator rule is to identify the compact connected special-color object
and return the complete input crop at its tight bounding box. The direct numpy
rule passed 265/265 fixtures and independent fresh 5000/5000, errors 0.

The exact cost-325 graph is not structurally admissible: a runtime trace finds
14 declared/runtime shape contradictions and 20,329 truthful intermediate
bytes on one known example versus the reported 255 memory. On independent
fresh seed 150799913 it scored 4978/5000 in both modes; all 22 failures were
output-shape violations (60x60 instead of 30x30), with no ORT exception.

Archive static floors 212, 214, 230, 231, and 232 do not survive real scoring:
their actual costs/correctness are respectively 1457/wrong, 259/wrong,
57371/wrong, 348/correct, and 68469/wrong. Every one also uses the same
CenterCropPad shape-cloak family; the 212/214 models additionally use a
17-input giant Einsum.

As a ground-up control, `candidate_task036_truthful_gather.onnx` replaces the
variable Slice/pad carrier with a fixed 5x5 int64 GatherND and a validity mask.
It has fully static truthful shapes, standard domain, no giant Einsum, no UB,
no lookup table, no shape cloak, and real cost **1428 = 1194 + 234**. It passes
265/265 known examples in both modes and independent fresh 5000/5000 in both
modes, margin 1.0, runtime/session errors 0. It proves the rule can be expressed
safely, but costs 1103 more than the exact base, so it is not a winner.

## Decision

No model reaches the intersection of strict lower real cost, generator-rule
soundness, complete known dual correctness, fresh5000 dual correctness, and
the no-cloak/no-lookup/no-UB structure contract. `winner_manifest.json` is
therefore empty.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps(winner_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

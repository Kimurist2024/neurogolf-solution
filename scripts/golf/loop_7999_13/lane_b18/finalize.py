#!/usr/bin/env python3
"""Write the final B18 evidence bundle from completed audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text())


def row_for(audit: dict[str, Any], fragment: str) -> dict[str, Any]:
    return next(row for row in audit["rows"] if fragment in row["path"])


def main() -> int:
    audit = load("candidate_audit.json")
    fresh = load("fresh5000_baselines.json")
    differential = load("wave12_differential.json")
    base89 = row_for(audit, "lane_b18/baseline/task089.onnx")
    wave89 = row_for(audit, "wave12_exact_shave/task089.onnx")
    base255 = row_for(audit, "lane_b18/baseline/task255.onnx")
    fresh_rows = {
        (row["task"], row["mode"]): row for row in fresh["rows"]
    }
    result = {
        "status": "NO_SAFE_IMPROVEMENT",
        "strict_survivor_count": 0,
        "promotion": None,
        "baseline_zip": "submission_base_7999.13.zip",
        "baseline_zip_sha256": load("baseline_structure.json")["base_zip_sha256"],
        "assignment_cost_correction": {
            "task089": {"assigned": 1361, "exact_zip_recomputed": 1361},
            "task255": {"assigned": 1162, "exact_zip_recomputed": 1336},
        },
        "task089": {
            "baseline": {
                "sha256": base89["sha256"],
                "score": base89["actual_score"],
                "structure_pass": base89["structure"]["pass"],
                "shape_cloak": base89["runtime_shapes"]["shape_cloak"],
                "shape_mismatch_count": len(
                    base89["runtime_shapes"].get("mismatches", [])
                ),
                "known_dual": base89["known_dual"],
                "fresh5000_dual": [
                    fresh_rows[(89, "disabled")],
                    fresh_rows[(89, "default")],
                ],
            },
            "wave12_exact_shave": {
                "path": wave89["path"],
                "sha256": wave89["sha256"],
                "score": wave89["actual_score"],
                "graph_change": {
                    "removed_node_index": 88,
                    "op": "ReduceMax",
                    "inputs": ["decode_big"],
                    "outputs": ["keep_red_big"],
                },
                "structure_pass": wave89["structure"]["pass"],
                "shape_cloak": wave89["runtime_shapes"]["shape_cloak"],
                "shape_mismatch_count": len(
                    wave89["runtime_shapes"].get("mismatches", [])
                ),
                "known_dual": wave89["known_dual"],
                "differential": differential,
                "decision": "REJECT",
                "reasons": [
                    "disabled ORT: runtime error on all 267 known examples",
                    "disabled ORT: runtime error on all 5000 fresh examples",
                    "default ORT: both baseline and candidate fail session creation",
                    "50 declared/runtime shape mismatches",
                    "local try_candidate validator fails gold inference",
                ],
            },
            "truthful_controls": [
                {
                    "path": row["path"],
                    "score": row["actual_score"],
                    "shape_cloak": row["runtime_shapes"].get("shape_cloak"),
                    "known_dual": row["known_dual"],
                }
                for row in audit["rows"]
                if row["path"].endswith("task089/cand_u8.onnx")
                or row["path"].endswith("task089/candidate_rebuild_v11.onnx")
            ],
        },
        "task255": {
            "baseline": {
                "sha256": base255["sha256"],
                "score": base255["actual_score"],
                "structure_pass": base255["structure"]["pass"],
                "shape_cloak": base255["runtime_shapes"]["shape_cloak"],
                "shape_mismatch_count": len(
                    base255["runtime_shapes"].get("mismatches", [])
                ),
                "known_dual": base255["known_dual"],
                "fresh5000_dual": [
                    fresh_rows[(255, "disabled")],
                    fresh_rows[(255, "default")],
                ],
            },
            "generator_ambiguity_proof": {
                "same_input": True,
                "same_output": False,
                "output_diff_cells": 15,
                "source": "scripts/golf/scratch_codex/task255/ambiguity_proof.py",
            },
            "decision": "REJECT_ALL",
            "reasons": [
                "generator is non-functional: two valid configs have identical input and different output",
                "baseline fresh accuracy is 4723/5000 = 94.46% in both ORT modes",
                "no deterministic input-only ONNX can be exact on every generated instance",
            ],
        },
        "external_validator": {
            "status": "NOT_RUN",
            "reason": "No candidate passed the mandatory local/dual-ORT/runtime-shape pre-gate",
            "requested_path": "/Users/kimura2003/Downloads/neurogolf_team_validator_v1",
            "requested_path_present": False,
        },
        "root_submission_modified": False,
        "eligible_for_fresh5000_from_strict_pregate": audit[
            "eligible_for_fresh5000"
        ],
    }
    (HERE / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "external_validator.json").write_text(
        json.dumps(result["external_validator"], indent=2) + "\n"
    )
    (HERE / "failure_manifest.json").write_text(
        json.dumps(
            {
                "status": result["status"],
                "strict_survivor_count": 0,
                "task089_wave12_reasons": result["task089"][
                    "wave12_exact_shave"
                ]["reasons"],
                "task255_reasons": result["task255"]["reasons"],
                "external_validator": result["external_validator"],
            },
            indent=2,
        )
        + "\n"
    )
    report = """# B18 task089 / task255 report

## Verdict

**No safe improvement was found and nothing was promoted.** The exact
`submission_base_7999.13.zip` members recompute to task089 cost **1361** and
task255 cost **1336**. The assignment's task255 value 1162 does not match the
exact ZIP member (SHA256 `4d3ebe16cc55...`).

## task089

The exact baseline passes all 267 known examples only under ORT with graph
optimizations disabled. Runtime tracing finds **50 declared/actual shape
mismatches**. On fresh5000 it scores 4977/5000 (99.54%) under disabled ORT, but
default ORT cannot create a session.

The priority Wave12 candidate is exactly the archive rank-1 candidate
(SHA256 `33db6c4a4422...`) and reduces measured cost 1361 -> 1184 by removing
one apparently dead `ReduceMax(decode_big -> keep_red_big)` node. That node is
not semantically consumed, but it is operationally required by the incumbent's
shape-cloaked ORT buffer plan:

- disabled ORT: candidate runtime error on 267/267 known examples;
- disabled ORT: candidate runtime error on 5000/5000 fresh examples;
- default ORT: baseline and candidate both fail session creation;
- local `try_candidate` passes structural validation but fails gold inference;
- runtime trace still shows 50 declared/actual shape mismatches.

Therefore the 95%-accuracy base-equivalence exception does not apply: the base
fresh rate is above 95%, but the shave is not executable and cannot produce a
single raw output for a bitwise comparison. The cheapest truthful no-cloak
known-correct controls cost 4142 (`candidate_rebuild_v11.onnx`) and 20362
(`cand_u8.onnx`), both worse than 1361.

## task255

The exact baseline is cost 1336, has 16 declared/actual shape mismatches, and
passes all 265 known examples in both ORT modes. It fails fresh generation in
both modes at **4723/5000 = 94.46%**, below the permitted 95% threshold.

More importantly, the generator is provably non-functional as an input/output
mapping. The independent ambiguity script constructs two valid generator
configurations with byte-identical inputs and outputs differing in 15 cells.
No deterministic input-only ONNX can be exact for both, so a cheaper public-fit
candidate is not a sound score improvement.

## Gate disposition

Full checker, strict shape/data propagation, known dual-ORT, runtime shape
tracing, fresh5000, raw/sanitized differential, and the local candidate
validator leave **zero strict survivors**. The external validator was not run
because no model reached that gate; additionally the requested
`/Users/kimura2003/Downloads/neurogolf_team_validator_v1` directory is absent
in this environment. Root ZIP/CSV/best/artifacts/handcrafted files were not
modified by B18.
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"status": result["status"], "survivors": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

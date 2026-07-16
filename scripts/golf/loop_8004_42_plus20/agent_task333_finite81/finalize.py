#!/usr/bin/env python3
"""Fail-closed finalizer for the task333 finite/residual-support lane."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WINNER_SHA = "0628a573302f0a816d010482ed8b883caac7c307a27f47c9b53df85e2042a6bc"
CONFIGS = (
    "disable_all_threads1",
    "disable_all_threads4",
    "default_threads1",
    "default_threads4",
)


def load(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    inventory = load(HERE / "candidate_inventory.json")
    strict = load(HERE / "strict_lower_audit.json")
    proof = load(HERE / "sign_equivalence_proof.json")
    factor = load(HERE / "factor_support_audit.json")
    generator = load(HERE / "generator_support_analysis.json")
    margins = {name: load(HERE / "margin_support" / f"{name}.json") for name in CONFIGS}

    assert inventory["baseline_zip_sha256"] == digest(ROOT / "submission_base_8005.17.zip")
    assert inventory["baseline_profile"] == {"memory": 0, "params": 423, "cost": 423}
    assert inventory["unique_sha_count"] == 41
    assert inventory["strict_lower_unique_count"] == 4
    winner = next(row for row in strict["rows"] if row["sha256"] == WINNER_SHA)
    rejected = [row for row in strict["rows"] if row["sha256"] != WINNER_SHA]
    assert winner["profile"] == {"memory": 0, "params": 421, "cost": 421}
    assert winner["known_perfect_all_configs"]
    assert winner["runtime_output_shape_truthful"]
    assert all(not row["known_perfect_all_configs"] for row in rejected)
    assert sorted(row["profile"]["cost"] for row in rejected) == [412, 412, 412]
    assert proof["proved"] and factor["complete"]
    assert not generator["raw_support"]["full_raw_enumeration_feasible"]
    assert generator["residual_reduction"]["semantic_generator_residual_states"] == 0
    assert all(row["perfect"] and row["total"] == 2000 for row in margins.values())

    static = winner["static"]
    static_pass = (
        static["full_check"]
        and static["strict_data_prop"]
        and static["all_node_outputs_static_positive"]
        and static["standard_domains"]
        and static["conv_bias_ub0"]
        and not static["lookup"]
        and static["nested_graph_count"] == 0
        and static["functions_count"] == 0
        and static["sparse_initializer_count"] == 0
        and not static["banned_ops"]
        and static["finite_initializers"]
    )
    assert static_pass
    gain = math.log(423 / 421)
    margin_summary = {
        "executed_valid_generator_cases": sum(row["total"] for row in margins.values()),
        "right": sum(row["right"] for row in margins.values()),
        "wrong": sum(row["wrong"] for row in margins.values()),
        "runtime_errors": sum(row["runtime_errors"] for row in margins.values()),
        "nonfinite_values": sum(row["nonfinite_values"] for row in margins.values()),
        "near_positive_values": sum(row["near_positive_values"] for row in margins.values()),
        "sign_differences_vs_baseline": sum(row["sign_differences_vs_baseline"] for row in margins.values()),
        "raw_different_values_vs_baseline": sum(row["raw_different_values_vs_baseline"] for row in margins.values()),
        "max_abs_raw_difference_vs_baseline": max(row["max_abs_raw_difference_vs_baseline"] for row in margins.values()),
        "min_positive": min(row["min_positive"] for row in margins.values()),
        "configs": margins,
        "perfect_all_configs": True,
    }
    assert margin_summary["right"] == margin_summary["executed_valid_generator_cases"] == 8000
    assert margin_summary["wrong"] == margin_summary["runtime_errors"] == margin_summary["nonfinite_values"] == margin_summary["near_positive_values"] == margin_summary["sign_differences_vs_baseline"] == 0

    accepted = {
        "task": 333,
        "path": winner["extracted_path"],
        "canonical_source": "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx",
        "sha256": WINNER_SHA,
        "baseline_cost": 423,
        "candidate_cost": 421,
        "actual_profile": winner["profile"],
        "projected_gain": gain,
        "static": static,
        "known_four_configs": winner["known_four_configs"],
        "runtime_output_shape_truthful": True,
        "exact_all_input_termwise_proof": proof,
        "generator_support_analysis": generator,
        "changed_factor_full_support_four_configs": factor,
        "whole_model_margin_four_configs": margin_summary,
        "accepted": True,
    }
    result = {
        "baseline_zip": inventory["baseline_zip"],
        "baseline_zip_sha256": inventory["baseline_zip_sha256"],
        "baseline_task333_sha256": inventory["baseline_task333_sha256"],
        "baseline_profile": inventory["baseline_profile"],
        "inventory": {
            "onnx_files_seen": inventory["onnx_files_seen"],
            "zip_files_seen": inventory["zip_files_seen"],
            "zip_task333_members_seen": inventory["zip_task333_members_seen"],
            "total_source_references": inventory["total_source_references"],
            "unique_sha_count": inventory["unique_sha_count"],
            "strict_lower_unique_count": inventory["strict_lower_unique_count"],
        },
        "strict_lower_candidates": [
            {
                "sha256": row["sha256"],
                "path": row["extracted_path"],
                "source_count": row["source_count"],
                "actual_profile": row["profile"],
                "known_right_by_config": {name: data["right"] for name, data in row["known_four_configs"].items()},
                "known_perfect_all_configs": row["known_perfect_all_configs"],
                "decision": "accept" if row["sha256"] == WINNER_SHA else "reject_known",
            }
            for row in strict["rows"]
        ],
        "accepted": [accepted],
        "accepted_count": 1,
        "aggregate_projected_gain": gain,
        "policy": {
            "strictly_lower_actual_cost": True,
            "known_four_configs_required": 1.0,
            "runtime_errors": 0,
            "nonfinite_values": 0,
            "near_positive_values": 0,
            "giant_contraction": "requires all-input algebraic proof plus complete changed-factor support and whole-model four-config margin",
            "raw_generator_support": "not substituted by fresh sampling; exact all-input termwise proof eliminates semantic dependence on the exponential generator support",
            "truthful_static_shapes_standard_ub0_lookup0_cloak0": True,
        },
        "promotion_performed": False,
        "protected_root_files_modified": False,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    manifest = {
        "baseline_zip": inventory["baseline_zip"],
        "baseline_zip_sha256": inventory["baseline_zip_sha256"],
        "winners": [
            {
                "task": 333,
                "member": "task333.onnx",
                "source": accepted["canonical_source"],
                "sha256": WINNER_SHA,
                "old_cost": 423,
                "new_cost": 421,
                "gain": gain,
                "full_gate_pass": True,
            }
        ],
        "winner_count": 1,
        "aggregate_projected_gain": gain,
        "promotion_performed": False,
        "protected_files_modified": False,
    }
    (HERE / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    candidate_lines = []
    for row in sorted(strict["rows"], key=lambda item: (item["profile"]["cost"], item["sha256"])):
        rights = "/".join(str(data["right"]) for data in row["known_four_configs"].values())
        candidate_lines.append(
            f"| `{row['sha256'][:12]}…` | {row['profile']['cost']} | {rights} | "
            f"{'accept' if row['sha256'] == WINNER_SHA else 'reject known'} |"
        )
    margin_lines = []
    for name in CONFIGS:
        row = margins[name]
        margin_lines.append(
            f"| {name} | {row['right']}/{row['total']} | {row['wrong']} | {row['runtime_errors']} | "
            f"{row['nonfinite_values']} | {row['near_positive_values']} | {row['sign_differences_vs_baseline']} | {row['min_positive']:.8g} |"
        )
    report = f"""# task333 finite/residual-support audit

## Outcome

One candidate is accepted as a non-promoted winner against immutable
`submission_base_8005.17.zip` (`{inventory['baseline_zip_sha256']}`):
`task333_r01.onnx`, SHA `{WINNER_SHA}`.  Actual cost is `423 -> 421` and
projected gain is `{gain:.12f}`.  No ZIP or protected root file was modified.

## Complete candidate inventory

The scan visited {inventory['onnx_files_seen']} task333 ONNX files and
{inventory['zip_task333_members_seen']} task333 members across
{inventory['zip_files_seen']} ZIPs: {inventory['total_source_references']} source
references deduplicated to {inventory['unique_sha_count']} SHA values.  Only four
unique SHA values are strictly lower than the actual baseline cost 423.

| SHA | cost | known right (four configs) | decision |
|---|---:|---|---|
{chr(10).join(candidate_lines)}

All three cost-412 latent-prune files fail the mandatory known gate.  The sole
known-perfect SHA is the cost-421 sign absorption.

## Why exponential generator support can be reduced exactly

The raw generator support is not practically enumerable.  A conservative
constructive subset alone has `36*56*3^63 =
{generator['raw_support']['constructive_valid_two_colour_lower_bound']}` valid
inputs.  This lane does not replace that support with fresh sampling.

The baseline has one Einsum and three input occurrences.  The removed factor is
`GE=[1,-1]`.  The candidate sets `HC_new[Z,d]=GE[Z]*HC_old[Z,d]`; the shared HC
use is compensated by `GHHT_new[t,U]=GHHT_old[t,U]*GE[U]`.  Therefore its second
use contains `GE[U]^2=1`.  Every monomial for every complete Einsum index
assignment is exactly unchanged, for every possible input tensor.  All other
initializers are byte-identical.

The complete changed-factor support is only `2*10 + 3*2*10 = 80` entries.  All
80 entries were executed and matched exactly in disabled/default ORT with
threads 1/4, with runtime errors 0 and nonfinite values 0.

## Whole-model platform and margin evidence

Known is `265/265` independently in every configuration.  In addition, two
independent generator seeds supplied 2,000 valid cases per configuration:

| configuration | truth | wrong | runtime | nonfinite | (0,0.25) | sign diff vs baseline | min positive |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(margin_lines)}

Floating contraction order can change raw magnitudes; that is recorded rather
than hidden.  Across all 8,000 whole-model cases the maximum raw difference is
`{margin_summary['max_abs_raw_difference_vs_baseline']:.8g}`, while sign
differences, truth errors, runtime errors, nonfinite values, and near-positive
values are all zero.

## Structural gates

Actual profiler is `0 memory + 421 params = 421`.  ONNX full check, strict
data-propagating shape inference, positive static shapes, truthful runtime
output `[1,10,30,30]`, standard domains, Conv-family UB0, lookup0, nested
graph/function/sparse0, banned-op0, and finite initializer gates all pass.  The
35-input giant Einsum is accepted only because the all-input termwise proof and
complete changed-factor residual audit above close its guarantee gap.

## Evidence

- `candidate_inventory.json`: all 1,793 source references and 41 unique SHA rows
- `strict_lower_audit.json`: actual cost and known x4 for all four lower SHA rows
- `sign_equivalence_proof.json`: exact all-input monomial proof
- `generator_support_analysis.json`: raw-support lower bound and reduction
- `factor_support_audit.json`: complete 80-entry residual x4
- `margin_support/*.json`: two-seed whole-model truth/margin x4
- `result.json`, `winner_manifest.json`: machine-readable disposition
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"accepted": 1, "gain": gain, "whole_model_cases": 8000}, indent=2))


if __name__ == "__main__":
    main()

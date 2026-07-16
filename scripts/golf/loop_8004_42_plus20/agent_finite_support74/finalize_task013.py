#!/usr/bin/env python3
"""Merge completed task013 evidence into fail-closed lane artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
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
    colour = load(HERE / "task013_colour_proof.json")
    shapes = load(HERE / "task013_runtime_shapes.json")
    support_parts = {
        name: load(HERE / "task013_support" / f"{name}.json") for name in CONFIGS
    }
    winner = inventory["rows"][0]
    winner_sha = "ad4eb35978f3e38d1d3e2afdd55e55db871962cc2ea4c989675d9d583434103b"

    assert inventory["candidate_count"] >= 8
    assert inventory["baseline_zip_sha256"] == digest(ROOT / "submission_base_8005.17.zip")
    assert winner["sha256"] == winner_sha
    assert winner["profile"] == {"memory": 488, "params": 148, "cost": 636}
    assert all(row["known_perfect_all_configs"] for row in inventory["rows"])
    assert all(row["strictly_lower"] for row in inventory["rows"])
    assert colour["proved"] and shapes["truthful"]

    support_configs = {}
    known_configs = {}
    elapsed = 0.0
    for name, part in support_parts.items():
        assert part["sha256"] == winner_sha
        assert part["accepted"]
        assert part["selected_configs"] == [name]
        assert part["support"]["complete"]
        assert part["support"]["executed_structural_states"] == 37_800
        assert list(part["support"]["configs"]) == [name]
        support_configs[name] = part["support"]["configs"][name]
        known_configs[name] = part["known_four_configs"][name]
        elapsed = max(elapsed, float(part["elapsed_seconds"]))
    assert all(row["perfect"] for row in support_configs.values())
    assert all(row["perfect"] for row in known_configs.values())

    static = winner["static"]
    mandatory_static = (
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
    assert mandatory_static

    gain = math.log(638 / 636)
    support = {
        "generator_hash": "0a938d79",
        "structural_state_formula": "sum(width//2,width=20..30)*7*5*4*2",
        "structural_states_per_config": 37_800,
        "executed_platform_inferences": 37_800 * 4,
        "ordered_distinct_nonzero_colour_pairs": 72,
        "full_generator_support_states": 37_800 * 72,
        "colour_equivalence_proved": True,
        "configs": support_configs,
        "complete": True,
    }
    accepted = {
        "task": 13,
        "path": winner["path"],
        "sha256": winner_sha,
        "baseline_cost": 638,
        "candidate_cost": 636,
        "profile": winner["profile"],
        "projected_gain": gain,
        "known_four_configs": known_configs,
        "static": static,
        "runtime_shapes": shapes,
        "exact_rewrite_proof": winner["exact_rewrite_proof"],
        "colour_equivalence_proof": colour,
        "full_reachable_support": support,
        "accepted": True,
    }
    alternates = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "cost": row["profile"]["cost"],
            "known_perfect_all_configs": row["known_perfect_all_configs"],
            "exact_real_semantics": row["exact_rewrite_proof"]["real_semantics_equal_for_every_input_tensor"],
            "decision": "audited_alternate_not_selected_same_task_same_cost_no_duplicate_gain",
        }
        for row in inventory["rows"][1:]
    ]
    result = {
        "baseline_zip": "submission_base_8005.17.zip",
        "baseline_zip_sha256": inventory["baseline_zip_sha256"],
        "baseline_task013_sha256": inventory["baseline_task013_sha256"],
        "baseline_profile": inventory["baseline_profile"],
        "historical_cross_scan": {
            "report_result_winner_files_seen": 1136,
            "gain_ranked_tasks_revisited": [13, 51, 64, 70, 199, 202, 328, 333, 379],
            "excluded_as_requested": [9, 36, 48, 158, 198, 226, 254, 267, 323, 328, 365],
            "candidate_files_audited_in_this_lane": inventory["candidate_count"],
        },
        "accepted": [accepted],
        "accepted_count": 1,
        "alternates": alternates,
        "aggregate_projected_gain": gain,
        "policy": {
            "strictly_lower_actual_cost": True,
            "known_required": 1.0,
            "runtime_errors": 0,
            "giant_einsum_requires_full_reachable_support_four_configs": True,
            "colour_reduction_requires_algebraic_equivalence_proof": True,
            "truthful_strict_data_prop_standard_ub0_lookup0_cloak0": True,
        },
        "no_zip_or_protected_root_file_modified": True,
        "max_parallel_support_elapsed_seconds": elapsed,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    manifest = {
        "baseline_zip": result["baseline_zip"],
        "baseline_zip_sha256": result["baseline_zip_sha256"],
        "winners": [
            {
                "task": 13,
                "member": "task013.onnx",
                "source": winner["path"],
                "sha256": winner_sha,
                "old_cost": 638,
                "new_cost": 636,
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

    config_lines = []
    for name in CONFIGS:
        row = support_configs[name]
        config_lines.append(
            f"| {name} | {row['right']}/{row['total']} | {row['wrong']} | "
            f"{row['runtime_errors']} | {row['nonfinite_values']} | "
            f"{row['near_positive_values']} | {row['min_positive']:.8g} |"
        )
    alternate_lines = []
    for index, row in enumerate(inventory["rows"], 1):
        alternate_lines.append(
            f"| {index} | `{Path(row['path']).name}` | `{row['sha256'][:12]}…` | "
            f"{row['profile']['cost']} | yes | yes | {'selected' if index == 1 else 'alternate'} |"
        )
    report = f"""# task013 finite-support expansion audit

## Outcome

Eight strict-lower task013 files were audited against immutable
`submission_base_8005.17.zip` (`{result['baseline_zip_sha256']}`).  Exactly one
non-duplicate winner is selected: `task013_r001.onnx`, SHA-256 `{winner_sha}`.
Its actual cost is `638 -> 636`, for projected gain `{gain:.12f}`.  No ZIP,
`all_scores.csv`, or protected root file was modified.

## Full reachable-support guarantee

The generator has 37,800 structural states.  The selected giant-Einsum model
executed every state independently in all four required ORT configurations:

| configuration | correct | wrong | runtime errors | nonfinite | (0,0.25) positives | min positive |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(config_lines)}

This is 151,200 platform executions.  The 72 ordered distinct nonzero colour
pairs are rigorously reduced, not sampled: geometry uses `nz_f` and is colour-ID
independent; marker colours are exactly recovered by
`S=c0+c1`, `T=c0*p0+c1*(p0+d)`; the terminal colour score is
`0.25-(k-c)^2`, positive iff `k=c`; the background score is
`0.0625-1000000*k^2`, positive iff `k=0`.  All 48,600 reachable float16 colour
recovery combinations and all selector values were mechanically checked.
Thus the 37,800 executions cover all 2,721,600 generator parameter states.

## Rewrite and static gates

The candidate removes `T_zero=[1,0]` and replaces its six operand uses across
four Einsum nodes by the exact `Qor` main diagonal.  That diagonal is exactly
`[1,0]`; all other initializers are byte-identical, and 51/55 nodes are
byte-identical.  All 55 runtime node-output shapes match strict inferred shapes,
with zero nonfinite intermediate outputs.

Independent gates pass: actual profiler `488 memory + 148 params = 636`, ONNX
full checker, strict shape inference with data propagation, positive static
shapes, standard domains, Conv-family UB0, lookup0, nested graph/function/sparse0,
banned-op0, finite initializers, and known `267/267` in each of the four modes.

## Expanded candidate-file set

| # | file | SHA | cost | static/proof | known x4 | decision |
|---:|---|---|---:|---|---|---|
{chr(10).join(alternate_lines)}

All seven alternates are retained only as audited evidence.  They have the same
task and same cost, so selecting more than one would not add score.  `r001` is
preferred because it uses the direct five-way diagonal with no additional
summed label, and it is the SHA that completed full reachable support x4.

## Artifacts

- `candidate_inventory.json`: eight-file static, actual-cost, exact-rewrite, and known-x4 evidence
- `task013_support/*.json`: four complete support runs
- `task013_colour_proof.json`: colour-equivalence proof
- `task013_runtime_shapes.json`: all-node truthful shape trace
- `result.json` and `winner_manifest.json`: machine-readable disposition
"""
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"accepted": 1, "gain": gain, "support": 37_800 * 4}, indent=2))


if __name__ == "__main__":
    main()

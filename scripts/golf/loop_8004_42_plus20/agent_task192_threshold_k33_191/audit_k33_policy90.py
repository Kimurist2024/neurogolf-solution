#!/usr/bin/env python3
"""Actual-ONNX POLICY90 audit for task192 fixed-threshold k33.

POLICY90 admissibility is deliberately separate from all-support exactness.
This script is evidence-only and never promotes or edits the candidate/stage.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/root_task192_threshold_188/candidates"
    / "task192_hardsigmoid_k33.onnx"
)
STAGED = ROOT / "others/71407/task192.onnx"
AUTHORITY = ROOT / "submission_base_8009.46.zip"
ROOT_SUBMISSION = ROOT / "submission.zip"
ALL_SCORES = ROOT / "all_scores.csv"
SUPPORT190 = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task192_threshold_support_190/audit/result.json"
)

EXPECTED = {
    "authority_zip": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "root_submission": "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927",
    "all_scores": "8c99379ce1c77e6894d53b593ccb5ae10f01c3c70e5666f832b1b2454c709b78",
    "authority_task192": "e7f9a11b93b611acfa4bba39e90e1ddf24223d50add4277fe9716f21f6ede10c",
    "staged": "19fbdce89a5c89f5ff376b2fbbdb630ead5535d5ed5ebe7d9914a4de89e5023c",
    "candidate": "e6515b2ddf32c2eb80581aa3267e24683d2aa53d9445483b2a2a0752f94072d5",
}
FRESH_SEEDS = (192_800_661, 192_930_007)
FRESH_PER_SEED = 5000
POLICY_ACCURACY = 0.90
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(
    "task192_k33_191_base",
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task192_threshold_review_189"
    / "audit_threshold.py",
)

# BASE inserts both scripts/ and the ARC generator directory into sys.path.
from lib import scoring  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def hash_snapshot(authority_task_data: bytes) -> dict[str, Any]:
    observed = {
        "authority_zip": sha256_path(AUTHORITY),
        "root_submission": sha256_path(ROOT_SUBMISSION),
        "all_scores": sha256_path(ALL_SCORES),
        "authority_task192": sha256_bytes(authority_task_data),
        "staged": sha256_path(STAGED),
        "candidate": sha256_path(CANDIDATE),
    }
    return {"observed": observed, "expected": EXPECTED, "all_match": observed == EXPECTED}


def update_range(stats: dict[str, dict[str, int | None]], name: str, value: int) -> None:
    item = stats.setdefault(name, {"min": None, "max": None})
    item["min"] = value if item["min"] is None else min(int(item["min"]), value)
    item["max"] = value if item["max"] is None else max(int(item["max"]), value)


def color_stats_k33(examples: list[dict[str, Any]]) -> dict[str, Any]:
    ranges: dict[str, dict[str, int | None]] = {}
    box_color_is_argmax = 0
    unique_argmax = 0
    threshold_exact_one = 0
    threshold_condition_success = 0
    false_negative_count = 0
    distractor_selected_count = 0
    for example in examples:
        grid = example["input"]
        output = example["output"]
        flat = [value for row in grid for value in row]
        out_flat = [value for row in output for value in row]
        counts = [flat.count(color) for color in range(10)]
        dominant = max(range(1, 10), key=lambda color: counts[color])
        ranked = sorted((counts[color], color) for color in range(1, 10))
        runner_up = ranked[-2][0]
        output_colors = sorted(set(out_flat) - {0})
        box_color = output_colors[0] if len(output_colors) == 1 else -1
        threshold_channels = sum(count >= 34 for count in counts[1:])
        positive_counts = [count for count in counts[1:] if count > 0]
        condition = (
            box_color == dominant
            and counts[dominant] >= 34
            and runner_up <= 33
            and threshold_channels == 1
        )
        update_range(ranges, "height", len(grid))
        update_range(ranges, "width", len(grid[0]))
        update_range(ranges, "dominant_color_count", counts[dominant])
        update_range(ranges, "runner_up_color_count", runner_up)
        update_range(ranges, "dominant_minus_runner_up", counts[dominant] - runner_up)
        update_range(ranges, "threshold_selected_channel_count", threshold_channels)
        update_range(ranges, "positive_nonzero_color_channel_count", len(positive_counts))
        update_range(ranges, "minimum_positive_nonzero_color_count", min(positive_counts))
        update_range(ranges, "maximum_positive_nonzero_color_count", max(positive_counts))
        update_range(ranges, "box_color_input_count", counts[box_color] if box_color > 0 else -1)
        update_range(ranges, "output_box_cell_count", sum(value != 0 for value in out_flat))
        box_color_is_argmax += int(box_color == dominant)
        unique_argmax += int(counts[dominant] > runner_up)
        threshold_exact_one += int(threshold_channels == 1)
        threshold_condition_success += int(condition)
        false_negative_count += int(counts[dominant] <= 33)
        distractor_selected_count += int(runner_up >= 34)
    total = len(examples)
    return {
        "total": total,
        "ranges": ranges,
        "box_color_is_argmax": box_color_is_argmax,
        "unique_nonzero_argmax": unique_argmax,
        "hard_sigmoid_selects_exactly_one_channel": threshold_exact_one,
        "threshold_condition_success": threshold_condition_success,
        "threshold_condition_failures": total - threshold_condition_success,
        "box_not_selected_count": false_negative_count,
        "distractor_selected_count": distractor_selected_count,
        "all_support_exact_on_sample": threshold_condition_success == total,
        "threshold_semantics": (
            "HardSigmoid(alpha=1,beta=-33) maps integer hist count <=33 to 0 "
            "and >=34 to 1. It matches ArgMax+OneHot only when exactly the true "
            "dominant channel reaches 34."
        ),
    }


def explicit_parameters() -> dict[str, dict[str, Any]]:
    false_negative = {
        "width": 10,
        "height": 10,
        "rows": [0],
        "cols": [0],
        "color": 1,
        "boxrows": [0, 0, 4],
        "boxcols": [0, 4, 0],
        "wides": [3, 3, 3],
        "talls": [3, 3, 3],
        "boxcolor": 2,
    }
    outside = [
        (row, col)
        for row in range(5, 20)
        for col in range(20)
        if (row + col) % 2 == 0
    ][:37]
    false_positive = {
        "width": 20,
        "height": 20,
        "rows": [row for row, _ in outside],
        "cols": [col for _, col in outside],
        "color": 1,
        "boxrows": [0, 0, 0],
        "boxcols": [0, 5, 10],
        "wides": [4, 4, 4],
        "talls": [4, 4, 4],
        "boxcolor": 2,
    }
    return {"false_negative": false_negative, "false_positive": false_positive}


def explicit_counterexamples(candidate_data: bytes, staged_data: bytes) -> dict[str, Any]:
    generator = importlib.import_module("task_7e0986d6")
    common = importlib.import_module("common")
    rows = {}
    for name, parameters in explicit_parameters().items():
        example = generator.generate(**parameters)
        points = sorted(zip(parameters["rows"], parameters["cols"]))
        reachable = bool(
            common.remove_neighbors(points) == points
            and not common.overlaps(
                parameters["boxrows"],
                parameters["boxcols"],
                parameters["wides"],
                parameters["talls"],
                1,
            )
        )
        comparison = BASE.run_cases(
            [example],
            BASE.make_session(candidate_data, True, 1),
            BASE.make_session(staged_data, True, 1),
            check_rule=True,
        )
        rows[name] = {
            "parameters": parameters,
            "reachable_generator_parameters": reachable,
            "color_count_statistics": color_stats_k33([example]),
            "actual_onnx_comparison": comparison,
            "candidate_is_counterexample": bool(
                comparison["candidate_right"] == 0
                and comparison["staged_right"] == 1
                and comparison["candidate_runtime_errors"] == 0
                and comparison["candidate_nonfinite_values"] == 0
                and comparison["sign_equal"] == 0
            ),
        }
    return {
        "all_support_exact": False,
        "cases": rows,
        "both_actual_counterexamples": all(
            row["candidate_is_counterexample"] for row in rows.values()
        ),
        "interpretation": (
            "These reachable inputs prove k33 is not an all-support identity. "
            "That fact is disclosed but is not a POLICY90 rejection criterion."
        ),
    }


def policy90_fresh_pass(row: dict[str, Any]) -> bool:
    comparison = row["comparison"]
    return bool(
        row["generated"] == FRESH_PER_SEED
        and row["generation_errors"] == 0
        and comparison["converted"] == FRESH_PER_SEED
        and comparison["conversion_errors"] == 0
        and comparison["candidate_accuracy"] is not None
        and float(comparison["candidate_accuracy"]) >= POLICY_ACCURACY
        and comparison["candidate_runtime_errors"] == 0
        and comparison["staged_runtime_errors"] == 0
        and comparison["one_sided_runtime_errors"] == 0
        and comparison["candidate_nonfinite_values"] == 0
        and comparison["staged_nonfinite_values"] == 0
        and comparison["candidate_near_positive_values"] == 0
        and comparison["staged_right"] == FRESH_PER_SEED
        and comparison["readable_rule_right"] == FRESH_PER_SEED
    )


def main() -> int:
    ort.set_default_logger_severity(4)
    candidate_data = CANDIDATE.read_bytes()
    staged_data = STAGED.read_bytes()
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = archive.read("task192.onnx")
    before = hash_snapshot(authority_data)
    if not before["all_match"]:
        raise RuntimeError(f"immutable input hash mismatch: {before}")

    support = json.loads(SUPPORT190.read_text())
    support_k33 = support["monte_carlo"]["aggregate"]["thresholds"]["33"]
    print("structure, official cost, and truthful shapes", flush=True)
    structural = BASE.structural_audit(candidate_data)
    official = {
        "immutable_8009_46": BASE.official_score(authority_data, "k33_immutable"),
        "staged_argmax_onehot": BASE.official_score(staged_data, "k33_staged"),
        "candidate_hardsigmoid_k33": BASE.official_score(candidate_data, "candidate_k33"),
    }
    known = BASE.known_examples()
    benchmark = scoring.convert_to_numpy(known[0])
    if benchmark is None:
        raise RuntimeError("first known example did not convert")
    candidate_shape = BASE.runtime_shape_trace(candidate_data, benchmark["input"])
    staged_shape = BASE.runtime_shape_trace(staged_data, benchmark["input"])
    known_stats = color_stats_k33(known)

    known_four: dict[str, Any] = {}
    for disable, threads, label in CONFIGS:
        print(f"known config {label}", flush=True)
        known_four[label] = BASE.run_cases(
            known,
            BASE.make_session(candidate_data, disable, threads),
            BASE.make_session(staged_data, disable, threads),
            check_rule=True,
        )

    explicit = explicit_counterexamples(candidate_data, staged_data)
    print("explicit all-support counterexamples confirmed", flush=True)
    fresh_rows = []
    for seed in FRESH_SEEDS:
        print(f"fresh seed {seed}: generating", flush=True)
        examples, generation_errors, attempts = BASE.fresh_examples(seed, FRESH_PER_SEED)
        comparison = BASE.run_cases(
            examples,
            BASE.make_session(candidate_data, True, 1),
            BASE.make_session(staged_data, True, 1),
            check_rule=True,
            progress_label=f"fresh seed {seed}",
        )
        row = {
            "seed": seed,
            "requested": FRESH_PER_SEED,
            "generated": len(examples),
            "generation_attempts": attempts,
            "generation_errors": generation_errors,
            "color_count_statistics": color_stats_k33(examples),
            "comparison": comparison,
        }
        row["policy90_pass"] = policy90_fresh_pass(row)
        fresh_rows.append(row)

    after = hash_snapshot(authority_data)
    reasons: list[str] = []
    if not structural["pass"]:
        reasons.append("structural_gate_failed")
    expected_costs = {
        "immutable_8009_46": 1609,
        "staged_argmax_onehot": 1149,
        "candidate_hardsigmoid_k33": 1138,
    }
    for label, expected_cost in expected_costs.items():
        row = official[label]
        if row is None or not row.get("correct") or int(row.get("cost", -1)) != expected_cost:
            reasons.append(f"official_score_failed_{label}")
    if not candidate_shape["truthful"] or not staged_shape["truthful"]:
        reasons.append("runtime_shapes_not_truthful")
    if not all(row["perfect"] for row in known_four.values()):
        reasons.append("known_four_not_perfect")
    if not explicit["both_actual_counterexamples"]:
        reasons.append("explicit_counterexample_execution_failed")
    if not all(row["policy90_pass"] for row in fresh_rows):
        reasons.append("fresh_seed_below_policy90_or_runtime_nonfinite_error")
    if not before["all_match"] or not after["all_match"] or before != after:
        reasons.append("immutable_hash_changed")

    candidate_cost = int(official["candidate_hardsigmoid_k33"]["cost"])
    staged_cost = int(official["staged_argmax_onehot"]["cost"])
    immutable_cost = int(official["immutable_8009_46"]["cost"])
    aggregate_total = sum(int(row["comparison"]["total"]) for row in fresh_rows)
    aggregate_right = sum(int(row["comparison"]["candidate_right"]) for row in fresh_rows)
    aggregate_raw = sum(int(row["comparison"]["raw_bitwise_equal"]) for row in fresh_rows)
    aggregate_sign = sum(int(row["comparison"]["sign_equal"]) for row in fresh_rows)
    result = {
        "status": "POLICY90_ADMISSIBLE" if not reasons else "REJECT_POLICY90",
        "policy90_admissible": not reasons,
        "stage_action": "NONE_DO_NOT_TOUCH_STAGE",
        "reasons": sorted(set(reasons)),
        "policy": {
            "source": "user explicitly authorizes normal candidates at >=90% accuracy",
            "classification": "SOUND-derived fixed-threshold simplification; not private-zero-only guarantee lane",
            "accuracy_threshold_each_seed": POLICY_ACCURACY,
            "fresh_seeds": list(FRESH_SEEDS),
            "fresh_per_seed": FRESH_PER_SEED,
            "known_four_required_perfect": True,
            "runtime_errors_allowed": 0,
            "nonfinite_values_allowed": 0,
            "all_support_exactness_required": False,
            "all_support_counterexamples_must_be_disclosed": True,
        },
        "inputs": {
            "authority_zip": str(AUTHORITY.relative_to(ROOT)),
            "staged": str(STAGED.relative_to(ROOT)),
            "candidate": str(CANDIDATE.relative_to(ROOT)),
            "support190": str(SUPPORT190.relative_to(ROOT)),
            "hashes_before": before,
            "hashes_after": after,
        },
        "structural": structural,
        "official_scores": official,
        "cost_comparison": {
            "immutable_cost": immutable_cost,
            "staged_cost": staged_cost,
            "candidate_cost": candidate_cost,
            "candidate_reduction_vs_immutable": immutable_cost - candidate_cost,
            "candidate_reduction_vs_staged": staged_cost - candidate_cost,
            "projected_gain_vs_immutable": math.log(immutable_cost / candidate_cost),
            "projected_gain_vs_staged": math.log(staged_cost / candidate_cost),
        },
        "candidate_runtime_shape_trace": candidate_shape,
        "staged_runtime_shape_trace": staged_shape,
        "known_color_count_statistics": known_stats,
        "known_four_configs": known_four,
        "fresh_actual_onnx_two_seeds": fresh_rows,
        "fresh_aggregate": {
            "total": aggregate_total,
            "candidate_right": aggregate_right,
            "candidate_accuracy": aggregate_right / aggregate_total,
            "raw_exact_equal_to_staged": aggregate_raw,
            "sign_equal_to_staged": aggregate_sign,
            "runtime_errors": sum(
                int(row["comparison"]["candidate_runtime_errors"]) for row in fresh_rows
            ),
            "nonfinite_values": sum(
                int(row["comparison"]["candidate_nonfinite_values"]) for row in fresh_rows
            ),
        },
        "support190_expected_k33": support_k33,
        "explicit_reachable_counterexamples": explicit,
        "all_support_exact": False,
        "all_support_label": (
            "FALSE: k33 is not algebraically/generator-support exact; POLICY90 only"
        ),
        "root_or_other_stage_modified": False,
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"status={result['status']} known4={all(row['perfect'] for row in known_four.values())} "
        f"fresh={[row['comparison']['candidate_right'] for row in fresh_rows]} "
        f"aggregate={aggregate_right}/{aggregate_total} all_support={result['all_support_exact']}",
        flush=True,
    )
    return 0 if not reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())

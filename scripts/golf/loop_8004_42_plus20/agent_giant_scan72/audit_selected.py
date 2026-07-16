#!/usr/bin/env python3
"""Fail-closed re-audit of the best retained giant/fan-in candidates.

This does not promote or rewrite any model.  It independently checks four ORT
configurations (disabled/default x threads 1/4), known correctness, raw margin,
static structure, official-like cost, and a reproducible task202 generator
counterexample.  Full-support work is attempted only after the cheaper model
passes all earlier gates.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402


SWEEP_PATH = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_sweep_wave30b/audit_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("wave30b_audit", SWEEP_PATH)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)

CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

CANDIDATES = {
    202: {
        "path": ROOT / "scripts/golf/loop_7999_13/lane_a18/candidates/task202_r03.onnx",
        "baseline_cost": 48,
        "candidate_cost": 28,
        "gain": math.log(48 / 28),
        "lineage": "private-zero; retained fresh has seven generator counterexamples",
    },
    70: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent/task070_r001.onnx",
        "baseline_cost": 75,
        "candidate_cost": 64,
        "gain": math.log(75 / 64),
        "lineage": "private-risk latent prune",
    },
    199: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/prune_latent/task199_r001.onnx",
        "baseline_cost": 261,
        "candidate_cost": 241,
        "gain": math.log(261 / 241),
        "lineage": "latent prune",
    },
    333: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep33/shared_sign/task333_r01.onnx",
        "baseline_cost": 423,
        "candidate_cost": 421,
        "gain": math.log(423 / 421),
        "lineage": "exact sign/gauge absorption, changes 36-input contraction to 35",
    },
    328: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task328_r001.onnx",
        "baseline_cost": 558,
        "candidate_cost": 554,
        "gain": math.log(558 / 554),
        "lineage": "exact initializer contraction reuse; retained margin failure",
    },
    379: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task379_r001.onnx",
        "baseline_cost": 1949,
        "candidate_cost": 1947,
        "gain": math.log(1949 / 1947),
        "lineage": "exact duplicated-mode factor; retained fresh truth 4999/5000",
    },
    13: {
        "path": ROOT / "scripts/golf/loop_8004_42_plus20/root_sweep29/reuse_contract/task013_r001.onnx",
        "baseline_cost": 638,
        "candidate_cost": 636,
        "gain": math.log(638 / 636),
        "lineage": "replace T_zero=[1,0] by the exact Qor main diagonal",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def make_session(path: Path, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_stats() -> dict[str, Any]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "skipped": 0,
        "nonfinite_values": 0,
        "near_positive_values": 0,
        "min_positive": None,
        "max_abs_raw": 0.0,
        "output_shapes": [],
        "first_failure": None,
    }


def update_raw(stats: dict[str, Any], raw: np.ndarray) -> None:
    array = np.asarray(raw)
    finite = np.isfinite(array)
    stats["nonfinite_values"] += int(array.size - np.count_nonzero(finite))
    safe = array[finite]
    if safe.size:
        positive = safe[safe > 0]
        stats["near_positive_values"] += int(np.count_nonzero((safe > 0) & (safe < 0.25)))
        if positive.size:
            current = float(positive.min())
            stats["min_positive"] = (
                current
                if stats["min_positive"] is None
                else min(float(stats["min_positive"]), current)
            )
        stats["max_abs_raw"] = max(
            float(stats["max_abs_raw"]), float(np.abs(safe).max(initial=0.0))
        )
    shape = list(array.shape)
    if shape not in stats["output_shapes"]:
        stats["output_shapes"].append(shape)


def audit_known(task: int, path: Path) -> dict[str, Any]:
    examples = scoring.load_examples(task)
    rows: dict[str, Any] = {}
    for disable_all, threads, label in CONFIGS:
        stats = empty_stats()
        try:
            session = make_session(path, disable_all, threads)
        except Exception as exc:  # noqa: BLE001
            stats["session_error"] = f"{type(exc).__name__}: {exc}"
            rows[label] = stats
            continue
        for split in ("train", "test", "arc-gen"):
            for index, example in enumerate(examples[split]):
                benchmark = scoring.convert_to_numpy(example)
                if benchmark is None:
                    stats["skipped"] += 1
                    continue
                try:
                    raw = session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    update_raw(stats, raw)
                    if np.array_equal(raw > 0, benchmark["output"].astype(bool)):
                        stats["right"] += 1
                    else:
                        stats["wrong"] += 1
                        if stats["first_failure"] is None:
                            stats["first_failure"] = {
                                "split": split,
                                "index": index,
                                "different_cells": int(
                                    np.count_nonzero(
                                        (raw > 0) != benchmark["output"].astype(bool)
                                    )
                                ),
                            }
                except Exception as exc:  # noqa: BLE001
                    stats["runtime_errors"] += 1
                    if stats["first_failure"] is None:
                        stats["first_failure"] = {
                            "split": split,
                            "index": index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
        stats["total_convertible"] = stats["right"] + stats["wrong"] + stats["runtime_errors"]
        stats["perfect"] = (
            stats["right"] == stats["total_convertible"]
            and stats["wrong"] == 0
            and stats["runtime_errors"] == 0
            and stats["nonfinite_values"] == 0
            and stats["near_positive_values"] == 0
        )
        rows[label] = stats
    return {"configs": rows, "perfect_all_configs": all(row.get("perfect", False) for row in rows.values())}


def task202_counterexample(path: Path) -> dict[str, Any]:
    """Reproduce the retained r03 counterexample and run it in all configs."""
    seed = 71_418_202
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module("task_855e0971")
    sessions = {
        label: make_session(path, disable_all, threads)
        for disable_all, threads, label in CONFIGS
    }
    valid = attempts = skips = 0
    while valid < 500:
        attempts += 1
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            skips += 1
            continue
        valid += 1
        first_raw = sessions["disable_all_threads1"].run(
            [sessions["disable_all_threads1"].get_outputs()[0].name],
            {sessions["disable_all_threads1"].get_inputs()[0].name: benchmark["input"]},
        )[0]
        if np.array_equal(first_raw > 0, benchmark["output"].astype(bool)):
            continue
        configs: dict[str, Any] = {}
        for label, session in sessions.items():
            raw = session.run(
                [session.get_outputs()[0].name],
                {session.get_inputs()[0].name: benchmark["input"]},
            )[0]
            configs[label] = {
                "correct": bool(np.array_equal(raw > 0, benchmark["output"].astype(bool))),
                "different_cells": int(
                    np.count_nonzero((raw > 0) != benchmark["output"].astype(bool))
                ),
                "nonfinite_values": int(np.count_nonzero(~np.isfinite(raw))),
                "near_positive_values": int(np.count_nonzero((raw > 0) & (raw < 0.25))),
            }
        return {
            "seed": seed,
            "valid_case": valid,
            "attempt": attempts,
            "conversion_skips_before_failure": skips,
            "input_grid": example["input"],
            "expected_grid": example["output"],
            "configs": configs,
            "fails_all_configs": all(not row["correct"] for row in configs.values()),
        }
    return {"seed": seed, "searched_valid": valid, "counterexample_found": False}


def task013_diagonal_proof(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    diagonal = np.asarray([arrays["Qor"][(i, i, i, i, i)] for i in range(2)], dtype=np.float32)
    width_sum = sum(width // 2 for width in range(20, 31))
    structural_states = width_sum * 7 * 5 * 4 * 2
    color_orderings = 9 * 8
    return {
        "qor_main_diagonal": diagonal.tolist(),
        "removed_initializer": [1.0, 0.0],
        "exact_tensor_equal": bool(np.array_equal(diagonal, np.asarray([1.0, 0.0], dtype=np.float32))),
        "generator_structural_states": structural_states,
        "ordered_distinct_color_pairs": color_orderings,
        "full_generator_support_count": structural_states * color_orderings,
        "note": "Operator-level real semantics are identical, but the 51-input floating contraction plan changed; all-support execution in all four ORT configs was not completed, so the requested platform guarantee is fail-closed.",
    }


def runtime_shape_trace_sanitized(task: int, path: Path) -> dict[str, Any]:
    """Trace all node outputs while preserving output order across sanitization."""
    model = onnx.load(path)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    del traced.graph.output[:]
    expected: list[tuple[str, list[int]]] = []
    for node in traced.graph.node:
        for name in node.output:
            value = typed.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            dims = [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]
            traced.graph.output.append(copy.deepcopy(value))
            expected.append((name, dims))
    traced = scoring.sanitize_model(traced)
    if traced is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    session = ort.InferenceSession(
        traced.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    benchmark = scoring.convert_to_numpy(scoring.load_examples(task)["train"][0])
    assert benchmark is not None
    arrays = session.run(None, {session.get_inputs()[0].name: benchmark["input"]})
    mismatches = [
        {"name": name, "declared": dims, "actual": list(np.asarray(array).shape)}
        for (name, dims), array in zip(expected, arrays)
        if dims != list(np.asarray(array).shape)
    ]
    return {
        "traced": len(expected),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "truthful": not mismatches,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    rows: dict[str, Any] = {}
    # Only candidates that can still teach us something are rerun.  The others
    # have retained terminal evidence (margin failure or fresh counterexample).
    rerun_known = {202, 70, 199, 333, 13}
    for task, meta in CANDIDATES.items():
        path = meta["path"]
        data = path.read_bytes()
        row: dict[str, Any] = {
            "task": task,
            "path": relative(path),
            "sha256": sha256(path),
            "baseline_cost": meta["baseline_cost"],
            "candidate_cost": meta["candidate_cost"],
            "projected_gain": meta["gain"],
            "lineage": meta["lineage"],
            "static": SWEEP.static_audit(data),
        }
        if task in rerun_known:
            print(f"task{task:03d} known x4", flush=True)
            row["known_four_configs"] = audit_known(task, path)
        if task in {202, 70, 199, 333, 13}:
            try:
                row["runtime_shape_trace"] = SWEEP.runtime_shape_trace(task, data)
            except Exception as exc:  # noqa: BLE001
                row["runtime_shape_trace"] = {
                    "truthful": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        rows[str(task)] = row
        (HERE / "audit_checkpoint.json").write_text(json.dumps(rows, indent=2) + "\n")

    print("task202 counterexample x4", flush=True)
    rows["202"]["generator_counterexample"] = task202_counterexample(CANDIDATES[202]["path"])
    rows["13"]["finite_support_proof"] = task013_diagonal_proof(CANDIDATES[13]["path"])

    # Retained terminal evidence is hash-matched and intentionally not rerun
    # after the earlier gate already makes adoption impossible.
    rows["328"]["retained_terminal_evidence"] = {
        "source": "scripts/golf/loop_7999_13/lane_b26/winner_manifest.json",
        "known_dual": "267/267 both modes, errors0",
        "fresh16_dual": "16/16 both modes, errors0",
        "near_positive_each_mode": 4,
        "min_positive": 7.316870026530253e-11,
        "terminal_reason": "raw margin violation",
    }
    rows["379"]["retained_terminal_evidence"] = {
        "source": "scripts/golf/loop_8002_63/LOOP_STATUS.md",
        "known_raw_equal": "266/266",
        "fresh_raw_equal": "5000/5000 both modes",
        "fresh_truth": "4999/5000",
        "terminal_reason": "observed legal generator counterexample; no 100% pass guarantee",
    }

    for task, row in rows.items():
        reasons: list[str] = []
        known = row.get("known_four_configs")
        if known is not None and not known["perfect_all_configs"]:
            reasons.append("known_or_raw_margin_gate_failed_in_at_least_one_config")
        if task == "202":
            reasons.append("generator_counterexample_reproduced_in_all_four_configs")
        if task == "328":
            reasons.append("raw_margin_gate_failed")
        if task == "379":
            reasons.append("generator_truth_is_not_100_percent")
        if task in {"333", "13"}:
            reasons.append("all_generator_support_x_four_ort_configs_not_proven")
        if task in {"70", "199"}:
            reasons.append("strict_lower_candidate_is_known_incorrect")
        row["accepted"] = False
        row["rejection_reasons"] = sorted(set(reasons))

    output = {
        "baseline": "submission_base_8005.16.zip",
        "baseline_sha256": hashlib.sha256((ROOT / "submission_base_8005.16.zip").read_bytes()).hexdigest(),
        "cross_scan": {
            "report_result_winner_files_seen": 1136,
            "wave30b_candidate_files": 41,
            "tasks_considered": [13, 51, 64, 70, 199, 202, 328, 333, 379],
            "excluded_dedicated_lanes": [198, 254, 267, 323],
            "task051": "current cost279; factor candidate cost280 is not strictly lower",
            "task064": "no retained known-perfect candidate below current cost271",
        },
        "rows": rows,
        "accepted": [],
        "aggregate_gain": 0.0,
        "zip_modified": False,
        "protected_files_modified": False,
        "verdict": "NO_GUARANTEED_WINNER",
    }
    (HERE / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

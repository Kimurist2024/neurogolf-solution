#!/usr/bin/env python3
"""Independent four-configuration raw/truth audit for task158."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / "others/71407/task158.onnx"
CANDIDATE = (
    ROOT
    / "scripts/golf/loop_8004_42_plus20/agent_task158_residual_215"
    / "candidates/task158_exact_anchor_role_bits.onnx"
)
PARENT_SHA = "127984c6807d84559bbf74fd58e3b09a66459d142cef65a8635647e64f5e59fd"
CANDIDATE_SHA = "e7101699bfc022fa794e15d7f374a8febe3e2680b8388c67b9a81cdc9962ced0"
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (15_821_901, 15_821_902)
FRESH_PER_SEED = 1_500

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected payload")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_metrics(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "attempts": 0,
        "valid": 0,
        "conversion_skips": 0,
        "parent_right": 0,
        "candidate_right": 0,
        "raw_bitwise_equal": 0,
        "threshold_equal": 0,
        "runtime_errors": {"parent": 0, "candidate": 0},
        "nonfinite_values": {"parent": 0, "candidate": 0},
        "candidate_near_positive_0_to_0_25": 0,
        "candidate_min_positive": None,
        "first_failure": None,
    }


def consume(
    metrics: dict[str, Any],
    parent_session: ort.InferenceSession,
    candidate_session: ort.InferenceSession,
    example: dict[str, Any],
    index: int,
) -> None:
    metrics["attempts"] += 1
    benchmark = scoring.convert_to_numpy(example)
    if benchmark is None:
        metrics["conversion_skips"] += 1
        return
    metrics["valid"] += 1
    expected = benchmark["output"].astype(bool)
    outputs: dict[str, np.ndarray] = {}
    for name, session in (("parent", parent_session), ("candidate", candidate_session)):
        try:
            value = np.asarray(
                session.run(
                    [session.get_outputs()[0].name],
                    {session.get_inputs()[0].name: benchmark["input"]},
                )[0]
            )
        except Exception as exc:  # noqa: BLE001
            metrics["runtime_errors"][name] += 1
            metrics["first_failure"] = metrics["first_failure"] or {
                "index": index,
                "model": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        outputs[name] = value
        finite_count = int(np.count_nonzero(np.isfinite(value)))
        metrics["nonfinite_values"][name] += int(value.size - finite_count)
        metrics[f"{name}_right"] += int(
            value.shape == expected.shape and np.array_equal(value > 0, expected)
        )
        if name == "candidate":
            positive = value[value > 0]
            if positive.size:
                minimum = float(positive.min())
                old = metrics["candidate_min_positive"]
                metrics["candidate_min_positive"] = minimum if old is None else min(old, minimum)
                metrics["candidate_near_positive_0_to_0_25"] += int(np.count_nonzero(positive < 0.25))
    if len(outputs) != 2:
        return
    left, right = outputs["parent"], outputs["candidate"]
    raw_equal = bool(
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )
    threshold_equal = bool(np.array_equal(left > 0, right > 0))
    metrics["raw_bitwise_equal"] += int(raw_equal)
    metrics["threshold_equal"] += int(threshold_equal)
    if not raw_equal and metrics["first_failure"] is None:
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        metrics["first_failure"] = {
            "index": index,
            "comparison": "candidate_vs_parent_raw",
            "max_abs_delta": float(np.nanmax(delta)),
        }


def finalize(metrics: dict[str, Any]) -> dict[str, Any]:
    valid = int(metrics["valid"])
    errors = sum(int(value) for value in metrics["runtime_errors"].values())
    nonfinite = sum(int(value) for value in metrics["nonfinite_values"].values())
    metrics["runtime_errors_total"] = errors
    metrics["nonfinite_values_total"] = nonfinite
    metrics["raw_equivalent"] = bool(
        valid > 0
        and metrics["raw_bitwise_equal"] == valid
        and metrics["threshold_equal"] == valid
    )
    metrics["truth_correct"] = bool(
        valid > 0
        and metrics["parent_right"] == valid
        and metrics["candidate_right"] == valid
    )
    metrics["pass"] = bool(
        metrics["raw_equivalent"]
        and metrics["truth_correct"]
        and errors == 0
        and nonfinite == 0
        and metrics["candidate_near_positive_0_to_0_25"] == 0
        and metrics["conversion_skips"] == 0
    )
    return metrics


def run_known(
    parent_data: bytes,
    candidate_data: bytes,
    disable_all: bool,
    threads: int,
    label: str,
) -> dict[str, Any]:
    parent = make_session(parent_data, disable_all, threads)
    candidate = make_session(candidate_data, disable_all, threads)
    result = empty_metrics(label)
    examples = scoring.load_examples(158)
    index = 0
    for split in ("train", "test", "arc-gen"):
        for example in examples.get(split, []):
            consume(result, parent, candidate, example, index)
            index += 1
    return finalize(result)


def run_fresh(
    parent_data: bytes,
    candidate_data: bytes,
    disable_all: bool,
    threads: int,
    label: str,
    seed: int,
) -> dict[str, Any]:
    parent = make_session(parent_data, disable_all, threads)
    candidate = make_session(candidate_data, disable_all, threads)
    module = importlib.import_module("task_6aa20dc0")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    result = empty_metrics(label)
    while result["valid"] < FRESH_PER_SEED and result["attempts"] < FRESH_PER_SEED * 20:
        example = module.generate()
        consume(result, parent, candidate, example, int(result["attempts"]))
    return finalize(result)


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "evidence").mkdir(exist_ok=True)
    hashes = {"parent": digest(PARENT), "candidate": digest(CANDIDATE)}
    if hashes != {"parent": PARENT_SHA, "candidate": CANDIDATE_SHA}:
        raise RuntimeError(f"payload hash mismatch: {hashes}")
    parent_data = PARENT.read_bytes()
    candidate_data = CANDIDATE.read_bytes()
    report: dict[str, Any] = {
        "task": 158,
        "hashes": hashes,
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_per_seed": FRESH_PER_SEED,
        "known": {},
        "fresh": {},
    }
    output = HERE / "evidence/runtime_four_config.json"
    for disable_all, threads, label in CONFIGS:
        row = run_known(parent_data, candidate_data, disable_all, threads, label)
        report["known"][label] = row
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"known {label}: pass={row['pass']} valid={row['valid']}", flush=True)
    for seed in FRESH_SEEDS:
        report["fresh"][str(seed)] = {}
        for disable_all, threads, label in CONFIGS:
            row = run_fresh(
                parent_data, candidate_data, disable_all, threads, label, seed
            )
            report["fresh"][str(seed)][label] = row
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"fresh seed={seed} {label}: pass={row['pass']} valid={row['valid']}",
                flush=True,
            )
    rows = list(report["known"].values()) + [
        row for modes in report["fresh"].values() for row in modes.values()
    ]
    report["summary"] = {
        "known_cases_per_config": min(int(row["valid"]) for row in report["known"].values()),
        "known_raw_comparisons": sum(int(row["valid"]) for row in report["known"].values()),
        "fresh_cases_per_seed_per_config": FRESH_PER_SEED,
        "fresh_raw_comparisons": sum(
            int(row["valid"]) for modes in report["fresh"].values() for row in modes.values()
        ),
        "raw_mismatches_total": sum(int(row["valid"] - row["raw_bitwise_equal"]) for row in rows),
        "parent_truth_mismatches_total": sum(int(row["valid"] - row["parent_right"]) for row in rows),
        "candidate_truth_mismatches_total": sum(int(row["valid"] - row["candidate_right"]) for row in rows),
        "runtime_errors_total": sum(int(row["runtime_errors_total"]) for row in rows),
        "nonfinite_values_total": sum(int(row["nonfinite_values_total"]) for row in rows),
        "near_positive_total": sum(int(row["candidate_near_positive_0_to_0_25"]) for row in rows),
        "all_pass": all(bool(row["pass"]) for row in rows),
    }
    report["pass"] = report["summary"]["all_pass"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

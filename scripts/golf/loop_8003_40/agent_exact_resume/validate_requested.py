#!/usr/bin/env python3
"""Independent dual-ORT known/fresh audit for the requested exact candidates."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402
from build_and_scan import sha256_bytes, sha256_path, strict_gate  # noqa: E402


BASELINE = ROOT / "submission_base_8003.40.zip"
TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
CANDIDATES = {
    48: HERE / "candidates/task048_requested.onnx",
    233: HERE / "candidates/task233_requested.onnx",
    333: HERE / "candidates/task333_requested.onnx",
}

# These three requested probes are evidence-only.  The current safety policy
# rejects their byte changes even if a sampled differential is perfect:
# task048/task333 change floating Einsum contraction structure, while task233
# is a highest-risk task with a dust-sized gain.
POLICY_REJECT = {48, 233, 333}


def make_session(model: onnx.ModelProto, disable_all: bool) -> onnxruntime.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable_all
        else onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    return onnxruntime.InferenceSession(sanitized.SerializeToString(), options)


def converted_known(task: int) -> list[dict[str, np.ndarray]]:
    examples = scoring.load_examples(task)
    rows = []
    for subset in ("train", "test", "arc-gen"):
        for example in examples.get(subset, []):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                rows.append(converted)
    return rows


def converted_fresh(task: int, count: int, seed: int) -> tuple[list[dict[str, np.ndarray]], int, int]:
    module = importlib.import_module(f"task_{TASK_MAP[f'{task:03d}']}")
    random.seed(seed)
    rows = []
    generation_errors = 0
    attempts = 0
    while len(rows) < count and attempts < count * 10:
        attempts += 1
        try:
            converted = scoring.convert_to_numpy(module.generate())
            if converted is not None:
                rows.append(converted)
        except Exception:  # noqa: BLE001
            generation_errors += 1
    return rows, generation_errors, attempts


def compare(
    baseline_session: onnxruntime.InferenceSession,
    candidate_session: onnxruntime.InferenceSession,
    examples: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    row: dict[str, object] = {
        "total": len(examples),
        "baseline_right": 0,
        "candidate_right": 0,
        "baseline_runtime_errors": 0,
        "candidate_runtime_errors": 0,
        "one_sided_runtime_errors": 0,
        "raw_bitwise_equal": 0,
        "decoded_equal": 0,
        "max_abs_raw_difference": 0.0,
    }
    for example in examples:
        expected = example["output"] > 0
        baseline_raw = candidate_raw = None
        try:
            baseline_raw = baseline_session.run(["output"], {"input": example["input"]})[0]
        except Exception:  # noqa: BLE001
            row["baseline_runtime_errors"] += 1
        try:
            candidate_raw = candidate_session.run(["output"], {"input": example["input"]})[0]
        except Exception:  # noqa: BLE001
            row["candidate_runtime_errors"] += 1
        if (baseline_raw is None) != (candidate_raw is None):
            row["one_sided_runtime_errors"] += 1
        if baseline_raw is None or candidate_raw is None:
            continue
        baseline_decoded = baseline_raw > 0
        candidate_decoded = candidate_raw > 0
        row["baseline_right"] += int(np.array_equal(baseline_decoded, expected))
        row["candidate_right"] += int(np.array_equal(candidate_decoded, expected))
        row["raw_bitwise_equal"] += int(
            np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
        )
        row["decoded_equal"] += int(np.array_equal(baseline_decoded, candidate_decoded))
        baseline_numeric = np.nan_to_num(baseline_raw).astype(np.float64, copy=False)
        candidate_numeric = np.nan_to_num(candidate_raw).astype(np.float64, copy=False)
        difference = np.abs(baseline_numeric - candidate_numeric)
        row["max_abs_raw_difference"] = max(
            float(row["max_abs_raw_difference"]), float(difference.max(initial=0.0))
        )
    executable = int(row["total"]) - int(row["candidate_runtime_errors"])
    row["candidate_accuracy"] = (
        int(row["candidate_right"]) / executable if executable > 0 else None
    )
    return row


def scored(model: onnx.ModelProto, task: int, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory(prefix=f"exact_resume_{task}_{label}_") as directory:
        return scoring.score_and_verify(
            model, task, directory, label=label, require_correct=True
        )


def audit(task: int, path: Path, count: int) -> dict[str, object]:
    with zipfile.ZipFile(BASELINE) as archive:
        baseline_bytes = archive.read(f"task{task:03d}.onnx")
    baseline = onnx.load_model_from_string(baseline_bytes)
    candidate = onnx.load(path)
    known = converted_known(task)
    fresh, generation_errors, attempts = converted_fresh(
        task, count, seed=8_003_400 + task
    )
    report: dict[str, object] = {
        "task": task,
        "baseline": str(BASELINE.relative_to(ROOT)),
        "baseline_member_sha256": sha256_bytes(baseline_bytes),
        "candidate": str(path.relative_to(ROOT)),
        "candidate_sha256": sha256_path(path),
        "structural_gate": strict_gate(candidate),
        "known_count": len(known),
        "fresh_requested": count,
        "fresh_executable": len(fresh),
        "fresh_generation_errors": generation_errors,
        "fresh_generation_attempts": attempts,
        "modes": {},
    }
    for disable_all, label in ((True, "disable_all"), (False, "default")):
        baseline_session = make_session(baseline, disable_all)
        candidate_session = make_session(candidate, disable_all)
        report["modes"][label] = {
            "known": compare(baseline_session, candidate_session, known),
            "fresh": compare(baseline_session, candidate_session, fresh),
        }

    report["baseline_score"] = scored(baseline, task, "baseline")
    report["candidate_score"] = scored(candidate, task, "candidate")
    margin_stable, margin_min = scoring.model_margin_stable(candidate, task)
    report["margin_stable"] = bool(margin_stable)
    report["margin_min"] = margin_min

    baseline_cost = report["baseline_score"]["cost"] if report["baseline_score"] else None
    candidate_cost = report["candidate_score"]["cost"] if report["candidate_score"] else None
    if baseline_cost is not None and candidate_cost is not None:
        report["cost_reduction"] = baseline_cost - candidate_cost
        report["projected_gain"] = math.log(baseline_cost / candidate_cost)
    else:
        report["cost_reduction"] = None
        report["projected_gain"] = None

    exact = bool(report["structural_gate"].get("pass"))
    for mode in report["modes"].values():
        for subset in ("known", "fresh"):
            row = mode[subset]
            exact &= (
                row["candidate_runtime_errors"] == 0
                and row["one_sided_runtime_errors"] == 0
                and row["raw_bitwise_equal"] == row["total"]
                and row["decoded_equal"] == row["total"]
                and row["candidate_accuracy"] is not None
                and row["candidate_accuracy"] >= 0.95
            )
    exact &= bool(report["margin_stable"])
    exact &= (
        report["cost_reduction"] is not None and int(report["cost_reduction"]) > 0
    )
    report["sampled_exact_gate"] = exact
    report["policy_rejected"] = task in POLICY_REJECT
    report["verdict"] = (
        "ACCEPT_EXACT_AUDIT" if exact and task not in POLICY_REJECT else "REJECT"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    summary = []
    for task, candidate in CANDIDATES.items():
        print(f"auditing task{task:03d} ({args.count} fresh)", flush=True)
        report = audit(task, candidate, args.count)
        output = HERE / f"audit_task{task:03d}.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary.append(
            {
                "task": task,
                "candidate_sha256": report["candidate_sha256"],
                "cost_reduction": report["cost_reduction"],
                "projected_gain": report["projected_gain"],
                "verdict": report["verdict"],
                "output": str(output.relative_to(ROOT)),
            }
        )
        print(json.dumps(summary[-1]), flush=True)
    (HERE / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if all(row["verdict"] == "ACCEPT_EXACT_AUDIT" for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())

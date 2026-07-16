#!/usr/bin/env python3
"""Fail-closed known/fresh/raw-equivalence audit for exact Selu candidates."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = Path("/private/tmp/ng800946_rank")
CANDIDATES = HERE / "candidates"
TASKS = (13, 90, 134, 209, 233, 366)
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (127_000_001, 127_000_002)
FRESH_PER_SEED = 1500
FRESH_COUNTS = {13: 250, 90: 500, 134: 1500, 209: 1500, 233: 500, 366: 1500}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text(encoding="utf-8"))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(data: bytes, disable: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model returned None")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def evaluate_cases(
    baseline: bytes,
    candidate: bytes,
    cases: list[dict[str, Any]],
    disable: bool,
    threads: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "total": len(cases),
        "valid": 0,
        "conversion_skips": 0,
        "candidate_right": 0,
        "baseline_right": 0,
        "raw_equal": 0,
        "threshold_equal": 0,
        "runtime_errors": {"candidate": 0, "baseline": 0},
        "nonfinite_values": {"candidate": 0, "baseline": 0},
        "first_failure": None,
    }
    try:
        sessions = {
            "candidate": make_session(candidate, disable, threads),
            "baseline": make_session(baseline, disable, threads),
        }
    except Exception as exc:  # noqa: BLE001
        row["session_error"] = f"{type(exc).__name__}: {exc}"
        row["perfect"] = False
        return row
    for index, example in enumerate(cases):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            row["conversion_skips"] += 1
            continue
        row["valid"] += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                value = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"][label] += 1
                row["first_failure"] = row["first_failure"] or {
                    "index": index,
                    "model": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            outputs[label] = value
            row["nonfinite_values"][label] += int(value.size - np.count_nonzero(np.isfinite(value)))
            row[f"{label}_right"] += int(value.shape == expected.shape and np.array_equal(value > 0, expected))
        if len(outputs) == 2:
            raw_equal = bool(
                outputs["candidate"].dtype == outputs["baseline"].dtype
                and outputs["candidate"].shape == outputs["baseline"].shape
                and np.array_equal(
                    np.ascontiguousarray(outputs["candidate"]).view(np.uint8),
                    np.ascontiguousarray(outputs["baseline"]).view(np.uint8),
                )
            )
            threshold_equal = np.array_equal(outputs["candidate"] > 0, outputs["baseline"] > 0)
            row["raw_equal"] += int(raw_equal)
            row["threshold_equal"] += int(threshold_equal)
            if not raw_equal:
                delta = np.abs(outputs["candidate"].astype(np.float64) - outputs["baseline"].astype(np.float64))
                row["first_failure"] = row["first_failure"] or {
                    "index": index,
                    "comparison": "candidate_vs_authority",
                    "max_abs_delta": float(np.nanmax(delta)),
                }
    total = row["valid"]
    row["runtime_errors_total"] = sum(row["runtime_errors"].values())
    row["nonfinite_values_total"] = sum(row["nonfinite_values"].values())
    row["candidate_accuracy"] = row["candidate_right"] / total if total else None
    row["baseline_accuracy"] = row["baseline_right"] / total if total else None
    row["exact_equivalent"] = bool(
        total > 0
        and row["raw_equal"] == total
        and row["threshold_equal"] == total
        and row["runtime_errors_total"] == 0
        and row["nonfinite_values"]["candidate"] == row["nonfinite_values"]["baseline"]
    )
    row["perfect_truth"] = bool(
        row["exact_equivalent"]
        and row["candidate_right"] == total
        and row["baseline_right"] == total
    )
    return row


def known(task: int) -> list[dict[str, Any]]:
    payload = scoring.load_examples(task)
    return [
        item
        for split in ("train", "test", "arc-gen")
        for item in payload.get(split, [])
    ]


def generate(task: int, seed: int, count: int) -> tuple[list[dict[str, Any]], int]:
    module = importlib.import_module(f"task_{TASK_MAP[f'{task:03d}']}")
    common = importlib.import_module("common")
    rows: list[dict[str, Any]] = []
    attempts = 0
    random.seed(seed)
    common.random.seed(seed)
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        example = module.generate()
        if scoring.convert_to_numpy(example) is not None:
            rows.append(example)
    return rows, attempts


def dims(value: onnx.ValueInfoProto) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def runtime_shape_truth(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
    typed = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    traced = copy.deepcopy(model)
    names: list[str] = []
    existing = {value.name for value in traced.graph.output}
    for node in traced.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
                if name not in existing:
                    traced.graph.output.append(copy.deepcopy(typed[name]))
                    existing.add(name)
    benchmark = scoring.convert_to_numpy(known(task)[0])
    if benchmark is None:
        return {"truthful": False, "error": "known[0] conversion failed"}
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 4
    try:
        session = ort.InferenceSession(traced.SerializeToString(), options, providers=["CPUExecutionProvider"])
        arrays = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}
    mismatches = []
    nonfinite = 0
    for name, array in zip(names, arrays):
        value = np.asarray(array)
        declared = dims(typed[name])
        actual = list(value.shape)
        if declared != actual:
            mismatches.append({"name": name, "declared": declared, "actual": actual})
        if value.dtype.kind in "fc":
            nonfinite += int(value.size - np.count_nonzero(np.isfinite(value)))
    return {
        "traced": len(names),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_values": nonfinite,
        "truthful": not mismatches and nonfinite == 0,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    build = json.loads((HERE / "build.json").read_text(encoding="utf-8"))
    build_rows = {int(row["task"]): row for row in build["rows"]}
    report: dict[str, Any] = {"tasks": {}, "winners": []}
    for task in TASKS:
        base = (AUTHORITY / f"task{task:03d}.onnx").read_bytes()
        candidate = (CANDIDATES / f"task{task:03d}.onnx").read_bytes()
        row: dict[str, Any] = {
            "hash": TASK_MAP[f"{task:03d}"],
            "authority_sha256": digest(base),
            "candidate_sha256": digest(candidate),
            "build": build_rows[task],
            "runtime_shape_truth": runtime_shape_truth(task, candidate),
            "known_four_configs": {},
            "fresh": [],
            "reasons": [],
            "warnings": [],
            "accepted": False,
        }
        known_cases = known(task)
        for disable, threads, label in CONFIGS:
            row["known_four_configs"][label] = evaluate_cases(
                base, candidate, known_cases, disable, threads
            )
        if not build_rows[task]["candidate_structure"]["pass"]:
            row["reasons"].append("static_structure_gate")
        if not build_rows[task]["strict_lower"]:
            row["reasons"].append("not_strict_lower")
        if not row["runtime_shape_truth"].get("truthful", False):
            row["warnings"].append("inherited_authority_runtime_shape_cloak")
        if not all(item.get("exact_equivalent", False) for item in row["known_four_configs"].values()):
            row["reasons"].append("known_or_raw_equivalence_failed")
        if not row["reasons"]:
            for seed in FRESH_SEEDS:
                fresh_cases, attempts = generate(task, seed + task * 10_000, FRESH_COUNTS[task])
                stream = {"seed": seed + task * 10_000, "attempts": attempts, "modes": {}}
                for disable, label in ((True, "disable_all"), (False, "default")):
                    stream["modes"][label] = evaluate_cases(base, candidate, fresh_cases, disable, 1)
                row["fresh"].append(stream)
                print(f"task{task:03d} fresh seed={stream['seed']} valid={len(fresh_cases)}", flush=True)
            if not all(
                mode.get("exact_equivalent", False)
                for stream in row["fresh"]
                for mode in stream["modes"].values()
            ):
                row["reasons"].append("fresh_or_raw_equivalence_failed")
            minimum_accuracy = min(
                mode.get("candidate_accuracy") or 0.0
                for stream in row["fresh"]
                for mode in stream["modes"].values()
            )
            row["fresh_minimum_accuracy"] = minimum_accuracy
            if minimum_accuracy < 0.90:
                row["reasons"].append("fresh_accuracy_below_user_policy90")
        row["accepted"] = not row["reasons"]
        if row["accepted"]:
            report["winners"].append(task)
        report["tasks"][str(task)] = row
        (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"task{task:03d} accepted={row['accepted']} reasons={row['reasons']}", flush=True)
    report["summary"] = {
        "tasks": len(TASKS),
        "winners": report["winners"],
        "winner_count": len(report["winners"]),
        "fresh_per_seed": FRESH_PER_SEED,
        "fresh_counts_per_seed": {str(task): FRESH_COUNTS[task] for task in TASKS},
        "fresh_seeds": list(FRESH_SEEDS),
    }
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

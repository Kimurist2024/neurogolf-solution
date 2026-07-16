#!/usr/bin/env python3
"""Raw pass-through and truthful-shape audit for surviving domain identities."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
OUTPUT = HERE / "audit.json"
TASKS = (264, 377, 388)
FRESH_SEEDS = (241_000_001, 241_000_002)
FRESH_CASES = 2000
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from lib import scoring  # noqa: E402


def session(data: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )


def known_cases(task: int) -> list[dict[str, np.ndarray]]:
    result = []
    examples = scoring.load_examples(task)
    for subset in ("train", "test", "arc-gen"):
        for example in examples[subset]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                result.append(benchmark)
    return result


def compare_cases(
    authority: bytes,
    candidate: bytes,
    cases: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    output = {}
    for disabled, threads, name in CONFIGS:
        stats: dict[str, Any] = {
            "cases": len(cases),
            "raw_equal": 0,
            "threshold_equal": 0,
            "candidate_correct": 0,
            "authority_errors": 0,
            "candidate_errors": 0,
            "candidate_nonfinite": 0,
            "first_failure": None,
        }
        try:
            authority_session = session(authority, disabled, threads)
            candidate_session = session(candidate, disabled, threads)
        except Exception as exc:  # noqa: BLE001
            stats["session_error"] = f"{type(exc).__name__}: {exc}"
            output[name] = stats
            continue
        for case_index, benchmark in enumerate(cases):
            feed_a = {authority_session.get_inputs()[0].name: benchmark["input"]}
            feed_c = {candidate_session.get_inputs()[0].name: benchmark["input"]}
            try:
                raw_a = authority_session.run([authority_session.get_outputs()[0].name], feed_a)[0]
            except Exception as exc:  # noqa: BLE001
                stats["authority_errors"] += 1
                if stats["first_failure"] is None:
                    stats["first_failure"] = {"case": case_index, "authority_error": f"{type(exc).__name__}: {exc}"}
                continue
            try:
                raw_c = candidate_session.run([candidate_session.get_outputs()[0].name], feed_c)[0]
            except Exception as exc:  # noqa: BLE001
                stats["candidate_errors"] += 1
                if stats["first_failure"] is None:
                    stats["first_failure"] = {"case": case_index, "candidate_error": f"{type(exc).__name__}: {exc}"}
                continue
            equal = raw_equal(raw_a, raw_c)
            threshold = np.array_equal(raw_a > 0, raw_c > 0)
            correct = np.array_equal(raw_c > 0, benchmark["output"] > 0)
            stats["raw_equal"] += int(equal)
            stats["threshold_equal"] += int(threshold)
            stats["candidate_correct"] += int(correct)
            stats["candidate_nonfinite"] += int(
                np.issubdtype(raw_c.dtype, np.floating) and not np.isfinite(raw_c).all()
            )
            if stats["first_failure"] is None and not (equal and threshold and correct):
                stats["first_failure"] = {
                    "case": case_index,
                    "raw_equal": equal,
                    "threshold_equal": threshold,
                    "candidate_correct": correct,
                }
        output[name] = stats
    return output


def shape_of(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def truthful_shapes(data: bytes, benchmark: dict[str, np.ndarray]) -> dict[str, Any]:
    try:
        model = onnx.load_model_from_string(data)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        typed = {
            value.name: value
            for value in list(inferred.graph.value_info) + list(inferred.graph.output)
            if value.type.HasField("tensor_type")
            and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)
        }
        names = []
        for node in inferred.graph.node:
            for name in node.output:
                if name and name in typed and name not in names:
                    names.append(name)
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        runtime = session(exposed.SerializeToString(), True, 1)
        values = runtime.run(names, {runtime.get_inputs()[0].name: benchmark["input"]})
        mismatches = [
            {"tensor": name, "declared": shape_of(typed[name]), "runtime": list(value.shape)}
            for name, value in zip(names, values)
            if shape_of(typed[name]) != list(value.shape)
        ]
        return {
            "traced_outputs": len(names),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "truthful": not mismatches,
        }
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def fresh_cases(task: int, seed: int, count: int) -> list[dict[str, np.ndarray]]:
    mapping = json.loads(TASK_MAP.read_text())
    generator = importlib.import_module(f"task_{mapping[f'{task:03d}']}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    result = []
    attempts = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError(f"generator stalled at {len(result)}/{count}")
        benchmark = scoring.convert_to_numpy(generator.generate())
        if benchmark is not None:
            result.append(benchmark)
    return result


def perfect(report: dict[str, Any]) -> bool:
    return all(
        not item.get("session_error")
        and item["raw_equal"] == item["cases"]
        and item["threshold_equal"] == item["cases"]
        and item["candidate_correct"] == item["cases"]
        and item["authority_errors"] == 0
        and item["candidate_errors"] == 0
        and item["candidate_nonfinite"] == 0
        for item in report.values()
    )


def main() -> None:
    output = {"fresh_seeds": list(FRESH_SEEDS), "fresh_cases_per_seed": FRESH_CASES, "tasks": []}
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            authority = archive.read(f"task{task:03d}.onnx")
            path = HERE / "candidates" / f"task{task:03d}_domain_identity.onnx"
            candidate = path.read_bytes()
            known = known_cases(task)
            known_report = compare_cases(authority, candidate, known)
            row: dict[str, Any] = {
                "task": task,
                "known_case_count": len(known),
                "known_four_config": known_report,
                "authority_shapes": truthful_shapes(authority, known[0]),
                "candidate_shapes": truthful_shapes(candidate, known[0]),
                "fresh": [],
            }
            if perfect(known_report) and row["candidate_shapes"].get("truthful"):
                for seed in FRESH_SEEDS:
                    cases = fresh_cases(task, seed + task, FRESH_CASES)
                    row["fresh"].append({
                        "seed": seed + task,
                        "four_config": compare_cases(authority, candidate, cases),
                    })
            row["accepted"] = (
                perfect(known_report)
                and row["candidate_shapes"].get("truthful") is True
                and len(row["fresh"]) == len(FRESH_SEEDS)
                and all(perfect(item["four_config"]) for item in row["fresh"])
            )
            output["tasks"].append(row)
            print(json.dumps({
                "task": task,
                "known_perfect": perfect(known_report),
                "truthful": row["candidate_shapes"].get("truthful"),
                "fresh_seeds_run": len(row["fresh"]),
                "accepted": row["accepted"],
            }))
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed cost-26..50 search against the immutable 8011.05 authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
SCORES = ROOT / "all_scores.csv"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
CANDIDATES = HERE / "candidates"
EVIDENCE = HERE / "evidence.json"
EXPECTED_IO = [1, 10, 30, 30]
FRESH_PER_SEED = 2_000


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("cost26_50_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


SUPPORT = load_support()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def score_cost(cost: int) -> float:
    return 25.0 - float(np.log(max(1, cost)))


def load_costs() -> dict[int, int]:
    result = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(",")
        if len(fields) >= 4:
            result[int(fields[1][4:])] = int(fields[3])
    return result


def make_output_only(
    node: onnx.NodeProto,
    initializers: list[onnx.TensorProto] | None = None,
    *,
    output_dtype: int = TensorProto.FLOAT,
    opset: int = 21,
    name: str,
) -> onnx.ModelProto:
    graph = helper.make_graph(
        [node], name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, EXPECTED_IO)],
        [helper.make_tensor_value_info("output", output_dtype, EXPECTED_IO)],
        initializer=initializers or [],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="cost26_50_search_297",
    )
    model.ir_version = 10
    return model


def reverse_sequence_candidate(length: int, axis: int) -> onnx.ModelProto:
    lengths = numpy_helper.from_array(np.asarray([length], dtype=np.int64), "lengths")
    node = helper.make_node(
        "ReverseSequence", ["input", "lengths"], ["output"],
        batch_axis=0, time_axis=axis,
    )
    return make_output_only(
        node, [lengths], name=f"reverse_sequence_l{length}_a{axis}", opset=21
    )


def runtime_exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def four_exact(rows: dict[str, Any]) -> bool:
    if len(rows) != 4 or not all(runtime_exact(row) for row in rows.values()):
        return False
    return len({row.get("sign_sha256") for row in rows.values()}) == 1


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error",
        "first_shape_mismatch", "first_sign_mismatch", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def audited_cost(structure: dict[str, Any]) -> int | None:
    trace = structure.get("runtime_intermediate_trace", {})
    memory = trace.get("single_example_intermediate_bytes")
    if not isinstance(memory, int):
        return None
    return int(structure["initializer_elements"]) + memory


def validate_finalist(
    task: int,
    model: onnx.ModelProto,
    authority_cost: int,
    task_map: dict[str, str],
    label: str,
) -> dict[str, Any]:
    data = model.SerializeToString()
    structure = SUPPORT.structural_audit(task, model, data)
    cost = audited_cost(structure)
    result: dict[str, Any] = {
        "task": task,
        "label": label,
        "sha256": sha256(data),
        "file_bytes": len(data),
        "authority_cost": authority_cost,
        "structure": structure,
        "audited_actual_cost": cost,
        "strict_lower_actual": isinstance(cost, int) and cost < authority_cost,
    }
    if not structure["pass"] or not result["strict_lower_actual"]:
        result["decision"] = "REJECT_STRUCTURE_OR_COST"
        return result
    cases, counts = SUPPORT.known_cases(task)
    known = SUPPORT.evaluate_four(data, cases)
    result["known_counts"] = counts
    result["known_four"] = {name: compact(row) for name, row in known.items()}
    result["known_exact"] = four_exact(known)
    if not result["known_exact"]:
        result["decision"] = "REJECT_KNOWN"
        return result
    official = SUPPORT.official_profile(task, model, label)
    result["official_profile"] = official
    if official is None or int(official["cost"]) != cost or not official["correct"]:
        result["decision"] = "REJECT_OFFICIAL_PROFILE"
        return result
    fresh_rows = []
    for seed in (297_000_000 + task, 297_100_000 + task):
        fresh, generation = SUPPORT.fresh_cases(task, seed, task_map)
        four = SUPPORT.evaluate_four(data, fresh)
        exact = bool(
            generation["accepted"] == FRESH_PER_SEED
            and generation["generation_errors"] == 0
            and generation["conversion_skips"] == 0
            and four_exact(four)
        )
        fresh_rows.append({
            "seed": seed,
            "generation": generation,
            "four": {name: compact(row) for name, row in four.items()},
            "exact": exact,
        })
        print(json.dumps({
            "task": task,
            "label": label,
            "fresh_seed": seed,
            "exact": exact,
            "right": {name: row["right"] for name, row in four.items()},
        }), flush=True)
    result["fresh"] = fresh_rows
    result["fresh_exact"] = all(row["exact"] for row in fresh_rows)
    if not result["fresh_exact"]:
        result["decision"] = "REJECT_FRESH"
        return result
    result["decision"] = "ACCEPT"
    path = CANDIDATES / f"task{task:03d}_{label}_cost{cost}.onnx"
    path.write_bytes(data)
    result["candidate_path"] = str(path.relative_to(ROOT))
    result["score_gain"] = score_cost(cost) - score_cost(authority_cost)
    return result


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority SHA256 mismatch")
    costs = load_costs()
    tasks = sorted(task for task, cost in costs.items() if 26 <= cost <= 50)
    if len(tasks) != 35:
        raise RuntimeError(f"cost26..50 inventory changed: {len(tasks)}")
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    CANDIDATES.mkdir(parents=True, exist_ok=True)

    # Fixed-height vertical reversal in task385. The generator's height is
    # exactly 10, while the scorer pads to 30; ReverseSequence reverses only
    # the first ten rows and leaves the zero-padded suffix untouched.
    finalists = [
        validate_finalist(
            385, reverse_sequence_candidate(10, 2), costs[385], task_map,
            "reverse_first10_rows",
        )
    ]
    evidence = {
        "authority": {
            "path": str(AUTHORITY.relative_to(ROOT)),
            "sha256": AUTHORITY_SHA256,
            "baseline_lb": 8011.05,
        },
        "scope": {
            "cost_min": 26,
            "cost_max": 50,
            "task_count": len(tasks),
            "tasks": tasks,
            "fresh_per_seed": FRESH_PER_SEED,
            "fresh_seed_count": 2,
            "runtime_configs": [name for name, _, _ in SUPPORT.CONFIGS],
        },
        "finalists": finalists,
        "aggregate": dict(Counter(row["decision"] for row in finalists)),
        "elapsed_seconds": time.monotonic() - started,
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence": str(EVIDENCE.relative_to(ROOT)),
        "decisions": evidence["aggregate"],
    }), flush=True)
    return 0 if any(row["decision"] == "ACCEPT" for row in finalists) else 1


if __name__ == "__main__":
    raise SystemExit(main())

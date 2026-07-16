#!/usr/bin/env python3
"""Remove task399's scale scalar by using the exact red-cell count directly."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
OUTPUT = HERE / "task399_microcut.json"
CANDIDATE = HERE / "candidates/task399_direct_count_cost24.onnx"
FRESH_PER_SEED = 2_000


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("cost297_task399_support", SUPPORT_PATH)
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


def build() -> onnx.ModelProto:
    # Each legal 2x2 red box contributes exactly four non-background cells;
    # the generator's overlap guard makes all boxes disjoint. Background is
    # zero-hot, so sum(input) is exactly 4 * number_of_boxes.
    w = np.zeros((10, 1, 1, 1), dtype=np.int8)
    w[0, 0, 0, 0] = 1
    w[1, 0, 0, 0] = -1
    # ConvInteger computes (threshold - count) * W. With W[blue]=-1,
    # thresholds 4*k-2 turn blue on exactly when count >= 4*k.
    threshold = np.full((1, 1, 3, 3), 127, dtype=np.int8)
    threshold[0, 0, 0, 0] = 2
    threshold[0, 0, 0, 2] = 6
    threshold[0, 0, 1, 1] = 10
    threshold[0, 0, 2, 0] = 14
    threshold[0, 0, 2, 2] = 18
    nodes = [
        helper.make_node("Einsum", ["input"], ["s"], equation="nchw->n", name="red_count"),
        helper.make_node("Cast", ["s"], ["ci"], to=TensorProto.INT8, name="count_i8"),
        helper.make_node(
            "ConvInteger", ["thr", "W", "ci"], ["output"],
            kernel_shape=[1, 1], pads=[0, 0, 27, 27], name="threshold_pattern",
        ),
    ]
    graph = helper.make_graph(
        nodes, "task399_direct_count",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        initializer=[numpy_helper.from_array(threshold, "thr"), numpy_helper.from_array(w, "W")],
        value_info=[
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("ci", TensorProto.INT8, [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], producer_name="cost26_50_search_297")
    model.ir_version = 8
    return model


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total") and row.get("wrong") == 0
        and row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0 and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    model = build()
    data = model.SerializeToString()
    structure = SUPPORT.structural_audit(399, model, data)
    profile = SUPPORT.official_profile(399, model, "task399_direct_count")
    known_cases, known_counts = SUPPORT.known_cases(399)
    known_four = SUPPORT.evaluate_four(data, known_cases)
    known_exact = all(exact(row) for row in known_four.values())
    fresh = []
    if known_exact:
        for seed in (297_400_399, 297_500_399):
            cases, generation = SUPPORT.fresh_cases(399, seed, task_map)
            four = SUPPORT.evaluate_four(data, cases)
            fresh.append({
                "seed": seed,
                "generation": generation,
                "four": {name: compact(row) for name, row in four.items()},
                "exact": all(exact(row) for row in four.values()),
            })
    accepted = bool(
        structure["pass"] and profile is not None and int(profile["cost"]) < 25
        and profile["correct"] and known_exact and len(fresh) == 2
        and all(row["exact"] for row in fresh)
    )
    result = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": AUTHORITY_SHA256, "lb": 8011.05},
        "task": 399,
        "authority_cost": 25,
        "candidate_sha256": sha256(data),
        "structure": structure,
        "official_profile": profile,
        "known_counts": known_counts,
        "known_four": {name: compact(row) for name, row in known_four.items()},
        "known_exact": known_exact,
        "fresh": fresh,
        "accepted": accepted,
        "proof": "disjoint 2x2 boxes imply sum(input)=4*n; thresholds [2,6,10,14,18] encode n>=1..5",
        "elapsed_seconds": time.monotonic() - started,
    }
    if accepted:
        CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATE.write_bytes(data)
        result["candidate_path"] = str(CANDIDATE.relative_to(ROOT))
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted": accepted,
        "profile": profile,
        "known": {name: row["right"] for name, row in known_four.items()},
        "fresh": [{name: row["right"] for name, row in item["four"].items()} for item in fresh],
    }))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

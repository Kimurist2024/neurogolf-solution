#!/usr/bin/env python3
"""Reproduce exact-archive fresh defects without mutating submission artifacts."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASKS))
sys.path.insert(0, str(ROOT))

import common  # noqa: E402
import task_57aa92db as gen133  # noqa: E402
import task_db93a21d as gen349  # noqa: E402
from scripts.lib import scoring  # noqa: E402


MODELS_133 = {
    "exact_4403": HERE / "base/task133.onnx",
    "clean_rank_5570": ROOT / "scripts/golf/scratch_codex/task133/agent_clean_rank.onnx",
}
MODELS_349 = {
    "exact_3964": HERE / "base/task349.onnx",
    "tables_3956": ROOT
    / "scripts/golf/loop_7999_13/lane_b8/candidates/task349_radius_tables_len9.onnx",
    "relation_3954": ROOT
    / "scripts/golf/loop_7999_13/lane_b8/candidates/task349_radius_tables_len9_top_relation.onnx",
    "exact_or_4572": ROOT / "scripts/golf/scratch_codex/task349/agent_alt_exact_opt.onnx",
}


def onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for color in range(10):
        result[0, color, : arr.shape[0], : arr.shape[1]] = arr == color
    return result


def session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def run133() -> dict[str, object]:
    cases = []
    for seed in range(100):
        random.seed(seed)
        common.random.seed(seed)
        cases.append(gen133.generate())
    output: dict[str, object] = {}
    for label, path in MODELS_133.items():
        modes: dict[str, object] = {}
        for disable_all, mode in ((True, "disable_all"), (False, "default")):
            sess = session(path, disable_all)
            right = wrong = errors = 0
            first_failure: dict[str, object] | None = None
            for seed, case in enumerate(cases):
                try:
                    got = sess.run(["output"], {"input": onehot(case["input"])})[0] > 0
                    want = onehot(case["output"]) > 0
                    if np.array_equal(got, want):
                        right += 1
                    else:
                        wrong += 1
                        if first_failure is None:
                            first_failure = {"seed": seed, "kind": "wrong"}
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if first_failure is None:
                        first_failure = {
                            "seed": seed,
                            "kind": "runtime_error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            modes[mode] = {
                "right": right,
                "wrong": wrong,
                "errors": errors,
                "first_failure": first_failure,
            }
        output[label] = modes
    return output


def task349_case() -> dict[str, object]:
    random.seed(349101)
    np.random.seed(349101)
    case = None
    for _ in range(421):
        case = gen349.generate()
    assert case is not None
    return case


def run349() -> dict[str, object]:
    case = task349_case()
    x = onehot(case["input"])
    want = onehot(case["output"]) > 0
    output: dict[str, object] = {
        "seed": 349101,
        "valid_case": 421,
        "shape": [len(case["input"]), len(case["input"][0])],
        "models": {},
    }
    for label, path in MODELS_349.items():
        modes: dict[str, object] = {}
        for disable_all, mode in ((True, "disable_all"), (False, "default")):
            try:
                got = session(path, disable_all).run(["output"], {"input": x})[0] > 0
                diff = np.argwhere(got != want)
                modes[mode] = {
                    "different_entries": int(len(diff)),
                    "first_differences": diff[:16].tolist(),
                    "errors": 0,
                }
            except Exception as exc:  # noqa: BLE001
                modes[mode] = {
                    "different_entries": None,
                    "errors": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        output["models"][label] = modes
    return output


def main() -> None:
    ort.set_default_logger_severity(4)
    report = {"task133_fresh100": run133(), "task349_counterexample": run349()}
    path = HERE / "fresh_counterexamples.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

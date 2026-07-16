#!/usr/bin/env python3
"""Dual-ORT 5000-instance gate for every B17 pre-fresh survivor."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))

from lib import scoring  # noqa: E402


COUNT = 5000
SEED = 171799913


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def session(path: Path, mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def main() -> int:
    ort.set_default_logger_severity(4)
    audit = json.loads((HERE / "candidate_audit.json").read_text())
    eligible = audit["eligible_for_fresh5000"]
    runners: list[dict[str, Any]] = []
    for row in eligible:
        path = ROOT / row["path"]
        for mode in ("disabled", "default"):
            runners.append(
                {
                    "task": row["task"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "actual_cost": row["actual_score"]["cost"],
                    "mode": mode,
                    "session": session(path, mode),
                    "right": 0,
                    "wrong": 0,
                    "errors": 0,
                    "first_failure": None,
                    "margin_min_positive": None,
                    "margin_mid_count": 0,
                }
            )
    generators = {396: importlib.import_module("task_fcb5c309")}
    random.seed(SEED)
    for index in range(COUNT):
        examples = {task: generators[task].generate() for task in generators}
        encoded = {
            task: (encode(example["input"]), encode(example["output"]).astype(bool))
            for task, example in examples.items()
        }
        for runner in runners:
            input_value, expected = encoded[runner["task"]]
            try:
                raw = np.asarray(
                    runner["session"].run(["output"], {"input": input_value})[0]
                )
                positive = raw[raw > 0]
                if positive.size:
                    current = float(positive.min())
                    prior = runner["margin_min_positive"]
                    runner["margin_min_positive"] = current if prior is None else min(prior, current)
                runner["margin_mid_count"] += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
                if np.array_equal(raw > 0, expected):
                    runner["right"] += 1
                else:
                    runner["wrong"] += 1
                    runner["first_failure"] = runner["first_failure"] or {"index": index}
            except Exception as exc:  # noqa: BLE001
                runner["errors"] += 1
                runner["first_failure"] = runner["first_failure"] or {
                    "index": index,
                    "error": repr(exc),
                }
        if (index + 1) % 1000 == 0:
            print("progress", index + 1, flush=True)
    rows = []
    for runner in runners:
        runner.pop("session")
        runner["pass"] = (
            runner["right"] == COUNT
            and runner["wrong"] == 0
            and runner["errors"] == 0
            and runner["margin_mid_count"] == 0
        )
        rows.append(runner)
        print(runner, flush=True)
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sha.setdefault(row["sha256"], []).append(row)
    survivors = [
        {"sha256": digest, "rows": values}
        for digest, values in by_sha.items()
        if len(values) == 2 and all(value["pass"] for value in values)
    ]
    report = {
        "count": COUNT,
        "seed": SEED,
        "rows": rows,
        "dual_ort_survivors": survivors,
    }
    (HERE / "fresh5000_dual_ort.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

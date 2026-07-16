#!/usr/bin/env python3
"""Dual-ORT known and fresh5000 audit for every task125 candidate."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402

ARCHIVE_DIR = ROOT / "scripts/golf/loop_7999_13/lane_archive_top200"
MODELS = {
    "authority": HERE / "baseline_task125.onnx",
    **{f"archive_r{index:02d}": ARCHIVE_DIR / f"task125_r{index:02d}_static{static}.onnx"
       for index, static in enumerate((162, 167, 169, 170, 171, 171, 175, 186), 1)},
    "sound_pool14": ROOT / "scripts/golf/scratch_codex/task125/task125_pool14.onnx",
}
MODES = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "default": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def make_session(path: Path, level: ort.GraphOptimizationLevel) -> tuple[ort.InferenceSession | None, str | None]:
    try:
        model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
        if model is None:
            raise RuntimeError("sanitize failed")
        options = ort.SessionOptions()
        options.graph_optimization_level = level
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        return ort.InferenceSession(model.SerializeToString(), options), None
    except Exception as exc:
        return None, repr(exc)


def main() -> None:
    sessions = {
        (mode, label): make_session(path, level)
        for mode, level in MODES.items()
        for label, path in MODELS.items()
    }
    rows = {}
    for mode in MODES:
        for label in MODELS:
            rows[(mode, label)] = {
                "mode": mode,
                "label": label,
                "session_error": sessions[(mode, label)][1],
                "known_correct": 0,
                "known_wrong": 0,
                "known_errors": 0,
                "known_raw_equal_authority": 0,
                "fresh_correct": 0,
                "fresh_wrong": 0,
                "fresh_errors": 0,
                "fresh_raw_equal_authority": 0,
                "first_fresh_failure": None,
            }
    examples = scoring.load_examples(125)
    known = examples["train"] + examples["test"] + examples["arc-gen"]
    for example in known:
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        want = benchmark["output"].astype(bool)
        for mode in MODES:
            authority_session = sessions[(mode, "authority")][0]
            assert authority_session is not None
            authority_raw = authority_session.run(["output"], {"input": benchmark["input"]})[0]
            for label in MODELS:
                row = rows[(mode, label)]
                candidate_session = sessions[(mode, label)][0]
                if candidate_session is None:
                    row["known_errors"] += 1
                    continue
                try:
                    raw = candidate_session.run(["output"], {"input": benchmark["input"]})[0]
                    row["known_correct" if np.array_equal(raw > 0, want) else "known_wrong"] += 1
                    row["known_raw_equal_authority"] += int(
                        np.array_equal(raw, authority_raw, equal_nan=True)
                    )
                except Exception:
                    row["known_errors"] += 1

    module = importlib.import_module("task_543a7ed5")
    random.seed(125_800_263)
    started = time.monotonic()
    for case_index in range(5000):
        case = module.generate()
        benchmark = encode(case["input"])
        want = encode(case["output"]).astype(bool)
        for mode in MODES:
            authority_session = sessions[(mode, "authority")][0]
            assert authority_session is not None
            authority_raw = authority_session.run(["output"], {"input": benchmark})[0]
            for label in MODELS:
                row = rows[(mode, label)]
                candidate_session = sessions[(mode, label)][0]
                if candidate_session is None:
                    row["fresh_errors"] += 1
                    continue
                try:
                    raw = candidate_session.run(["output"], {"input": benchmark})[0]
                    equal_gold = np.array_equal(raw > 0, want)
                    row["fresh_correct" if equal_gold else "fresh_wrong"] += 1
                    row["fresh_raw_equal_authority"] += int(
                        np.array_equal(raw, authority_raw, equal_nan=True)
                    )
                    if not equal_gold and row["first_fresh_failure"] is None:
                        difference = np.argwhere((raw > 0) != want)
                        row["first_fresh_failure"] = {
                            "case": case_index,
                            "different_cells": int(len(difference)),
                            "first_difference": difference[0].tolist(),
                        }
                except Exception as exc:
                    row["fresh_errors"] += 1
                    if row["first_fresh_failure"] is None:
                        row["first_fresh_failure"] = {"case": case_index, "error": repr(exc)}
    report = {
        "known_cases": len(known),
        "fresh_cases": 5000,
        "seed": 125_800_263,
        "elapsed_seconds": time.monotonic() - started,
        "rows": list(rows.values()),
    }
    (HERE / "dual_known_fresh5000.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

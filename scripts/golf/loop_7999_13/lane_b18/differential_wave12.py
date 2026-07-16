#!/usr/bin/env python3
"""Raw-bitwise differential audit for the Wave12 task089 exact shave."""

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


BASE = HERE / "baseline/task089.onnx"
CANDIDATE = ROOT / "scripts/golf/loop_7999_13/wave12_exact_shave/task089.onnx"
COUNT = 5000
SEED = 181799913


def encode(grid: list[list[int]]) -> np.ndarray:
    value = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, cells in enumerate(grid):
        for col, color in enumerate(cells):
            value[0, color, row, col] = 1.0
    return value


def make_session(path: Path, mode: str, sanitized: bool) -> tuple[Any, str | None]:
    model = onnx.load(path)
    if sanitized:
        model = scoring.sanitize_model(copy.deepcopy(model))
        if model is None:
            return None, "sanitize_failed"
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    try:
        return ort.InferenceSession(model.SerializeToString(), options), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def compare_set(
    examples: list[tuple[str, np.ndarray, np.ndarray]],
    mode: str,
    sanitized: bool,
) -> dict[str, Any]:
    base, base_session_error = make_session(BASE, mode, sanitized)
    cand, candidate_session_error = make_session(CANDIDATE, mode, sanitized)
    row: dict[str, Any] = {
        "mode": mode,
        "sanitized": sanitized,
        "total": len(examples),
        "base_session_error": base_session_error,
        "candidate_session_error": candidate_session_error,
        "base_errors": 0,
        "candidate_errors": 0,
        "bitwise_equal": 0,
        "bitwise_different": 0,
        "base_gold_right": 0,
        "candidate_gold_right": 0,
        "first_failure": None,
    }
    if base is None:
        row["base_errors"] = len(examples)
    if cand is None:
        row["candidate_errors"] = len(examples)
    for label, input_value, expected in examples:
        base_raw = candidate_raw = None
        if base is not None:
            try:
                base_raw = np.asarray(base.run(["output"], {"input": input_value})[0])
                row["base_gold_right"] += int(
                    np.array_equal(base_raw > 0, expected)
                )
            except Exception as exc:  # noqa: BLE001
                row["base_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "label": label,
                    "side": "base",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if cand is not None:
            try:
                candidate_raw = np.asarray(
                    cand.run(["output"], {"input": input_value})[0]
                )
                row["candidate_gold_right"] += int(
                    np.array_equal(candidate_raw > 0, expected)
                )
            except Exception as exc:  # noqa: BLE001
                row["candidate_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "label": label,
                    "side": "candidate",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if base_raw is not None and candidate_raw is not None:
            if np.array_equal(base_raw, candidate_raw):
                row["bitwise_equal"] += 1
            else:
                row["bitwise_different"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "label": label,
                    "side": "differential",
                }
    row["pass"] = (
        row["base_errors"] == 0
        and row["candidate_errors"] == 0
        and row["bitwise_equal"] == len(examples)
        and row["bitwise_different"] == 0
    )
    return row


def main() -> int:
    ort.set_default_logger_severity(4)
    known: list[tuple[str, np.ndarray, np.ndarray]] = []
    loaded = scoring.load_examples(89)
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(loaded[subset]):
            converted = scoring.convert_to_numpy(example)
            if converted is not None:
                known.append(
                    (
                        f"{subset}[{index}]",
                        converted["input"],
                        converted["output"].astype(bool),
                    )
                )
    generator = importlib.import_module("task_3e980e27")
    fresh: list[tuple[str, np.ndarray, np.ndarray]] = []
    random.seed(SEED)
    for index in range(COUNT):
        example = generator.generate()
        fresh.append(
            (
                f"fresh[{index}]",
                encode(example["input"]),
                encode(example["output"]).astype(bool),
            )
        )
    rows = []
    for label, examples in (("known", known), ("fresh5000", fresh)):
        for sanitized in (False, True):
            for mode in ("disabled", "default"):
                row = compare_set(examples, mode, sanitized)
                row["set"] = label
                rows.append(row)
                print(row, flush=True)
    report = {
        "base": str(BASE.relative_to(ROOT)),
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "count": COUNT,
        "seed": SEED,
        "rows": rows,
        "dual_ort_bitwise_pass": all(row["pass"] for row in rows),
    }
    (HERE / "wave12_differential.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

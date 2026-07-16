#!/usr/bin/env python3
"""Independent task205 fresh-5000 gold check on both ORT modes."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


TASK = 205
TASK_HASH = "8731374e"
COUNT = 5000
SEED = 93_023_205
CANDIDATES = {
    "task205_r02": HERE / "candidates/task205_r02.onnx",
    "task205_c3_cost1038": HERE / "candidates/task205_c3_cost1038.onnx",
}


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
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def stats() -> dict[str, object]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "decoded_equal_baseline": 0,
        "first_failure": None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    random.seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    modes = ((True, "disable_all"), (False, "default"))
    base_sessions = {mode: session(HERE / "baseline/task205.onnx", disabled) for disabled, mode in modes}
    candidate_sessions = {
        (label, mode): session(path, disabled)
        for label, path in CANDIDATES.items()
        for disabled, mode in modes
    }
    base = {mode: stats() for _, mode in modes}
    candidates = {label: {mode: stats() for _, mode in modes} for label in CANDIDATES}
    grid_shapes: Counter[str] = Counter()
    output_shapes: Counter[str] = Counter()
    started = time.monotonic()
    attempts = valid = generation_errors = conversion_skips = 0
    first_generation_error = None
    while valid < COUNT:
        attempts += 1
        try:
            example = generator.generate()
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if first_generation_error is None:
                first_generation_error = {"attempt": attempts, "error": f"{type(exc).__name__}: {exc}"}
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            conversion_skips += 1
            continue
        valid += 1
        grid_shapes[f"{len(example['input'])}x{len(example['input'][0])}"] += 1
        output_shapes[f"{len(example['output'])}x{len(example['output'][0])}"] += 1
        expected = benchmark["output"] > 0
        feed = benchmark["input"]
        for _, mode in modes:
            base_raw = None
            base_row = base[mode]
            try:
                sess = base_sessions[mode]
                base_raw = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: feed})[0]
                if np.array_equal(base_raw > 0, expected):
                    base_row["right"] += 1
                else:
                    base_row["wrong"] += 1
                    if base_row["first_failure"] is None:
                        base_row["first_failure"] = {"valid_case": valid, "raw_shape": list(base_raw.shape)}
            except Exception as exc:  # noqa: BLE001
                base_row["runtime_errors"] += 1
                if base_row["first_failure"] is None:
                    base_row["first_failure"] = {"valid_case": valid, "runtime": f"{type(exc).__name__}: {exc}"}
            for label in CANDIDATES:
                row = candidates[label][mode]
                try:
                    sess = candidate_sessions[(label, mode)]
                    raw = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: feed})[0]
                    if np.array_equal(raw > 0, expected):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "valid_case": valid,
                                "raw_shape": list(raw.shape),
                                "expected_shape": list(expected.shape),
                                "different_cells": int(np.count_nonzero((raw > 0) != expected)) if raw.shape == expected.shape else None,
                            }
                    if base_raw is not None:
                        row["decoded_equal_baseline"] += int(np.array_equal(raw > 0, base_raw > 0))
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {"valid_case": valid, "runtime": f"{type(exc).__name__}: {exc}"}
        if valid % 500 == 0:
            print(f"task205 valid={valid}/{COUNT}", flush=True)
    passed = {
        label: all(
            candidates[label][mode]["right"] == COUNT
            and candidates[label][mode]["wrong"] == 0
            and candidates[label][mode]["runtime_errors"] == 0
            for _, mode in modes
        )
        for label in CANDIDATES
    }
    result = {
        "task": TASK,
        "task_hash": TASK_HASH,
        "seed": SEED,
        "requested_valid": COUNT,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "first_generation_error": first_generation_error,
        "grid_shapes": dict(sorted(grid_shapes.items())),
        "output_shapes": dict(sorted(output_shapes.items())),
        "admission_uses_generator_gold": True,
        "baseline": base,
        "candidates": candidates,
        "candidate_pass": passed,
        "all_candidates_pass": all(passed.values()),
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (HERE / "fresh_dual_5000.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {passed}", flush=True)
    return 0 if all(passed.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

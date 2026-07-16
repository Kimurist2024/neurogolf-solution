#!/usr/bin/env python3
"""Independent task158 fresh-5000 gold check on both ORT modes."""

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


TASK = 158
TASK_HASH = "6aa20dc0"
COUNT = 5000
SEED = 92_215_158


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


def stats() -> dict[str, object]:
    return {
        "right": 0,
        "wrong": 0,
        "runtime_errors": 0,
        "decoded_equal_baseline": 0,
        "raw_bitwise_equal_baseline": 0,
        "max_abs_raw_difference": 0.0,
        "first_failure": None,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    random.seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    modes = ((True, "disable_all"), (False, "default"))
    base_sessions = {
        mode: session(HERE / "baseline/task158.onnx", disabled)
        for disabled, mode in modes
    }
    candidate_sessions = {
        mode: session(HERE / "sound/task158_spec_anchor13.onnx", disabled)
        for disabled, mode in modes
    }
    base = {
        mode: {"right": 0, "wrong": 0, "runtime_errors": 0, "first_failure": None}
        for _, mode in modes
    }
    candidate = {mode: stats() for _, mode in modes}
    started = time.monotonic()
    attempts = valid = generation_errors = conversion_skips = 0
    grid_shapes: Counter[str] = Counter()
    first_generation_error = None
    while valid < COUNT:
        attempts += 1
        try:
            example = generator.generate()
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if first_generation_error is None:
                first_generation_error = {
                    "attempt": attempts,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            conversion_skips += 1
            continue
        valid += 1
        grid_shapes[f"{len(example['input'])}x{len(example['input'][0])}"] += 1
        expected = benchmark["output"] > 0
        feed = benchmark["input"]
        for _, mode in modes:
            base_raw = None
            try:
                sess = base_sessions[mode]
                base_raw = sess.run(
                    [sess.get_outputs()[0].name], {sess.get_inputs()[0].name: feed}
                )[0]
                if np.array_equal(base_raw > 0, expected):
                    base[mode]["right"] += 1
                else:
                    base[mode]["wrong"] += 1
                    if base[mode]["first_failure"] is None:
                        base[mode]["first_failure"] = {
                            "valid_case": valid,
                            "different_cells": int(np.count_nonzero((base_raw > 0) != expected)),
                        }
            except Exception as exc:  # noqa: BLE001
                base[mode]["runtime_errors"] += 1
                if base[mode]["first_failure"] is None:
                    base[mode]["first_failure"] = {
                        "valid_case": valid,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
            row = candidate[mode]
            try:
                sess = candidate_sessions[mode]
                raw = sess.run(
                    [sess.get_outputs()[0].name], {sess.get_inputs()[0].name: feed}
                )[0]
                if np.array_equal(raw > 0, expected):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "stage": "gold_mismatch",
                            "valid_case": valid,
                            "different_cells": int(np.count_nonzero((raw > 0) != expected)),
                        }
                if base_raw is not None:
                    row["decoded_equal_baseline"] += int(
                        np.array_equal(raw > 0, base_raw > 0)
                    )
                    row["raw_bitwise_equal_baseline"] += int(
                        np.array_equal(raw, base_raw, equal_nan=True)
                    )
                    difference = np.abs(
                        np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                        - np.nan_to_num(base_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    )
                    row["max_abs_raw_difference"] = max(
                        float(row["max_abs_raw_difference"]),
                        float(difference.max(initial=0.0)),
                    )
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "stage": "runtime",
                        "valid_case": valid,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        if valid % 250 == 0:
            print(f"task158 valid={valid}/{COUNT}", flush=True)
            partial = {
                "task": TASK,
                "task_hash": TASK_HASH,
                "seed": SEED,
                "valid": valid,
                "attempts": attempts,
                "grid_shapes": dict(sorted(grid_shapes.items())),
                "baseline": base,
                "candidate": candidate,
                "complete": False,
            }
            (HERE / "fresh_dual_5000.json").write_text(json.dumps(partial, indent=2) + "\n")
    passed = all(
        candidate[mode]["right"] == COUNT
        and candidate[mode]["wrong"] == 0
        and candidate[mode]["runtime_errors"] == 0
        for _, mode in modes
    )
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
        "independent_from_prior_reported_seeds": True,
        "admission_uses_gold_not_baseline_equivalence": True,
        "baseline": base,
        "candidate": candidate,
        "passed": passed,
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    (HERE / "fresh_dual_5000.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE passed={passed}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

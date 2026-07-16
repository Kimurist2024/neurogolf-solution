#!/usr/bin/env python3
"""Independent fresh 5000-valid-case differential on both ORT modes."""

from __future__ import annotations

import copy
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


COUNT = 5000
SEED_BASE = 71_418_000
TASK_HASH = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
CANDIDATES = {
    63: ("r01",),
    139: ("r04",),
    202: ("r02", "r03"),
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
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_stats() -> dict[str, object]:
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
    output: dict[str, object] = {
        "count_per_mode": COUNT,
        "seed_base": SEED_BASE,
        "independent_from_quick20_and_k500": True,
        "tasks": {},
    }
    all_pass = True
    for task, variants in CANDIDATES.items():
        started = time.monotonic()
        seed = SEED_BASE + task
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        generator = importlib.import_module(f"task_{TASK_HASH[f'{task:03d}']}")
        modes = ((True, "disable_all"), (False, "default"))
        base_sessions = {
            label: session(HERE / "baseline" / f"task{task:03d}.onnx", disabled)
            for disabled, label in modes
        }
        candidate_sessions = {
            (variant, label): session(
                HERE / "candidates" / f"task{task:03d}_{variant}.onnx", disabled
            )
            for variant in variants
            for disabled, label in modes
        }
        base_stats = {
            label: {"right": 0, "wrong": 0, "runtime_errors": 0}
            for _, label in modes
        }
        candidate_stats = {
            variant: {label: empty_stats() for _, label in modes}
            for variant in variants
        }
        valid = attempts = generation_errors = conversion_skips = 0
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
            expected = benchmark["output"] > 0
            for _, mode in modes:
                base_raw = None
                try:
                    base_session = base_sessions[mode]
                    base_raw = base_session.run(
                        [base_session.get_outputs()[0].name],
                        {base_session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    if np.array_equal(base_raw > 0, expected):
                        base_stats[mode]["right"] += 1
                    else:
                        base_stats[mode]["wrong"] += 1
                except Exception:  # noqa: BLE001
                    base_stats[mode]["runtime_errors"] += 1
                for variant in variants:
                    stats = candidate_stats[variant][mode]
                    try:
                        candidate_session = candidate_sessions[(variant, mode)]
                        candidate_raw = candidate_session.run(
                            [candidate_session.get_outputs()[0].name],
                            {
                                candidate_session.get_inputs()[0].name: benchmark[
                                    "input"
                                ]
                            },
                        )[0]
                        if np.array_equal(candidate_raw > 0, expected):
                            stats["right"] += 1
                        else:
                            stats["wrong"] += 1
                            if stats["first_failure"] is None:
                                stats["first_failure"] = {
                                    "stage": "gold_mismatch",
                                    "valid_case": valid,
                                    "different_cells": int(
                                        np.count_nonzero((candidate_raw > 0) != expected)
                                    ),
                                }
                        if base_raw is not None:
                            stats["decoded_equal_baseline"] += int(
                                np.array_equal(candidate_raw > 0, base_raw > 0)
                            )
                            stats["raw_bitwise_equal_baseline"] += int(
                                np.array_equal(candidate_raw, base_raw, equal_nan=True)
                            )
                            difference = np.abs(
                                np.nan_to_num(candidate_raw, nan=0.0, posinf=0.0, neginf=0.0)
                                - np.nan_to_num(base_raw, nan=0.0, posinf=0.0, neginf=0.0)
                            )
                            stats["max_abs_raw_difference"] = max(
                                float(stats["max_abs_raw_difference"]),
                                float(difference.max(initial=0.0)),
                            )
                    except Exception as exc:  # noqa: BLE001
                        stats["runtime_errors"] += 1
                        if stats["first_failure"] is None:
                            stats["first_failure"] = {
                                "stage": "runtime",
                                "valid_case": valid,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
            if valid % 500 == 0:
                print(f"task{task:03d} valid={valid}/{COUNT}", flush=True)
        candidate_pass = {}
        for variant in variants:
            passed = all(
                candidate_stats[variant][mode]["right"] == COUNT
                and candidate_stats[variant][mode]["wrong"] == 0
                and candidate_stats[variant][mode]["runtime_errors"] == 0
                and candidate_stats[variant][mode]["decoded_equal_baseline"] == COUNT
                for _, mode in modes
            )
            candidate_pass[variant] = passed
            all_pass = all_pass and passed
        row = {
            "seed": seed,
            "requested_valid": COUNT,
            "valid": valid,
            "attempts": attempts,
            "generation_errors": generation_errors,
            "conversion_skips": conversion_skips,
            "first_generation_error": first_generation_error,
            "baseline": base_stats,
            "candidates": candidate_stats,
            "candidate_pass": candidate_pass,
            "elapsed_seconds": time.monotonic() - started,
        }
        output["tasks"][str(task)] = row
        (HERE / "fresh_dual_5000.json").write_text(
            json.dumps(output, indent=2) + "\n"
        )
        print(f"task{task:03d} DONE {candidate_pass}", flush=True)
    output["all_candidates_pass"] = all_pass
    (HERE / "fresh_dual_5000.json").write_text(json.dumps(output, indent=2) + "\n")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

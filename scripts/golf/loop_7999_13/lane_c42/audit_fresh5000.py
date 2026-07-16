#!/usr/bin/env python3
"""Dual-ORT raw differential for the task379 rank-2 QV candidate."""

from __future__ import annotations

import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "golf" / "loop_7999_13"))
from dual_ort_fresh import MAP, make_session, scoring  # noqa: E402


TASK = 379
COUNT = 5000
SEED = 800_263_379_042
BASELINE = HERE / "baseline" / "task379.onnx"
CANDIDATE = HERE / "candidates" / "task379_qv_middle_rank2.onnx"
OUTPUT = HERE / "task379_qv_middle_rank2_fresh5000.json"


def main() -> int:
    baseline = onnx.load(BASELINE)
    candidate = onnx.load(CANDIDATE)
    generator = importlib.import_module(f"task_{MAP[f'{TASK:03d}']}")
    random.seed(SEED)
    examples: list[dict[str, np.ndarray]] = []
    generation_errors = 0
    while len(examples) < COUNT:
        try:
            benchmark = scoring.convert_to_numpy(generator.generate())
            if benchmark is not None:
                examples.append(benchmark)
        except Exception:  # noqa: BLE001
            generation_errors += 1

    report: dict[str, object] = {
        "task": TASK,
        "count": COUNT,
        "seed": SEED,
        "generation_errors": generation_errors,
        "modes": {},
    }
    passed = generation_errors == 0
    for disabled, label in ((True, "disable_all"), (False, "default")):
        baseline_session = make_session(baseline, disabled)
        candidate_session = make_session(candidate, disabled)
        row = {
            "baseline_right": 0,
            "candidate_right": 0,
            "baseline_runtime_errors": 0,
            "candidate_runtime_errors": 0,
            "raw_bitwise_equal": 0,
            "decoded_equal": 0,
            "max_abs_raw_difference": 0.0,
        }
        for benchmark in examples:
            expected = benchmark["output"] > 0
            try:
                baseline_raw = baseline_session.run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                row["baseline_runtime_errors"] += 1
                continue
            try:
                candidate_raw = candidate_session.run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
            except Exception:  # noqa: BLE001
                row["candidate_runtime_errors"] += 1
                continue
            row["baseline_right"] += int(np.array_equal(baseline_raw > 0, expected))
            row["candidate_right"] += int(np.array_equal(candidate_raw > 0, expected))
            row["raw_bitwise_equal"] += int(
                np.array_equal(baseline_raw, candidate_raw, equal_nan=True)
            )
            row["decoded_equal"] += int(
                np.array_equal(baseline_raw > 0, candidate_raw > 0)
            )
            difference = np.abs(
                np.nan_to_num(baseline_raw, nan=0.0, posinf=0.0, neginf=0.0)
                - np.nan_to_num(candidate_raw, nan=0.0, posinf=0.0, neginf=0.0)
            )
            row["max_abs_raw_difference"] = max(
                row["max_abs_raw_difference"], float(difference.max(initial=0.0))
            )
        row["pass"] = bool(
            row["baseline_runtime_errors"] == 0
            and row["candidate_runtime_errors"] == 0
            and row["raw_bitwise_equal"] == COUNT
            and row["decoded_equal"] == COUNT
        )
        report["modes"][label] = row
        passed &= row["pass"]
        print(label, json.dumps(row), flush=True)
    report["pass"] = bool(passed)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "output": str(OUTPUT.relative_to(ROOT))}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

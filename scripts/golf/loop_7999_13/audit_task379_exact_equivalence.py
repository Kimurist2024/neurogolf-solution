#!/usr/bin/env python3
"""Audit task379's exact tensor-mode reuse against the exact 7999.13 member."""

from __future__ import annotations

import importlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import onnx

from dual_ort_fresh import MAP, ROOT, make_session, scoring


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "lane_tensor_mode_reuse" / "task379_nv_from_qv.onnx"
OUTPUT = HERE / "lane_tensor_mode_reuse" / "task379_exact_equivalence_5000.json"
COUNT = 5000
SEED = 379_799_913


def main() -> int:
    with zipfile.ZipFile(ROOT / "submission_base_7999.13.zip") as archive:
        baseline = onnx.load_model(io.BytesIO(archive.read("task379.onnx")))
    candidate = onnx.load(CANDIDATE)
    module = importlib.import_module(f"task_{MAP['379']}")
    random.seed(SEED)
    examples = []
    while len(examples) < COUNT:
        benchmark = scoring.convert_to_numpy(module.generate())
        if benchmark is not None:
            examples.append(benchmark)

    result: dict[str, object] = {
        "task": 379,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "count": COUNT,
        "seed": SEED,
        "modes": {},
    }
    all_equal = True
    for disabled, label in ((True, "disable_all"), (False, "default")):
        base_session = make_session(baseline, disabled)
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
                raw_base = base_session.run(["output"], {"input": benchmark["input"]})[0]
            except Exception:
                row["baseline_runtime_errors"] += 1
                continue
            try:
                raw_candidate = candidate_session.run(
                    ["output"], {"input": benchmark["input"]}
                )[0]
            except Exception:
                row["candidate_runtime_errors"] += 1
                continue
            row["baseline_right"] += int(np.array_equal(raw_base > 0, expected))
            row["candidate_right"] += int(np.array_equal(raw_candidate > 0, expected))
            row["raw_bitwise_equal"] += int(
                np.array_equal(raw_base, raw_candidate, equal_nan=True)
            )
            row["decoded_equal"] += int(
                np.array_equal(raw_base > 0, raw_candidate > 0)
            )
            difference = np.abs(
                np.nan_to_num(raw_base, nan=0.0, posinf=0.0, neginf=0.0)
                - np.nan_to_num(raw_candidate, nan=0.0, posinf=0.0, neginf=0.0)
            )
            row["max_abs_raw_difference"] = max(
                row["max_abs_raw_difference"], float(difference.max(initial=0.0))
            )
        result["modes"][label] = row
        all_equal &= (
            row["baseline_runtime_errors"] == 0
            and row["candidate_runtime_errors"] == 0
            and row["raw_bitwise_equal"] == COUNT
            and row["decoded_equal"] == COUNT
        )
    result["exact_equivalent_on_audit"] = all_equal
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

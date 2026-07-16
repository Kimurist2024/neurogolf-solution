#!/usr/bin/env python3
"""Dual-ORT raw/decoded differential against an exact baseline ZIP member."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import onnx

from dual_ort_fresh import MAP, ROOT, make_session, scoring


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "submission_base_7999.13.zip")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=799_913)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    baseline_path = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    with zipfile.ZipFile(baseline_path) as archive:
        baseline = onnx.load_model(io.BytesIO(archive.read(f"task{args.task:03d}.onnx")))
    candidate = onnx.load(candidate_path)
    module = importlib.import_module(f"task_{MAP[f'{args.task:03d}']}")
    random.seed(args.seed)
    examples = []
    generation_errors = 0
    while len(examples) < args.count:
        try:
            benchmark = scoring.convert_to_numpy(module.generate())
            if benchmark is not None:
                examples.append(benchmark)
            else:
                generation_errors += 1
        except Exception:
            generation_errors += 1

    result: dict[str, object] = {
        "task": args.task,
        "candidate": str(candidate_path.relative_to(ROOT)),
        "baseline": str(baseline_path.relative_to(ROOT)),
        "count": args.count,
        "seed": args.seed,
        "generation_errors": generation_errors,
        "modes": {},
    }
    all_decoded_equal = generation_errors == 0
    all_raw_equal = generation_errors == 0
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
            row["raw_bitwise_equal"] += int(np.array_equal(raw_base, raw_candidate, equal_nan=True))
            row["decoded_equal"] += int(np.array_equal(raw_base > 0, raw_candidate > 0))
            # Boolean outputs cannot be subtracted by NumPy.  Cast only for
            # the diagnostic magnitude; bitwise/decoded equality above stays
            # in the model's native dtype.
            finite_base = np.nan_to_num(raw_base, nan=0.0, posinf=0.0, neginf=0.0)
            finite_candidate = np.nan_to_num(
                raw_candidate, nan=0.0, posinf=0.0, neginf=0.0
            )
            difference = np.abs(
                finite_base.astype(np.float64, copy=False)
                - finite_candidate.astype(np.float64, copy=False)
            )
            row["max_abs_raw_difference"] = max(
                row["max_abs_raw_difference"], float(difference.max(initial=0.0))
            )
        result["modes"][label] = row
        mode_clean = row["baseline_runtime_errors"] == row["candidate_runtime_errors"] == 0
        all_decoded_equal &= mode_clean and row["decoded_equal"] == args.count
        all_raw_equal &= mode_clean and row["raw_bitwise_equal"] == args.count
    result["raw_bitwise_equivalent_on_audit"] = all_raw_equal
    result["decoded_equivalent_on_audit"] = all_decoded_equal
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all_decoded_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())

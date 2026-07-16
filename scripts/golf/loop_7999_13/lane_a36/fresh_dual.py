#!/usr/bin/env python3
"""Gold and raw-equivalence audit for the A36 task158 candidate."""

from __future__ import annotations

import argparse
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


def session(path: Path, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    random.seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    generator = importlib.import_module("task_6aa20dc0")
    modes = ((True, "disable_all"), (False, "default"))
    paths = {
        "baseline": HERE / "baseline_task158.onnx",
        "candidate": HERE / "task158_perm_mask_reuse_7615.onnx",
    }
    sessions = {
        label: {mode: session(path, disabled) for disabled, mode in modes}
        for label, path in paths.items()
    }
    rows = {
        label: {
            mode: {
                "right": 0,
                "wrong": 0,
                "runtime_errors": 0,
                "raw_equal_other": 0,
                "decoded_equal_other": 0,
                "max_abs_raw_difference": 0.0,
                "first_failure": None,
            }
            for _, mode in modes
        }
        for label in paths
    }
    shapes: Counter[str] = Counter()
    generated = generation_errors = 0
    started = time.monotonic()
    while generated < args.count:
        try:
            example = generator.generate()
        except Exception:
            generation_errors += 1
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        generated += 1
        shapes[f"{len(example['input'])}x{len(example['input'][0])}"] += 1
        expected = benchmark["output"] > 0
        for _, mode in modes:
            raw: dict[str, np.ndarray] = {}
            for label in paths:
                row = rows[label][mode]
                try:
                    sess = sessions[label][mode]
                    value = sess.run(
                        [sess.get_outputs()[0].name],
                        {sess.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    raw[label] = value
                    if np.array_equal(value > 0, expected):
                        row["right"] += 1
                    else:
                        row["wrong"] += 1
                        if row["first_failure"] is None:
                            row["first_failure"] = {
                                "case": generated,
                                "different_cells": int(np.count_nonzero((value > 0) != expected)),
                            }
                except Exception as exc:  # noqa: BLE001
                    row["runtime_errors"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {"case": generated, "error": repr(exc)}
            if len(raw) == 2:
                same_raw = np.array_equal(raw["baseline"], raw["candidate"], equal_nan=True)
                same_decoded = np.array_equal(raw["baseline"] > 0, raw["candidate"] > 0)
                diff = np.abs(
                    np.nan_to_num(raw["baseline"], nan=0.0, posinf=0.0, neginf=0.0)
                    - np.nan_to_num(raw["candidate"], nan=0.0, posinf=0.0, neginf=0.0)
                )
                maximum = float(diff.max(initial=0.0))
                for label in paths:
                    rows[label][mode]["raw_equal_other"] += int(same_raw)
                    rows[label][mode]["decoded_equal_other"] += int(same_decoded)
                    rows[label][mode]["max_abs_raw_difference"] = max(
                        float(rows[label][mode]["max_abs_raw_difference"]), maximum
                    )
        if generated % 250 == 0:
            print(f"task158 {generated}/{args.count}", flush=True)
    expected_shapes = {
        f"{height}x{width}"
        for width in range(15, 26)
        for height in range(width - 1, width + 2)
    }
    result = {
        "task": 158,
        "seed": args.seed,
        "generated": generated,
        "generation_errors": generation_errors,
        "shape_counts": dict(sorted(shapes.items())),
        "expected_reachable_shapes": sorted(expected_shapes),
        "all_33_reachable_shapes_seen": set(shapes) == expected_shapes,
        "results": rows,
        "passed": all(
            rows["candidate"][mode]["right"] == args.count
            and rows["candidate"][mode]["wrong"] == 0
            and rows["candidate"][mode]["runtime_errors"] == 0
            and rows["candidate"][mode]["raw_equal_other"] == args.count
            for _, mode in modes
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "all_shapes": result["all_33_reachable_shapes_seen"]}))
    return 0 if result["passed"] and result["all_33_reachable_shapes_seen"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fresh audit with wrong outputs and runtime errors counted separately."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402


TASK_MAP = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("disable_all", "default"), required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(model_path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if args.mode == "disable_all":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    runner = ort.InferenceSession(model.SerializeToString(), options)
    generator = importlib.import_module(f"task_{TASK_MAP[f'{args.task:03d}']}")
    random.seed(args.seed)
    right = wrong = runtime_errors = generation_errors = 0
    first_wrong = first_runtime_error = None
    generated = 0
    while generated < args.count:
        try:
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
        except Exception as exc:  # noqa: BLE001
            generation_errors += 1
            if generation_errors > args.count:
                raise RuntimeError("generator repeatedly failed") from exc
            continue
        if benchmark is None:
            generation_errors += 1
            continue
        generated += 1
        try:
            raw = runner.run(["output"], {"input": benchmark["input"]})[0]
        except Exception as exc:  # noqa: BLE001
            runtime_errors += 1
            if first_runtime_error is None:
                first_runtime_error = {"index": generated - 1, "error": repr(exc)}
            continue
        if np.array_equal(raw > 0.0, benchmark["output"] > 0.0):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                first_wrong = {
                    "index": generated - 1,
                    "input_shape": list(benchmark["input"].shape),
                    "differing_elements": int(np.count_nonzero((raw > 0.0) != (benchmark["output"] > 0.0))),
                }
    report = {
        "task": args.task,
        "model": str(model_path.relative_to(ROOT)),
        "mode": args.mode,
        "seed": args.seed,
        "requested": args.count,
        "generated": generated,
        "generation_errors": generation_errors,
        "right": right,
        "wrong": wrong,
        "runtime_errors": runtime_errors,
        "accuracy": right / generated,
        "first_wrong": first_wrong,
        "first_runtime_error": first_runtime_error,
        "passes_user_95_percent_gate": right / generated >= 0.95 and runtime_errors == 0,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

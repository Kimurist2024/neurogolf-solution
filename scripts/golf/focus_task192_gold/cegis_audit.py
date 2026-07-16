#!/usr/bin/env python3
"""Fresh full-ONNX audit for task192, optionally caching only failures."""

from __future__ import annotations

import argparse
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(HERE))
from lib import scoring  # noqa: E402
import train_rank7_noise as training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument(
        "--model", type=Path,
        default=HERE / "candidates/task192_rank8_gap_exact_argmax.onnx",
    )
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args()
    generator = importlib.import_module("task_7e0986d6")
    random.seed(args.seed)
    model = scoring.sanitize_model(onnx.load(args.model))
    if model is None:
        raise RuntimeError("sanitize rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options)
    failures = []
    started = time.time()
    for index in range(args.count):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        predicted = scoring.run_network(session, benchmark["input"])
        if not np.array_equal(predicted, benchmark["output"]):
            failures.append(example)
        if (index + 1) % 500 == 0:
            print(json.dumps({
                "generated": index + 1, "failures": len(failures),
                "seconds": time.time() - started,
            }), flush=True)
    result = {
        "seed": args.seed, "total": args.count, "right": args.count - len(failures),
        "wrong": len(failures), "rate": (args.count - len(failures)) / args.count,
        "model": str(args.model),
    }
    (HERE / f"cegis_audit_{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.save_failures and failures:
        data = training.examples_to_points(failures)
        data.update({
            "generated_count": np.asarray(args.count, dtype=np.int64),
            "known_count": np.asarray(0, dtype=np.int64),
            "seed": np.asarray(args.seed, dtype=np.int64),
            "include_known": np.asarray(False, dtype=np.bool_),
        })
        np.savez_compressed(HERE / f"rank8_noise_cegis_{args.seed}.npz", **data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

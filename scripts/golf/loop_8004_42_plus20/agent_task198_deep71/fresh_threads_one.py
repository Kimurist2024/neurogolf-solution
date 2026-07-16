#!/usr/bin/env python3
"""Run the high47 generator audit with an explicit ORT thread count.

This wrapper deliberately reuses the already-reviewed task198 reference and
fresh-example driver.  Only session construction is replaced so the same
models/examples can be checked with ORT intra/inter-op thread counts 1 and 4.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "scripts/golf/loop_8004_42_plus20/agent_high47/fresh_reference_one.py"


def load_source():
    spec = importlib.util.spec_from_file_location("task198_high47_fresh", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--threads", type=int, choices=(1, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = load_source()

    def make_session(model: onnx.ModelProto, disable: bool):
        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if disable
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.intra_op_num_threads = options.inter_op_num_threads = args.threads
        options.log_severity_level = 4
        sanitized = source.scoring.sanitize_model(copy.deepcopy(model))
        return ort.InferenceSession(
            sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )

    source.make_session = make_session
    sys.argv = [
        str(SOURCE),
        "--task",
        "198",
        "--model",
        str(args.model),
        "--count",
        str(args.count),
        "--out",
        str(args.out),
    ]
    source.main()


if __name__ == "__main__":
    main()

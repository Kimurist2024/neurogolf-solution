#!/usr/bin/env python3
"""Fresh-generator dual-ORT gate for this lane's four assigned tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


REPO = Path(__file__).resolve().parents[4]
TASKS = REPO / "inputs" / "arc-gen-repo" / "tasks"
HASHES = {
    5: "045e512c",
    80: "39e1d7f9",
    101: "447fd412",
    133: "57aa92db",
}


def onehot(grid: list[list[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rows, cols = np.indices(arr.shape)
    out[0, arr, rows, cols] = 1.0
    return out


def make_session(path: Path, disabled: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES), required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=910_000)
    parser.add_argument("--mode", choices=("disabled", "default"), required=True)
    parser.add_argument(
        "--known",
        action="store_true",
        help="Run every stored train/test/arc-gen case instead of fresh generate().",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(TASKS))
    common = importlib.import_module("common")
    task = importlib.import_module(f"task_{HASHES[args.task]}")
    session = make_session(args.onnx, args.mode == "disabled")

    valid = failures = skipped = runtime_errors = unstable = 0
    first_failure: tuple[int, list[list[int]], list[list[int]]] | None = None
    seed = args.seed
    if args.known:
        data_path = REPO / "inputs" / "neurogolf-2026" / f"task{args.task:03d}.json"
        data = json.loads(data_path.read_text())
        examples = [
            example
            for split in ("train", "test", "arc-gen")
            for example in data.get(split, [])
        ]
    else:
        examples = None

    while examples is not None or valid < args.count:
        if examples is not None:
            if not examples:
                break
            example = examples.pop(0)
        else:
            random.seed(seed)
            common.random.seed(seed)
            example = task.generate()
        inp = example["input"]
        expected = example["output"]
        if len(inp) > 30 or len(inp[0]) > 30:
            skipped += 1
            seed += 1
            continue

        try:
            raw = session.run(None, {session.get_inputs()[0].name: onehot(inp)})[0]
        except Exception as exc:  # noqa: BLE001
            runtime_errors += 1
            print(f"runtime_error seed={seed}: {exc}")
            seed += 1
            valid += 1
            continue

        wanted = onehot(expected).astype(bool)
        actual = raw > 0
        if np.any((raw > 0) & (raw < 0.25)):
            unstable += 1
        if not np.array_equal(actual, wanted):
            failures += 1
            if first_failure is None:
                first_failure = (seed, inp, expected)
                diff = np.argwhere(actual != wanted)
                print(f"first_failure seed={seed} diff={diff[:12].tolist()}")
        valid += 1
        seed += 1

    stream = "known" if args.known else "fresh"
    print(
        f"task={args.task:03d} stream={stream} mode={args.mode} valid={valid} "
        f"failures={failures} runtime_errors={runtime_errors} "
        f"unstable={unstable} skipped_over30={skipped}"
    )
    return 0 if failures == runtime_errors == unstable == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

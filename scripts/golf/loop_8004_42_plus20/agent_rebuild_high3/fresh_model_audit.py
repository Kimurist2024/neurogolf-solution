#!/usr/bin/env python3
"""Multiple-seed dual-ORT generator audit for the only pre-fresh leads."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


GENERATORS = {
    18: "task_0e206a2e",
    233: "task_97a05b5b",
    286: "task_b782dc8a",
    366: "task_e6721834",
}


def session(model: onnx.ModelProto, disabled: bool) -> ort.InferenceSession:
    clean = scoring.sanitize_model(copy.deepcopy(model))
    if clean is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        clean.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def worker(arguments: tuple[int, str, int, int, int]) -> dict[str, object]:
    task, model_path, start, count, seed = arguments
    random.seed(seed)
    np.random.seed(seed % (2**32))
    generator = importlib.import_module(GENERATORS[task])
    model = onnx.load(model_path)
    sessions = {
        "disable_all": session(model, True),
        "default": session(model, False),
    }
    rows = {
        name: {"right": 0, "wrong": 0, "runtime_errors": 0, "first_wrong": []}
        for name in sessions
    }
    for offset in range(count):
        index = start + offset
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError("fresh generator unexpectedly exceeds 30")
        for name, runtime in sessions.items():
            row = rows[name]
            try:
                raw = runtime.run(
                    [runtime.get_outputs()[0].name],
                    {runtime.get_inputs()[0].name: benchmark["input"]},
                )[0]
                if np.array_equal(np.asarray(raw) > 0, benchmark["output"] > 0):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if len(row["first_wrong"]) < 10:
                        row["first_wrong"].append(index)
            except Exception as exc:  # noqa: BLE001
                row["runtime_errors"] += 1
                if len(row["first_wrong"]) < 10:
                    row["first_wrong"].append({"index": index, "error": type(exc).__name__})
    return {"seed": seed, "count": count, "modes": rows}


def run(task: int, model_path: Path, count: int, jobs: int) -> dict[str, object]:
    jobs = max(1, min(jobs, count))
    sizes = [count // jobs + (slot < count % jobs) for slot in range(jobs)]
    starts, cursor = [], 0
    for size in sizes:
        starts.append(cursor)
        cursor += size
    args = [
        (task, str(model_path), start, size, 9_000_000 + task * 1000 + slot)
        for slot, (start, size) in enumerate(zip(starts, sizes))
    ]
    with mp.get_context("spawn").Pool(jobs) as pool:
        workers = pool.map(worker, args)
    aggregate = {}
    for mode in ("disable_all", "default"):
        right = sum(item["modes"][mode]["right"] for item in workers)
        wrong = sum(item["modes"][mode]["wrong"] for item in workers)
        errors = sum(item["modes"][mode]["runtime_errors"] for item in workers)
        aggregate[mode] = {
            "total": count,
            "right": right,
            "wrong": wrong,
            "runtime_errors": errors,
            "accuracy": right / count,
        }
    return {
        "task": task,
        "model": str(model_path.relative_to(ROOT)),
        "fresh_requested": count,
        "seed_count": jobs,
        "workers": workers,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=tuple(GENERATORS))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fresh", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    result = run(args.task, args.model.resolve(), args.fresh, args.jobs)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()

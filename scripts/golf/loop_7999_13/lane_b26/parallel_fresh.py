#!/usr/bin/env python3
"""Parallel dual-ORT fresh-generator audit for B26 candidates."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402


HASHES = {328: "d22278a0", 358: "e21d9049"}


def make_session(path: str, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options)


def worker(payload: tuple[int, int, int, int, str]) -> dict[str, object]:
    task, shard, count, seed, model_path = payload
    generator = importlib.import_module(f"task_{HASHES[task]}")
    sessions = {
        "disable_all": make_session(model_path, True),
        "default": make_session(model_path, False),
    }
    row: dict[str, object] = {
        "shard": shard,
        "seed": seed,
        "requested": count,
        "generated": 0,
        "generation_errors": 0,
        "disable_all": {"right": 0, "wrong": 0, "errors": 0, "near_margin_cases": 0, "min_positive": None},
        "default": {"right": 0, "wrong": 0, "errors": 0, "near_margin_cases": 0, "min_positive": None},
    }
    for local_index in range(count):
        try:
            # Each worker is an isolated process, so the generator's use of the
            # module-global random state is deterministic without a global lock.
            random.seed(seed + local_index)
            example = generator.generate()
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise ValueError("generated grid is not scorer-convertible")
            row["generated"] += 1
        except Exception:
            row["generation_errors"] += 1
            continue
        for label, session in sessions.items():
            stats = row[label]
            try:
                raw = scoring._raw_output(session, benchmark["input"])
                positive = raw[raw > 0.0]
                if positive.size:
                    value = float(positive.min())
                    current = stats["min_positive"]
                    stats["min_positive"] = value if current is None else min(float(current), value)
                if np.any((raw > 0.0) & (raw < 0.25)):
                    stats["near_margin_cases"] += 1
                if np.array_equal(raw > 0.0, benchmark["output"].astype(bool)):
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
            except Exception:
                stats["errors"] += 1
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=sorted(HASHES))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-payload", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_payload:
        payload = json.loads(args.worker_payload)
        print(json.dumps(worker(tuple(payload))))
        return 0

    if args.task is None or args.model is None or args.seed is None or args.output is None:
        parser.error("--task, --model, --seed and --output are required")

    model = args.model.resolve()
    counts = [args.count // args.workers] * args.workers
    for index in range(args.count % args.workers):
        counts[index] += 1
    payloads = [
        (args.task, index, count, args.seed + index * 1_000_003, str(model))
        for index, count in enumerate(counts)
        if count
    ]
    # ProcessPoolExecutor uses POSIX named semaphores, which are unavailable in
    # the managed benchmark sandbox. Independent child interpreters provide the
    # same process isolation and true parallelism without that dependency.
    processes = [
        subprocess.Popen(
            [sys.executable, __file__, "--worker-payload", json.dumps(payload)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for payload in payloads
    ]
    rows: list[dict[str, object]] = []
    for process in processes:
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"audit worker failed ({process.returncode}): {stderr}")
        row = json.loads(stdout)
        rows.append(row)
        print(json.dumps({"shard": row["shard"], "generated": row["generated"], "disable_all": row["disable_all"], "default": row["default"]}), flush=True)

    aggregate: dict[str, object] = {
        "task": args.task,
        "model": str(model.relative_to(ROOT)),
        "requested": args.count,
        "workers": args.workers,
        "generated": sum(int(row["generated"]) for row in rows),
        "generation_errors": sum(int(row["generation_errors"]) for row in rows),
        "shards": sorted(rows, key=lambda row: int(row["shard"])),
    }
    for label in ("disable_all", "default"):
        values = [row[label]["min_positive"] for row in rows if row[label]["min_positive"] is not None]
        aggregate[label] = {
            key: sum(int(row[label][key]) for row in rows)
            for key in ("right", "wrong", "errors", "near_margin_cases")
        }
        aggregate[label]["min_positive"] = min(float(value) for value in values) if values else None
    aggregate["semantic_perfect"] = bool(
        aggregate["generated"] == args.count
        and aggregate["generation_errors"] == 0
        and all(
            aggregate[label]["right"] == args.count
            and aggregate[label]["wrong"] == 0
            and aggregate[label]["errors"] == 0
            for label in ("disable_all", "default")
        )
    )
    aggregate["margin_stable"] = bool(
        all(aggregate[label]["near_margin_cases"] == 0 for label in ("disable_all", "default"))
    )
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "shards"}, indent=2))
    return 0 if aggregate["semantic_perfect"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

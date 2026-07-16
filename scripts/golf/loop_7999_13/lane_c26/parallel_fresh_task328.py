#!/usr/bin/env python3
"""Parallel dual-ORT fresh audit for the unusually slow task328 Einsum."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs" / "arc-gen-repo" / "tasks"))
from lib import scoring  # noqa: E402


GENERATION_LOCK = threading.Lock()


def make_session(path: str, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disabled:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options)


def worker(payload: tuple[int, int, int, str]) -> dict[str, object]:
    shard, count, seed, model_path = payload
    generator = importlib.import_module("task_d22278a0")
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
        "disable_all": {"right": 0, "wrong": 0, "errors": 0, "near_margin": 0},
        "default": {"right": 0, "wrong": 0, "errors": 0, "near_margin": 0},
    }
    for local_index in range(count):
        try:
            # ARC-GEN's common module uses the process-global random module.
            # Keep each complete generate() call atomic and deterministically
            # reseed per case while the expensive ORT calls remain parallel.
            with GENERATION_LOCK:
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
                if np.any((raw > 0.0) & (raw < 0.25)):
                    stats["near_margin"] += 1
                if np.array_equal(raw > 0.0, benchmark["output"]):
                    stats["right"] += 1
                else:
                    stats["wrong"] += 1
            except Exception:
                stats["errors"] += 1
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=328_260_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    counts = [args.count // args.workers] * args.workers
    for index in range(args.count % args.workers):
        counts[index] += 1
    payloads = [
        (index, count, args.seed + index * 1_000_003, str(model))
        for index, count in enumerate(counts)
        if count
    ]
    rows: list[dict[str, object]] = []
    # ORT releases the GIL while executing the expensive Einsum.  Threads also
    # avoid macOS sandbox restrictions on POSIX semaphore discovery that block
    # ProcessPoolExecutor in the managed workspace.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, payload) for payload in payloads]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    aggregate: dict[str, object] = {
        "task": 328,
        "model": str(model.relative_to(ROOT)),
        "requested": args.count,
        "workers": args.workers,
        "generated": sum(int(row["generated"]) for row in rows),
        "generation_errors": sum(int(row["generation_errors"]) for row in rows),
        "disable_all": {},
        "default": {},
        "shards": sorted(rows, key=lambda row: int(row["shard"])),
    }
    for label in ("disable_all", "default"):
        aggregate[label] = {
            key: sum(int(row[label][key]) for row in rows)
            for key in ("right", "wrong", "errors", "near_margin")
        }
    aggregate["perfect"] = bool(
        aggregate["generated"] == args.count
        and aggregate["generation_errors"] == 0
        and all(
            aggregate[label]["right"] == args.count
            and aggregate[label]["wrong"] == 0
            and aggregate[label]["errors"] == 0
            and aggregate[label]["near_margin"] == 0
            for label in ("disable_all", "default")
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "shards"}, indent=2))
    return 0 if aggregate["perfect"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

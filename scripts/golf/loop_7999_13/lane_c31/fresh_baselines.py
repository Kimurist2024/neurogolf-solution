#!/usr/bin/env python3
"""Independent dual-ORT fresh-5000 control for C31 incumbents."""

from __future__ import annotations

import copy
import hashlib
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


COUNT = 5000
SEED_BASE = 31_799_130_000
TASKS = {199: "834ec97d", 212: "8d510a79"}


def make_session(path: Path, disabled: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    if model is None:
        raise RuntimeError(f"sanitize failed: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def empty_stats() -> dict[str, object]:
    return {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}


def run_task(task: int, task_hash: str) -> dict[str, object]:
    path = HERE / "baseline" / f"task{task:03d}.onnx"
    seed = SEED_BASE + task
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    generator = importlib.import_module(f"task_{task_hash}")
    sessions = {
        "disabled": make_session(path, True),
        "default": make_session(path, False),
    }
    stats = {mode: empty_stats() for mode in sessions}
    shapes: Counter[str] = Counter()
    attempts = generation_errors = conversion_skips = valid = 0
    started = time.monotonic()
    while valid < COUNT:
        attempts += 1
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            conversion_skips += 1
            continue
        valid += 1
        shapes[f"{len(example['input'])}x{len(example['input'][0])}"] += 1
        expected = benchmark["output"].astype(bool)
        for mode, sess in sessions.items():
            row = stats[mode]
            try:
                raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                actual = raw > 0.0
                if np.array_equal(actual, expected):
                    row["right"] += 1
                else:
                    row["wrong"] += 1
                    if row["first_failure"] is None:
                        row["first_failure"] = {
                            "valid_case": valid,
                            "different_cells": int(np.count_nonzero(actual != expected)),
                        }
            except Exception as exc:  # noqa: BLE001
                row["errors"] += 1
                if row["first_failure"] is None:
                    row["first_failure"] = {
                        "valid_case": valid,
                        "runtime": f"{type(exc).__name__}: {exc}",
                    }
        if valid % 500 == 0:
            print(task, valid, stats, flush=True)
    return {
        "task": task,
        "task_hash": task_hash,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": seed,
        "requested_valid": COUNT,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "grid_shapes": dict(sorted(shapes.items())),
        "modes": stats,
        "passed": all(
            row["right"] == COUNT and row["wrong"] == 0 and row["errors"] == 0
            for row in stats.values()
        ),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    result = []
    for task, task_hash in TASKS.items():
        result.append(run_task(task, task_hash))
        (HERE / "fresh_baselines.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if all(row["passed"] for row in result) else 2


if __name__ == "__main__":
    raise SystemExit(main())

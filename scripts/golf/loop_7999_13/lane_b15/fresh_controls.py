#!/usr/bin/env python3
"""Independent dual-ORT fresh5000 for the final B15 research controls."""

from __future__ import annotations

import copy
import importlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from scripts.lib.scoring import sanitize_model  # noqa: E402


CONTROLS = {
    "task023_archive1541": {
        "task": 23,
        "hash": "150deff5",
        "path": ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/task023_r03_static1541.onnx",
        "seed": 150815023,
    },
    "task036_truthful_gather": {
        "task": 36,
        "hash": "1f85a75f",
        "path": HERE / "candidate_task036_truthful_gather.onnx",
        "seed": 150815036,
    },
}
COUNT = 5000


def encode(grid: list[list[int]]) -> np.ndarray:
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            out[0, color, row, col] = 1.0
    return out


def worker(label: str, mode: str, queue: mp.Queue) -> None:
    config = CONTROLS[label]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{config['hash']}")
    model = sanitize_model(copy.deepcopy(onnx.load(config["path"])))
    if model is None:
        queue.put({"label": label, "mode": mode, "right": 0, "wrong": 0, "errors": 1, "session_error": "sanitize failed"})
        return
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    try:
        session = ort.InferenceSession(model.SerializeToString(), options)
    except Exception as exc:  # noqa: BLE001
        queue.put({"label": label, "mode": mode, "right": 0, "wrong": 0, "errors": 1, "session_error": repr(exc)})
        return

    right = wrong = errors = attempts = valid = oversize = output_shape_wrong = 0
    first_failure: dict[str, Any] | None = None
    min_nonzero_abs: float | None = None
    started = time.monotonic()
    while valid < COUNT:
        attempts += 1
        try:
            case = generator.generate()
        except Exception as exc:  # noqa: BLE001
            errors += 1
            first_failure = first_failure or {"stage": "generate", "error": repr(exc)}
            continue
        height, width = len(case["input"]), len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        try:
            raw = session.run(["output"], {"input": encode(case["input"])})[0]
            got = raw > 0.0
            want = encode(case["output"]).astype(bool)
            nonzero = np.abs(raw)[np.abs(raw) > 0]
            if nonzero.size:
                value = float(nonzero.min())
                min_nonzero_abs = value if min_nonzero_abs is None else min(min_nonzero_abs, value)
            if np.array_equal(got, want):
                right += 1
            else:
                wrong += 1
                if got.shape != want.shape:
                    output_shape_wrong += 1
                if first_failure is None:
                    first_failure = {
                        "valid_case": valid,
                        "input_shape": [height, width],
                        "got_shape": list(got.shape),
                        "want_shape": list(want.shape),
                    }
                    if got.shape == want.shape:
                        diff = np.argwhere(got != want)
                        first_failure["different_values"] = int(len(diff))
                        first_failure["first_difference"] = diff[0].tolist() if len(diff) else None
        except Exception as exc:  # noqa: BLE001
            errors += 1
            first_failure = first_failure or {"stage": "inference", "valid_case": valid, "error": repr(exc)}
    queue.put(
        {
            "label": label,
            "task": config["task"],
            "path": str(Path(config["path"]).relative_to(ROOT)),
            "mode": mode,
            "seed": seed,
            "valid": valid,
            "attempts": attempts,
            "oversize_skipped": oversize,
            "right": right,
            "wrong": wrong,
            "output_shape_wrong": output_shape_wrong,
            "errors": errors,
            "accuracy": right / valid,
            "min_nonzero_abs": min_nonzero_abs,
            "first_failure": first_failure,
            "elapsed_seconds": time.monotonic() - started,
        }
    )


def main() -> int:
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [
        context.Process(target=worker, args=(label, mode, queue))
        for label in CONTROLS
        for mode in ("disabled", "default")
    ]
    for process in processes:
        process.start()
    rows = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    rows.sort(key=lambda row: (row["label"], row["mode"]))
    (HERE / "fresh5000_controls_dual_ort.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    truthful = [row for row in rows if row["label"] == "task036_truthful_gather"]
    return 0 if all(row["right"] == COUNT and row["wrong"] == 0 and row["errors"] == 0 for row in truthful) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Actual-cost, dual-known, and dual-fresh audit of exact B15 baselines."""

from __future__ import annotations

import copy
import importlib
import json
import multiprocessing as mp
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from scripts.lib import scoring  # noqa: E402


HASHES = {23: "150deff5", 36: "1f85a75f"}
MODES = ("disabled", "default")
COUNT = 5000
SEED = 150799913


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def session(task: int, mode: str) -> ort.InferenceSession:
    model = scoring.sanitize_model(
        copy.deepcopy(onnx.load(HERE / "baseline" / f"task{task:03d}.onnx"))
    )
    if model is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    if mode == "disabled":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(model.SerializeToString(), options)


def actual_costs() -> dict[str, Any]:
    ort.set_default_logger_severity(4)
    result: dict[str, Any] = {}
    for task in HASHES:
        model = onnx.load(HERE / "baseline" / f"task{task:03d}.onnx")
        with tempfile.TemporaryDirectory(prefix=f"b15_{task:03d}_", dir="/tmp") as workdir:
            result[str(task)] = scoring.score_and_verify(
                model,
                task,
                workdir,
                label="exact",
                require_correct=False,
            )
    (HERE / "exact_costs.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def known(task: int, mode: str) -> dict[str, Any]:
    sess = session(task, mode)
    examples = scoring.load_examples(task)
    right = wrong = errors = skipped = 0
    first_failure: dict[str, Any] | None = None
    min_nonzero_abs: float | None = None
    started = time.monotonic()
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[subset]):
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                skipped += 1
                continue
            try:
                raw = sess.run(["output"], {"input": benchmark["input"]})[0]
                got = raw > 0.0
                want = benchmark["output"].astype(bool)
                abs_raw = np.abs(raw)
                nz = abs_raw[abs_raw > 0]
                if nz.size:
                    value = float(nz.min())
                    min_nonzero_abs = value if min_nonzero_abs is None else min(min_nonzero_abs, value)
                if np.array_equal(got, want):
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        if got.shape != want.shape:
                            first_failure = {
                                "subset": subset,
                                "index": index,
                                "kind": "output_shape",
                                "got_shape": list(got.shape),
                                "want_shape": list(want.shape),
                            }
                        else:
                            diff = np.argwhere(got != want)
                            first_failure = {
                                "subset": subset,
                                "index": index,
                                "different_values": int(len(diff)),
                                "first_difference": diff[0].tolist() if len(diff) else None,
                            }
            except Exception as exc:  # noqa: BLE001 - runtime errors are audit data
                errors += 1
                first_failure = first_failure or {
                    "subset": subset,
                    "index": index,
                    "error": repr(exc),
                }
    return {
        "task": task,
        "mode": mode,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped": skipped,
        "min_nonzero_abs": min_nonzero_abs,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - started,
    }


def fresh(task: int, mode: str, queue: mp.Queue) -> None:
    random.seed(SEED)
    np.random.seed(SEED & 0xFFFFFFFF)
    generator = importlib.import_module(f"task_{HASHES[task]}")
    sess = session(task, mode)
    valid = attempts = right = wrong = errors = oversize = output_shape_wrong = 0
    first_failure: dict[str, Any] | None = None
    min_nonzero_abs: float | None = None
    started = time.monotonic()
    while valid < COUNT:
        attempts += 1
        try:
            case = generator.generate()
        except Exception as exc:  # noqa: BLE001 - generator errors are audit data
            errors += 1
            first_failure = first_failure or {"stage": "generate", "error": repr(exc)}
            continue
        height, width = len(case["input"]), len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        try:
            raw = sess.run(["output"], {"input": encode(case["input"])})[0]
            got = raw > 0.0
            want = encode(case["output"]).astype(bool)
            abs_raw = np.abs(raw)
            nz = abs_raw[abs_raw > 0]
            if nz.size:
                value = float(nz.min())
                min_nonzero_abs = value if min_nonzero_abs is None else min(min_nonzero_abs, value)
            if np.array_equal(got, want):
                right += 1
            else:
                wrong += 1
                if got.shape != want.shape:
                    output_shape_wrong += 1
                if first_failure is None:
                    if got.shape != want.shape:
                        first_failure = {
                            "valid_case": valid,
                            "attempt": attempts,
                            "shape": [height, width],
                            "kind": "output_shape",
                            "got_shape": list(got.shape),
                            "want_shape": list(want.shape),
                        }
                    else:
                        diff = np.argwhere(got != want)
                        first_failure = {
                            "valid_case": valid,
                            "attempt": attempts,
                            "shape": [height, width],
                            "different_values": int(len(diff)),
                            "first_difference": diff[0].tolist() if len(diff) else None,
                        }
        except Exception as exc:  # noqa: BLE001 - runtime errors are audit data
            errors += 1
            first_failure = first_failure or {
                "stage": "inference",
                "valid_case": valid,
                "error": repr(exc),
            }
    queue.put(
        {
            "task": task,
            "mode": mode,
            "seed": SEED,
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
    costs = actual_costs()
    print("COSTS", json.dumps(costs), flush=True)
    known_rows = [known(task, mode) for task in HASHES for mode in MODES]
    (HERE / "known_dual_ort.json").write_text(json.dumps(known_rows, indent=2) + "\n")
    print("KNOWN_COMPLETE", json.dumps(known_rows), flush=True)
    if any(row["wrong"] or row["errors"] for row in known_rows):
        return 2

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [
        context.Process(target=fresh, args=(task, mode, queue))
        for task in HASHES
        for mode in MODES
    ]
    for process in processes:
        process.start()
    rows = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    rows.sort(key=lambda row: (row["task"], row["mode"]))
    (HERE / "fresh5000_dual_ort.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("FRESH_COMPLETE", json.dumps(rows), flush=True)
    summary = {
        "costs": costs,
        "known": known_rows,
        "fresh": rows,
        "runtime_errors": sum(row["errors"] for row in known_rows + rows),
    }
    (HERE / "baseline_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["runtime_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

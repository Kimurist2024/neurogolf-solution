#!/usr/bin/env python3
"""Dual-ORT known and fresh-generator audit of the exact B14 baselines.

This is deliberately a read-only validator for the two extracted models.  It
does not promote either model and never writes the project-level ZIP/CSV/ledger.
Known examples are completed before the four fresh jobs are started.
"""

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
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from scripts.lib import scoring  # noqa: E402


HASHES = {5: "045e512c", 80: "39e1d7f9"}
MODES = ("disabled", "default")
FRESH_COUNT = 5000
FRESH_SEED = 140799913


def encode(grid: list[list[int]]) -> np.ndarray:
    result = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for row, values in enumerate(grid):
        for col, color in enumerate(values):
            result[0, color, row, col] = 1.0
    return result


def make_session(task: int, mode: str) -> ort.InferenceSession:
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
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options)


def compare_raw(raw: np.ndarray, want: np.ndarray) -> tuple[bool, float | None]:
    mask = raw > 0.0
    nonzero = np.abs(raw)[np.abs(raw) > 0.0]
    margin = float(nonzero.min()) if nonzero.size else None
    return bool(np.array_equal(mask, want)), margin


def known_audit(task: int, mode: str) -> dict[str, Any]:
    session = make_session(task, mode)
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
                raw = session.run(["output"], {"input": benchmark["input"]})[0]
                ok, margin = compare_raw(raw, benchmark["output"].astype(bool))
                if margin is not None:
                    min_nonzero_abs = (
                        margin
                        if min_nonzero_abs is None
                        else min(min_nonzero_abs, margin)
                    )
                if ok:
                    right += 1
                else:
                    wrong += 1
                    if first_failure is None:
                        diff = np.argwhere((raw > 0.0) != benchmark["output"].astype(bool))
                        first_failure = {
                            "subset": subset,
                            "index": index,
                            "different_values": int(len(diff)),
                            "first_difference": diff[0].tolist() if len(diff) else None,
                        }
            except Exception as exc:  # noqa: BLE001 - audit must record every ORT error
                errors += 1
                first_failure = first_failure or {
                    "subset": subset,
                    "index": index,
                    "stage": "inference",
                    "error": repr(exc),
                }
    return {
        "task": task,
        "mode": mode,
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "skipped_oversize": skipped,
        "accuracy": right / (right + wrong + errors),
        "min_nonzero_abs": min_nonzero_abs,
        "first_failure": first_failure,
        "elapsed_seconds": time.monotonic() - started,
    }


def fresh_audit(task: int, mode: str, queue: mp.Queue) -> None:
    random.seed(FRESH_SEED)
    np.random.seed(FRESH_SEED & 0xFFFFFFFF)
    module = importlib.import_module(f"task_{HASHES[task]}")
    session = make_session(task, mode)
    valid = attempts = right = wrong = errors = oversize = 0
    first_failure: dict[str, Any] | None = None
    min_nonzero_abs: float | None = None
    started = time.monotonic()
    while valid < FRESH_COUNT:
        attempts += 1
        try:
            case = module.generate()
        except Exception as exc:  # noqa: BLE001 - generator errors are evidence
            errors += 1
            first_failure = first_failure or {
                "stage": "generate",
                "attempt": attempts,
                "error": repr(exc),
            }
            continue
        height, width = len(case["input"]), len(case["input"][0])
        if max(height, width) > 30:
            oversize += 1
            continue
        valid += 1
        try:
            raw = session.run(["output"], {"input": encode(case["input"])})[0]
            ok, margin = compare_raw(raw, encode(case["output"]).astype(bool))
            if margin is not None:
                min_nonzero_abs = (
                    margin if min_nonzero_abs is None else min(min_nonzero_abs, margin)
                )
            if ok:
                right += 1
            else:
                wrong += 1
                if first_failure is None:
                    diff = np.argwhere((raw > 0.0) != encode(case["output"]).astype(bool))
                    first_failure = {
                        "valid_case": valid,
                        "attempt": attempts,
                        "shape": [height, width],
                        "different_values": int(len(diff)),
                        "first_difference": diff[0].tolist() if len(diff) else None,
                    }
        except Exception as exc:  # noqa: BLE001 - audit must record every ORT error
            errors += 1
            first_failure = first_failure or {
                "stage": "inference",
                "valid_case": valid,
                "attempt": attempts,
                "error": repr(exc),
            }
    queue.put(
        {
            "task": task,
            "mode": mode,
            "seed": FRESH_SEED,
            "requested_valid": FRESH_COUNT,
            "valid": valid,
            "attempts": attempts,
            "oversize_skipped": oversize,
            "right": right,
            "wrong": wrong,
            "errors": errors,
            "accuracy": right / valid,
            "min_nonzero_abs": min_nonzero_abs,
            "first_failure": first_failure,
            "elapsed_seconds": time.monotonic() - started,
        }
    )


def main() -> int:
    known = [known_audit(task, mode) for task in HASHES for mode in MODES]
    (HERE / "known_dual_ort.json").write_text(json.dumps(known, indent=2) + "\n")
    print("KNOWN_COMPLETE", json.dumps(known), flush=True)
    if any(row["wrong"] or row["errors"] for row in known):
        raise SystemExit("known dual-ORT audit failed")

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes = [
        context.Process(target=fresh_audit, args=(task, mode, queue))
        for task in HASHES
        for mode in MODES
    ]
    for process in processes:
        process.start()
    fresh = [queue.get() for _ in processes]
    for process in processes:
        process.join()
    fresh.sort(key=lambda row: (row["task"], row["mode"]))
    (HERE / "fresh5000_dual_ort.json").write_text(json.dumps(fresh, indent=2) + "\n")
    print("FRESH_COMPLETE", json.dumps(fresh), flush=True)

    result = {
        "known": known,
        "fresh": fresh,
        "runtime_errors": sum(row["errors"] for row in known + fresh),
        "fresh_acceptance": {
            str(task): all(
                row["accuracy"] >= 0.95 and row["errors"] == 0
                for row in fresh
                if row["task"] == task
            )
            for task in HASHES
        },
    }
    (HERE / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["runtime_errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

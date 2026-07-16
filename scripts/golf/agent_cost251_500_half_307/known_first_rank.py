#!/usr/bin/env python3
"""One-configuration known-set screen for provisional strict-cost leads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INFILE = HERE / "cost_first_rank.json"
OUT = HERE / "known_first_rank.json"
MODELS = HERE / "profile_models"

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def session(model: onnx.ModelProto) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(sanitized.SerializeToString(), options,
                                providers=["CPUExecutionProvider"])


def known(task: int):
    result = []
    examples = scoring.load_examples(task)
    for split in ("train", "test", "arc-gen"):
        for example in examples[split]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is not None:
                result.append(benchmark)
    return result


def evaluate(runtime: ort.InferenceSession, cases) -> dict[str, object]:
    right = wrong = errors = nonfinite = shape = small = 0
    first_wrong = first_error = None
    for index, benchmark in enumerate(cases):
        try:
            raw = scoring._raw_output(runtime, benchmark["input"])
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_error is None:
                first_error = {"index": index, "type": type(exc).__name__,
                               "message": str(exc)}
            continue
        if tuple(raw.shape) != (1, 10, 30, 30):
            shape += 1
        if not np.all(np.isfinite(raw)):
            nonfinite += 1
            continue
        small += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        predicted = (raw > 0).astype(np.float32)
        if np.array_equal(predicted, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if first_wrong is None:
                first_wrong = {"index": index,
                               "mismatch_cells": int(np.count_nonzero(
                                   predicted != benchmark["output"]))}
    total = right + wrong + errors + nonfinite
    return {"total": total, "right": right, "wrong": wrong, "errors": errors,
            "nonfinite_cases": nonfinite, "shape_mismatches": shape,
            "small_positive_elements_0_to_0_25": small,
            "accuracy": right / total if total else None,
            "first_wrong": first_wrong, "first_error": first_error}


def worker(task: int, model_path: Path) -> int:
    result = evaluate(session(onnx.load(model_path)), known(task))
    print(json.dumps(result))
    return 0


def screen(row: dict[str, object]) -> dict[str, object]:
    task = int(row["task"])
    sha = str(row["sha256"])
    path = MODELS / f"task{task:03d}_{sha}.onnx"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--task", str(task), "--model", str(path)]
    try:
        completed = subprocess.run(command, text=True, capture_output=True,
                                   timeout=25, check=False)
    except subprocess.TimeoutExpired:
        return {**row, "known_status": "timeout", "known_disabled_t1": None}
    if completed.returncode != 0:
        return {**row, "known_status": "error", "known_disabled_t1": None,
                "known_stderr": completed.stderr[-2000:]}
    try:
        measured = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {**row, "known_status": "parse_error", "known_disabled_t1": None,
                "known_stdout": completed.stdout[-2000:],
                "known_stderr": completed.stderr[-2000:]}
    clean = (measured["errors"] == measured["nonfinite_cases"] == 0
             and measured["shape_mismatches"] == 0
             and measured["small_positive_elements_0_to_0_25"] == 0)
    return {**row, "known_status": "ok", "known_disabled_t1": measured,
            "known_clean": clean, "known_policy95": clean and measured["accuracy"] >= 0.95,
            "known_exact": clean and measured["accuracy"] == 1.0}


def orchestrate(workers: int) -> int:
    source = json.loads(INFILE.read_text())
    rows = source["provisional_strict_lower"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(screen, row): row for row in rows}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index % 20 == 0 or result.get("known_policy95"):
                known_row = result.get("known_disabled_t1") or {}
                print(json.dumps({"i": index, "n": len(rows), "task": result["task"],
                                  "cost": result["provisional_profile"]["cost"],
                                  "status": result["known_status"],
                                  "accuracy": known_row.get("accuracy"),
                                  "policy95": result.get("known_policy95", False)}),
                      flush=True)
    results.sort(key=lambda row: (int(row["task"]),
                                  int(row["provisional_profile"]["cost"]),
                                  str(row["sha256"])))
    survivors = [row for row in results if row.get("known_policy95")]
    payload = {"source": str(INFILE.relative_to(ROOT)), "candidate_count": len(rows),
               "workers": workers, "timeout_seconds": 25,
               "known_policy95_count": len(survivors),
               "known_policy95": survivors, "results": results,
               "protected_writes": "none; lane evidence only"}
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(rows), "known_policy95": len(survivors),
                      "evidence": str(OUT.relative_to(ROOT))}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task", type=int)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        if args.task is None or args.model is None:
            parser.error("worker requires --task and --model")
        return worker(args.task, args.model)
    return orchestrate(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())

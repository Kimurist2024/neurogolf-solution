#!/usr/bin/env python3
"""Independent known/fresh audit for the exact C9 ZIP members."""

from __future__ import annotations

import copy
import argparse
import hashlib
import importlib
import json
import random
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


TASKS = {
    310: "c909285e",
    330: "d2abd087",
    340: "d687bc17",
    354: "ddf7fa4f",
    361: "e40b9e2f",
    364: "e509e548",
    368: "e76a88a6",
}
COUNT = 5000
SEED_BASE = 7_999_130_900


def make_session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    assert model is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_complete(task: int, session: ort.InferenceSession) -> dict[str, object]:
    examples = scoring.load_examples(task)
    subsets: dict[str, dict[str, int]] = {}
    for name in ("train", "test", "arc-gen"):
        right = wrong = errors = 0
        for example in examples[name]:
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                continue
            try:
                actual = session.run(["output"], {"input": benchmark["input"]})[0] > 0
                if np.array_equal(actual, benchmark["output"] > 0):
                    right += 1
                else:
                    wrong += 1
            except Exception:  # noqa: BLE001
                errors += 1
        subsets[name] = {"right": right, "wrong": wrong, "errors": errors}
    return subsets


def fresh(
    task: int,
    task_hash: str,
    session: ort.InferenceSession,
    disable_all: bool,
) -> dict[str, object]:
    seed = SEED_BASE + 10_000 * task + (0 if disable_all else 1)
    random.seed(seed)
    generator = importlib.import_module(f"task_{task_hash}")
    correct = wrong = errors = 0
    first_failure: dict[str, object] | None = None
    for index in range(COUNT):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = session.run(["output"], {"input": benchmark["input"]})[0] > 0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "runtime",
                    "error": repr(exc),
                    "input_shape": list(np.asarray(example["input"]).shape),
                }
            continue
        expected = benchmark["output"] > 0
        if np.array_equal(actual, expected):
            correct += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "wrong",
                    "input_shape": list(np.asarray(example["input"]).shape),
                    "output_shape": list(np.asarray(example["output"]).shape),
                    "differing_elements": int(np.count_nonzero(actual != expected)),
                }
    return {
        "seed": seed,
        "total": COUNT,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "first_failure": first_failure,
        "ort_disable_all": disable_all,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", type=int, default=list(TASKS))
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    output_path = HERE / "baseline_audit.json"
    results: dict[str, object] = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else {}
    )
    for task in args.tasks:
        task_hash = TASKS[task]
        path = HERE / "base" / f"task{task:03d}.onnx"
        record: dict[str, object] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        try:
            disabled = make_session(path, True)
            record["known_disable_all"] = known_complete(task, disabled)
            record["fresh_disable_all"] = fresh(task, task_hash, disabled, True)
        except Exception as exc:  # noqa: BLE001
            record["disable_all_session_error"] = repr(exc)
        try:
            default = make_session(path, False)
            record["known_default_ort"] = known_complete(task, default)
            record["fresh_default_ort"] = fresh(task, task_hash, default, False)
        except Exception as exc:  # noqa: BLE001
            record["default_ort_session_error"] = repr(exc)
        results[str(task)] = record
        print(task, json.dumps(record, sort_keys=True), flush=True)
        output_path.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    output_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

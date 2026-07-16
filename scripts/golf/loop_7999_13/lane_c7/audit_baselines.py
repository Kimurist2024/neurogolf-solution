#!/usr/bin/env python3
"""Independent raw-generator audit of the seven exact C7 baseline members."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


TASKS = {
    29: "1c786137",
    91: "3f7978a0",
    301: "beb8660c",
    316: "cdecee7f",
    341: "d6ad076f",
    355: "de1cd16c",
    357: "e179c5f4",
}
DISABLED_COUNT = 3000
DEFAULT_COUNT = 100
SEED_BASE = 799_913_000


def make_session(path: Path, disable: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disable:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run(task: int, task_hash: str, disable: bool, count: int) -> dict[str, object]:
    path = HERE / "base" / f"task{task:03d}.onnx"
    random.seed(SEED_BASE + task + (0 if disable else 1_000_000))
    generator = importlib.import_module(f"task_{task_hash}")
    examples = [generator.generate() for _ in range(count)]
    try:
        session = make_session(path, disable)
    except Exception as exc:  # noqa: BLE001
        return {"correct": 0, "wrong": 0, "errors": count, "session_error": repr(exc)}
    correct = wrong = errors = 0
    first_failure = None
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            output = session.run(["output"], {"input": benchmark["input"]})[0] > 0.0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
            continue
        expected = benchmark["output"] > 0.0
        if np.array_equal(output, expected):
            correct += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "wrong",
                    "input_shape": list(np.asarray(example["input"]).shape),
                    "output_shape": list(np.asarray(example["output"]).shape),
                    "differing_elements": int(np.count_nonzero(output != expected)),
                }
    return {
        "seed": SEED_BASE + task + (0 if disable else 1_000_000),
        "total": count,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "first_failure": first_failure,
        "ort_disable_all": disable,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {}
    for task, task_hash in TASKS.items():
        path = HERE / "base" / f"task{task:03d}.onnx"
        result[str(task)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fresh_disable_all": run(task, task_hash, True, DISABLED_COUNT),
            "fresh_default_ort": run(task, task_hash, False, DEFAULT_COUNT),
        }
        print(task, result[str(task)]["fresh_disable_all"], result[str(task)]["fresh_default_ort"])
    (HERE / "baseline_fresh_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

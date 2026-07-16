#!/usr/bin/env python3
"""Fresh-generator audit of the exact C24 baseline members in both ORT modes."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASK_DIR = ROOT / "inputs" / "arc-gen-repo" / "tasks"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TASK_DIR))

from scripts.lib import scoring  # noqa: E402


def load_generator(name: str):
    path = TASK_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"c24_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATORS = {
    363: load_generator("task_e5062a87"),
    388: load_generator("task_f5b8619d"),
}


@contextmanager
def suppress_native_stderr():
    """Suppress the expected ORT shape-cloak warnings without hiding results."""
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(null)


def make_session(task: int, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(
        copy.deepcopy(onnx.load(HERE / "base" / f"task{task}.onnx"))
    )
    if model is None:
        raise RuntimeError("model sanitization failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def audit(task: int, count: int) -> dict[str, object]:
    with suppress_native_stderr():
        sessions = {
            "disable_all": make_session(task, True),
            "default": make_session(task, False),
        }
        totals = {
            mode: {"right": 0, "wrong": 0, "errors": 0, "first_failures": []}
            for mode in sessions
        }
        for seed in range(count):
            random.seed(seed)
            example = GENERATORS[task].generate()
            benchmark = scoring.convert_to_numpy(example)
            if benchmark is None:
                raise AssertionError(f"conversion failed task={task} seed={seed}")
            for mode, session in sessions.items():
                try:
                    raw = session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    if np.array_equal(raw > 0, benchmark["output"] > 0):
                        totals[mode]["right"] += 1
                    else:
                        totals[mode]["wrong"] += 1
                        if len(totals[mode]["first_failures"]) < 20:
                            totals[mode]["first_failures"].append(seed)
                except Exception as exc:  # noqa: BLE001
                    totals[mode]["errors"] += 1
                    if len(totals[mode]["first_failures"]) < 20:
                        totals[mode]["first_failures"].append(
                            {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                        )
    return {"task": task, "count": count, "modes": totals}


def main() -> None:
    output = {f"task{task}": audit(task, 5000) for task in (363, 388)}
    (HERE / "fresh_exact_audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    for task, record in output.items():
        print(task, record["modes"])


if __name__ == "__main__":
    main()

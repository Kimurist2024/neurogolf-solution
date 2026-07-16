#!/usr/bin/env python3
"""Independent dual-runtime fresh-5000 audit of the only cheap truthful task134 lead."""

from __future__ import annotations

import copy
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


TASK = 134
TASK_HASH = "5ad4f10b"
CANDIDATE = (
    ROOT / "scripts/golf/loop_7999_13/lane_archive_top200/task134_r04_static320.onnx"
)
COUNT = 5000
SEEDS = {True: 13_400_799_913, False: 13_400_799_914}


def make_session(disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(CANDIDATE)))
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


def run(disable_all: bool) -> dict[str, object]:
    seed = SEEDS[disable_all]
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    generator = importlib.import_module(f"task_{TASK_HASH}")
    session = make_session(disable_all)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    correct = wrong = errors = 0
    first_failure = None
    for index in range(COUNT):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = session.run([output_name], {input_name: benchmark["input"]})[0] > 0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {
                    "index": index,
                    "kind": "runtime",
                    "error": repr(exc),
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
                    "actual_shape": list(actual.shape),
                    "differing_elements": int(np.count_nonzero(actual != expected))
                    if actual.shape == expected.shape
                    else None,
                }
    return {
        "seed": seed,
        "ort_disable_all": disable_all,
        "total": COUNT,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "first_failure": first_failure,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    result = {
        "task": TASK,
        "path": str(CANDIDATE.relative_to(ROOT)),
        "sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
    }
    for disable_all, label in ((True, "disable_all"), (False, "default")):
        result[label] = run(disable_all)
        (HERE / "task134_r04_fresh5000.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        row = result[label]
        print(label, row["correct"], row["wrong"], row["errors"], flush=True)


if __name__ == "__main__":
    main()

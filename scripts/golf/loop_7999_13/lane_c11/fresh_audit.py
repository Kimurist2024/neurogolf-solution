#!/usr/bin/env python3
"""Independent dual-runtime fresh-5000 audit for C11 survivors."""

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


COUNT = 5000
SEED_BASE = 11_799_130_000
CANDIDATES = {
    "task090_r06": (90, "3eda0437", HERE / "candidates" / "task090_r06_static418.onnx"),
    "task090_r07": (90, "3eda0437", HERE / "candidates" / "task090_r07_static430.onnx"),
    "task143_r02": (143, "63613498", HERE / "candidates" / "task143_r02_static148.onnx"),
}


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


def run(
    label: str,
    task: int,
    task_hash: str,
    path: Path,
    disable_all: bool,
    index: int,
) -> dict[str, object]:
    seed = SEED_BASE + task * 10_000 + index * 10 + (0 if disable_all else 1)
    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    generator = importlib.import_module(f"task_{task_hash}")
    session = make_session(path, disable_all)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    correct = wrong = errors = 0
    first_failure: dict[str, object] | None = None
    for fresh_index in range(COUNT):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = session.run([output_name], {input_name: benchmark["input"]})[0] > 0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {
                    "index": fresh_index,
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
                    "index": fresh_index,
                    "kind": "wrong",
                    "input_shape": list(np.asarray(example["input"]).shape),
                    "output_shape": list(np.asarray(example["output"]).shape),
                    "actual_shape": list(actual.shape),
                    "differing_elements": int(np.count_nonzero(actual != expected))
                    if actual.shape == expected.shape
                    else None,
                }
    return {
        "label": label,
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ort_disable_all": disable_all,
        "seed": seed,
        "total": COUNT,
        "correct": correct,
        "wrong": wrong,
        "errors": errors,
        "first_failure": first_failure,
    }


def main() -> None:
    ort.set_default_logger_severity(4)
    output_path = HERE / "fresh_audit.json"
    results: dict[str, object] = {}
    for index, (label, (task, task_hash, path)) in enumerate(CANDIDATES.items()):
        record: dict[str, object] = {}
        for disable_all, key in ((True, "disable_all"), (False, "default")):
            record[key] = run(label, task, task_hash, path, disable_all, index)
            print(
                label,
                key,
                record[key]["correct"],
                record[key]["wrong"],
                record[key]["errors"],
                flush=True,
            )
            results[label] = record
            output_path.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()

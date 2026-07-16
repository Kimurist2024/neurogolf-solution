#!/usr/bin/env python3
"""Audit the closest source-derived/sound reference families for C8."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


MODELS = {
    54: ("264363fd", HERE / "sound_references/task054_spec_rebuild_overdraw.onnx"),
    209: ("8a004b2b", HERE / "sound_references/task209_fp16_rebuild.onnx"),
    367: ("e73095fd", HERE / "sound_references/task367_bitdiff.onnx"),
}
COUNT = 5000
SEED_BASE = 8_888_130_000


def session(path: Path, disable_all: bool) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load(path)))
    assert model is not None
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known(task: int, sess: ort.InferenceSession) -> dict[str, int]:
    right = wrong = errors = 0
    examples = scoring.load_examples(task)
    for example in examples["train"] + examples["test"] + examples["arc-gen"]:
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = sess.run(["output"], {"input": benchmark["input"]})[0] > 0
            if np.array_equal(actual, benchmark["output"] > 0):
                right += 1
            else:
                wrong += 1
        except Exception:  # noqa: BLE001
            errors += 1
    return {"right": right, "wrong": wrong, "errors": errors}


def fresh(
    task: int,
    task_hash: str,
    sess: ort.InferenceSession,
    disable_all: bool,
) -> dict[str, object]:
    seed = SEED_BASE + task * 10_000 + (0 if disable_all else 1)
    random.seed(seed)
    generator = importlib.import_module(f"task_{task_hash}")
    correct = wrong = errors = 0
    first_failure = None
    for index in range(COUNT):
        example = generator.generate()
        benchmark = scoring.convert_to_numpy(example)
        assert benchmark is not None
        try:
            actual = sess.run(["output"], {"input": benchmark["input"]})[0] > 0
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_failure is None:
                first_failure = {"index": index, "kind": "runtime", "error": repr(exc)}
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
    ort.set_default_logger_severity(4)
    result: dict[str, object] = {}
    for task, (task_hash, path) in MODELS.items():
        record: dict[str, object] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with tempfile.TemporaryDirectory(prefix="c8_ref_score_", dir="/tmp") as workdir:
            record["score"] = scoring.score_and_verify(
                onnx.load(path), task, workdir, label="c8ref", require_correct=False
            )
        try:
            disabled = session(path, True)
            record["known_disable_all"] = known(task, disabled)
            record["fresh_disable_all"] = fresh(task, task_hash, disabled, True)
        except Exception as exc:  # noqa: BLE001
            record["disable_all_session_error"] = repr(exc)
        try:
            default = session(path, False)
            record["fresh_default_ort"] = fresh(task, task_hash, default, False)
        except Exception as exc:  # noqa: BLE001
            record["default_ort_session_error"] = repr(exc)
        result[str(task)] = record
        print(task, record)
    (HERE / "sound_reference_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

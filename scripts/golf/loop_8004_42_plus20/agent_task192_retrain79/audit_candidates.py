#!/usr/bin/env python3
"""Audit task192 candidates on shared, reproducible fresh-generator sets.

The task generator is comparatively slow because it rejects overlapping box
layouts.  This auditor therefore generates each seed's examples once and runs
every candidate on the identical cases.  Archive members can be addressed as
``archive.zip::task192.onnx`` without extracting or modifying the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))

from lib import scoring  # noqa: E402


DEFAULT_MODELS = (
    "submission_base_8005.17.zip::task192.onnx",
    "artifacts/submission_h7904.zip::task192.onnx",
    "scripts/golf/loop_8004_42_plus20/agent_rebuild/candidates/task192_true_local_lp_k4.onnx",
    "scripts/golf/loop_8004_42_plus20/agent_rebuild/candidates/task192_true_local_lp_k5.onnx",
    "scripts/golf/loop_7999_13/lane_c35/candidates/task192_r01_static403.onnx",
    "scripts/golf/loop_7999_13/lane_c35/candidates/task192_r02_static493.onnx",
    "scripts/golf/loop_7999_13/lane_c35/candidates/task192_r03_static509.onnx",
    "scripts/golf/loop_7999_13/lane_c35/candidates/task192_r04_static561.onnx",
    "scripts/golf/loop_7999_13/lane_c35/candidates/task192_r05_static589.onnx",
)


def load_bytes(spec: str) -> bytes:
    if "::" not in spec:
        return (ROOT / spec).read_bytes()
    archive, member = spec.split("::", 1)
    with zipfile.ZipFile(ROOT / archive) as handle:
        return handle.read(member)


def session(model: onnx.ModelProto, optimized: bool) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected candidate")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def shape_of(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    actual_input = np.asarray(example["input"], dtype=np.int64)
    actual_output = np.asarray(example["output"], dtype=np.int64)
    height, width = actual_input.shape
    input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for color in range(10):
        input_tensor[0, color, :height, :width] = actual_input == color
    expected = np.zeros((1, 10, 30, 30), dtype=np.bool_)
    for color in range(10):
        expected[0, color, :height, :width] = actual_output == color
    # Outside the logical grid must be all-zero, not background one-hot.
    expected[:, 0, height:, :] = False
    expected[:, 0, :, width:] = False
    return input_tensor, expected


def run_examples(
    inference: ort.InferenceSession, examples: list[dict[str, object]]
) -> dict[str, object]:
    right = wrong = errors = 0
    differing = 0
    first_wrong = None
    min_true = float("inf")
    max_false = float("-inf")
    for index, example in enumerate(examples):
        input_tensor, expected = shape_of(example)
        try:
            raw = inference.run(None, {"input": input_tensor})[0]
        except Exception as exc:  # noqa: BLE001
            errors += 1
            if first_wrong is None:
                first_wrong = {"index": index, "runtime_error": repr(exc)}
            continue
        predicted = raw > 0
        mismatch = int(np.count_nonzero(predicted != expected))
        if mismatch:
            wrong += 1
            differing += mismatch
            if first_wrong is None:
                first_wrong = {
                    "index": index,
                    "differing_elements": mismatch,
                    "shape": list(np.asarray(example["input"]).shape),
                }
        else:
            right += 1
        if np.any(expected):
            min_true = min(min_true, float(np.min(raw[expected])))
        if np.any(~expected):
            max_false = max(max_false, float(np.max(raw[~expected])))
    total = len(examples)
    return {
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "total": total,
        "accuracy": right / total,
        "differing_elements": differing,
        "first_wrong": first_wrong,
        "min_true": min_true,
        "max_false": max_false,
    }


def score(model: onnx.ModelProto, label: str) -> dict[str, object] | None:
    with tempfile.TemporaryDirectory() as directory:
        result = scoring.score_and_verify(
            model, 192, directory, label=label, require_correct=False
        )
    if result is None:
        return None
    return {
        key: result.get(key)
        for key in ("memory", "params", "cost", "score", "correct")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seeds", default="79192901,79192902")
    parser.add_argument("--fresh", type=int, default=500)
    parser.add_argument("--output", type=Path, default=HERE / "candidate_audit.json")
    args = parser.parse_args()

    generator = importlib.import_module("task_7e0986d6")
    known_data = json.loads((ROOT / "inputs/neurogolf-2026/task192.json").read_text())
    known = known_data["train"] + known_data["test"] + known_data["arc-gen"]
    seed_values = [int(item) for item in args.seeds.split(",")]
    fresh_sets: dict[int, list[dict[str, object]]] = {}
    generation_errors: dict[int, int] = {}
    for seed in seed_values:
        random.seed(seed)
        examples = []
        errors = 0
        while len(examples) < args.fresh:
            try:
                examples.append(generator.generate())
            except Exception:  # noqa: BLE001
                errors += 1
        fresh_sets[seed] = examples
        generation_errors[seed] = errors
        print(f"generated seed={seed} count={len(examples)} errors={errors}", flush=True)

    rows = []
    for position, spec in enumerate(args.models):
        blob = load_bytes(spec)
        model = onnx.load_from_string(blob)
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        row: dict[str, object] = {
            "spec": spec,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "file_bytes": len(blob),
            "ops": dict(Counter(node.op_type for node in model.graph.node)),
            "score": score(model, f"audit{position}"),
            "known": {},
            "fresh": {},
        }
        for mode, optimized in (("disabled", False), ("default", True)):
            inference = session(model, optimized)
            row["known"][mode] = run_examples(inference, known)
            row["fresh"][mode] = {
                str(seed): run_examples(inference, fresh_sets[seed])
                for seed in seed_values
            }
        row["fresh_two_seed_min_accuracy"] = min(
            details["accuracy"]
            for mode in row["fresh"].values()
            for details in mode.values()
        )
        row["passes_two_seed_90_all_modes"] = (
            row["fresh_two_seed_min_accuracy"] >= 0.9
            and all(
                details["errors"] == 0
                for mode in row["fresh"].values()
                for details in mode.values()
            )
        )
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "task": 192,
                    "fresh_per_seed": args.fresh,
                    "seeds": seed_values,
                    "generation_errors": generation_errors,
                    "models": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            spec,
            row["score"],
            f"fresh_min={row['fresh_two_seed_min_accuracy']:.3%}",
            flush=True,
        )


if __name__ == "__main__":
    main()

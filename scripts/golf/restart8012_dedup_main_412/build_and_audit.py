#!/usr/bin/env python3
"""Exact task023 initializer consolidation on the 8012.15 authority."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
CANDIDATE = HERE / "task023_cost1317_exact.onnx"
SEEDS = (41202301, 41202302)
FRESH_PER_SEED = 1000
ALIASES = {
    "axis0": "slice_starts",       # both int64[1] == [0]
    "idx8": "slice_ends8",        # both int64[1] == [8]
    "mul3_u8": "shift3_u8",       # uint8 3; scalar broadcasts identically
    "v2": "shift2_u8",            # uint8 2; scalar broadcasts identically
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def authority_model() -> onnx.ModelProto:
    with zipfile.ZipFile(AUTHORITY) as archive:
        return onnx.load_model_from_string(archive.read("task023.onnx"))


def build(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    names = {init.name for init in model.graph.initializer}
    if not set(ALIASES).issubset(names) or not set(ALIASES.values()).issubset(names):
        raise RuntimeError("expected initializer names missing")
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in ALIASES:
                node.input[index] = ALIASES[name]
    kept = [init for init in model.graph.initializer if init.name not in ALIASES]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def session(data: bytes, level: ort.GraphOptimizationLevel, threads: int):
    model = scoring.sanitize_model(onnx.load_model_from_string(data))
    if model is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def evaluate_pair(base_session, candidate_session, examples: list[dict]) -> dict:
    right = wrong = errors = nonfinite = shape_mismatch = small_positive = raw_mismatch = 0
    minimum_positive = math.inf
    for example in examples:
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            base_raw = base_session.run(["output"], {"input": benchmark["input"]})[0]
            raw = candidate_session.run(["output"], {"input": benchmark["input"]})[0]
        except BaseException:
            errors += 1
            wrong += 1
            continue
        raw_mismatch += int(not np.array_equal(base_raw, raw))
        shape_mismatch += int(raw.shape != benchmark["output"].shape)
        nonfinite += int(not np.all(np.isfinite(raw)))
        small_positive += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        positive = raw[raw > 0]
        if positive.size:
            minimum_positive = min(minimum_positive, float(np.min(positive)))
        if np.array_equal((raw > 0).astype(np.float32), benchmark["output"]):
            right += 1
        else:
            wrong += 1
    total = right + wrong
    return {
        "total": total,
        "right": right,
        "wrong": wrong,
        "accuracy": right / total if total else 0.0,
        "runtime_errors": errors,
        "nonfinite_cases": nonfinite,
        "shape_mismatch_cases": shape_mismatch,
        "small_positive_values": small_positive,
        "raw_mismatch_vs_authority": raw_mismatch,
        "minimum_positive": None if minimum_positive is math.inf else minimum_positive,
    }


def profile(model: onnx.ModelProto, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="task023_412_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(copy.deepcopy(model), 23, workdir, label, require_correct=True)
    if result is None:
        raise RuntimeError(f"profile failed: {label}")
    return result


def main() -> int:
    ort.set_default_logger_severity(4)
    base = authority_model()
    candidate = build(base)
    onnx.save(candidate, CANDIDATE)
    base_data = base.SerializeToString()
    candidate_data = CANDIDATE.read_bytes()
    examples = scoring.load_examples(23)
    known = list(examples["train"]) + list(examples["test"]) + list(examples["arc-gen"])
    generator = importlib.import_module("task_150deff5")
    fresh = {}
    for seed in SEEDS:
        random.seed(seed)
        fresh[seed] = [generator.generate() for _ in range(FRESH_PER_SEED)]

    configs = {}
    for label, level, threads in (
        ("disable_threads1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
        ("disable_threads4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
        ("enable_threads1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
        ("enable_threads4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
    ):
        base_sess = session(base_data, level, threads)
        cand_sess = session(candidate_data, level, threads)
        configs[label] = {
            "known": evaluate_pair(base_sess, cand_sess, known),
            "fresh": {str(seed): evaluate_pair(base_sess, cand_sess, fresh[seed]) for seed in SEEDS},
        }

    base_profile = profile(base, "base")
    candidate_profile = profile(candidate, "candidate")
    rows = [subset for config in configs.values() for subset in [config["known"], *config["fresh"].values()]]
    policy90_pass = all(row["accuracy"] >= 0.90 for row in rows)
    raw_equivalent = all(row["raw_mismatch_vs_authority"] == 0 for row in rows)
    hard_gates_pass = (
        all(row["runtime_errors"] == 0 for row in rows)
        and all(row["nonfinite_cases"] == 0 for row in rows)
        and all(row["shape_mismatch_cases"] == 0 for row in rows)
        and all(row["small_positive_values"] == 0 for row in rows)
    )
    admitted = (
        candidate_profile["cost"] < base_profile["cost"]
        and hard_gates_pass
        and (policy90_pass or raw_equivalent)
    )
    result = {
        "task": 23,
        "known_black_catalog_hit": False,
        "aliases": ALIASES,
        "authority": {"sha256": sha256(base_data), "profile": base_profile},
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": sha256(candidate_data),
            "profile": candidate_profile,
            "gain": math.log(base_profile["cost"] / candidate_profile["cost"]),
        },
        "configs": configs,
        "policy90_pass": policy90_pass,
        "raw_equivalent_to_verified_authority": raw_equivalent,
        "admission_class": "EXACT_AUTHORITY_EQUIVALENT" if raw_equivalent else "POLICY90",
        "admitted": admitted,
        "root_or_others_modified": False,
    }
    (HERE / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admitted": admitted, "base": base_profile, "candidate": candidate_profile, "gain": result["candidate"]["gain"], "minimum_accuracy": min(row["accuracy"] for row in rows)}, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())

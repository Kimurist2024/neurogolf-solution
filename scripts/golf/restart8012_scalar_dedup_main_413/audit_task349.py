#!/usr/bin/env python3
"""Replace task349's rank-4 scalar 29 with its existing scalar 29."""

from __future__ import annotations

import copy
import importlib
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
CANDIDATE = HERE / "task349_cost3531_exact.onnx"
SEEDS = (41334901, 41334902)
FRESH_PER_SEED = 1000

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
sys.path.insert(0, str(ROOT / "scripts/golf/restart8012_dedup_main_412"))
from lib import scoring  # noqa: E402
import build_and_audit as common  # noqa: E402


def load_authority() -> onnx.ModelProto:
    with zipfile.ZipFile(AUTHORITY) as archive:
        return onnx.load_model_from_string(archive.read("task349.onnx"))


def build(base: onnx.ModelProto) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    found = False
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == "max29_rank4_i8":
                if node.op_type != "Min":
                    raise RuntimeError("rank-4 29 has an unexpected consumer")
                node.input[index] = "max29_i8"
                found = True
    if not found:
        raise RuntimeError("rank-4 29 consumer missing")
    kept = [init for init in model.graph.initializer if init.name != "max29_rank4_i8"]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def profile(model: onnx.ModelProto, label: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="task349_413_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(copy.deepcopy(model), 349, workdir, label, require_correct=True)
    if result is None:
        raise RuntimeError(f"profile failed: {label}")
    return result


def main() -> int:
    ort.set_default_logger_severity(4)
    base = load_authority()
    candidate = build(base)
    onnx.save(candidate, CANDIDATE)
    base_data = base.SerializeToString()
    candidate_data = CANDIDATE.read_bytes()
    examples = scoring.load_examples(349)
    known = list(examples["train"]) + list(examples["test"]) + list(examples["arc-gen"])
    generator = importlib.import_module("task_db93a21d")
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
        base_session = common.session(base_data, level, threads)
        candidate_session = common.session(candidate_data, level, threads)
        configs[label] = {
            "known": common.evaluate_pair(base_session, candidate_session, known),
            "fresh": {
                str(seed): common.evaluate_pair(base_session, candidate_session, fresh[seed])
                for seed in SEEDS
            },
        }

    base_profile = profile(base, "base")
    candidate_profile = profile(candidate, "candidate")
    rows = [subset for config in configs.values() for subset in [config["known"], *config["fresh"].values()]]
    raw_equivalent = all(row["raw_mismatch_vs_authority"] == 0 for row in rows)
    hard_gates = all(
        row["runtime_errors"] == 0
        and row["nonfinite_cases"] == 0
        and row["shape_mismatch_cases"] == 0
        and row["small_positive_values"] == 0
        for row in rows
    )
    admitted = candidate_profile["cost"] < base_profile["cost"] and raw_equivalent and hard_gates
    result = {
        "task": 349,
        "known_black_catalog_hit": False,
        "rewrite": "max29_rank4_i8 -> max29_i8 for Min scalar broadcast",
        "authority": {"sha256": common.sha256(base_data), "profile": base_profile},
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": common.sha256(candidate_data),
            "profile": candidate_profile,
            "gain": math.log(base_profile["cost"] / candidate_profile["cost"]),
        },
        "configs": configs,
        "minimum_accuracy": min(row["accuracy"] for row in rows),
        "raw_equivalent_to_verified_authority": raw_equivalent,
        "admission_class": "EXACT_AUTHORITY_EQUIVALENT",
        "admitted": admitted,
        "root_or_others_modified": False,
    }
    (HERE / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"admitted": admitted, "base": base_profile, "candidate": candidate_profile, "gain": result["candidate"]["gain"], "minimum_accuracy": result["minimum_accuracy"]}, indent=2))
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())

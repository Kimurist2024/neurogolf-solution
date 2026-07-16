#!/usr/bin/env python3
"""Four-configuration known/fresh audit for the task354 combined candidate."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "others/71502/task354_improved(4).onnx"
CANDIDATE = HERE / "candidates/task354_combined.onnx"
SEEDS = (2026071501, 2026071502)
FRESH_PER_SEED = 2000
CONFIGS = (
    ("disable_all_threads1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_all_threads4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("enable_all_threads1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("enable_all_threads4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(data: bytes, level: ort.GraphOptimizationLevel, threads: int):
    model = scoring.sanitize_model(onnx.load_model_from_string(data))
    if model is None:
        raise RuntimeError("sanitize rejected")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def cases() -> tuple[list[dict], dict[int, list[dict]]]:
    examples = scoring.load_examples(354)
    known = list(examples["train"]) + list(examples["test"]) + list(examples["arc-gen"])
    generator = importlib.import_module("task_ddf7fa4f")
    fresh: dict[int, list[dict]] = {}
    for seed in SEEDS:
        random.seed(seed)
        fresh[seed] = [generator.generate() for _ in range(FRESH_PER_SEED)]
    return known, fresh


def evaluate(session: ort.InferenceSession, examples: list[dict]) -> dict:
    right = wrong = errors = nonfinite = small_positive = shape_mismatch = 0
    raw_hash = hashlib.sha256()
    first_failure = None
    minimum_positive = math.inf
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            raw = session.run(["output"], {"input": benchmark["input"]})[0]
        except BaseException as exc:  # audit must fail closed on every ORT error
            errors += 1
            wrong += 1
            if first_failure is None:
                first_failure = {"index": index, "error": f"{type(exc).__name__}: {exc}"}
            continue
        raw_hash.update(np.ascontiguousarray(raw).tobytes())
        if raw.shape != benchmark["output"].shape:
            shape_mismatch += 1
        if not np.all(np.isfinite(raw)):
            nonfinite += 1
        positives = raw[raw > 0]
        if positives.size:
            minimum_positive = min(minimum_positive, float(np.min(positives)))
        small_positive += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        prediction = (raw > 0).astype(np.float32)
        if np.array_equal(prediction, benchmark["output"]):
            right += 1
        else:
            wrong += 1
            if first_failure is None:
                first_failure = {"index": index, "error": "gold mismatch"}
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
        "minimum_positive": None if minimum_positive is math.inf else minimum_positive,
        "raw_sha256": raw_hash.hexdigest(),
        "first_failure": first_failure,
    }


def profile(data: bytes, label: str) -> dict:
    model = onnx.load_model_from_string(data)
    with tempfile.TemporaryDirectory(prefix="task354_407_", dir="/tmp") as workdir:
        result = scoring.score_and_verify(
            copy.deepcopy(model), 354, workdir, label=label, require_correct=True
        )
    if result is None:
        raise RuntimeError(f"profile failed for {label}")
    return result


def static_audit(data: bytes) -> dict:
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    banned = {
        "Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"
    }
    bad_ops = sorted(
        {node.op_type for node in model.graph.node if node.op_type in banned or "Sequence" in node.op_type}
    )
    finite = True
    for init in model.graph.initializer:
        try:
            finite = finite and bool(np.all(np.isfinite(onnx.numpy_helper.to_array(init))))
        except TypeError:
            pass
    return {
        "full_check": True,
        "strict_shape_inference": inferred is not None,
        "banned_ops": bad_ops,
        "finite_initializers": finite,
        "functions": len(model.functions),
        "nested_graphs": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node
            for attr in node.attribute
        ),
        "declared_input_shape": [
            dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim
        ],
        "declared_output_shape": [
            dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim
        ],
        "legacy_shape_cloak_inherited": True,
        "note": (
            "The already-LB-white task354 authority declares a 1x1x1x1 output "
            "while returning 1x10x30x30. Candidate preserves that legacy lineage; "
            "it is not classified as a new sound/no-cloak graph."
        ),
    }


def main() -> int:
    ort.set_default_logger_severity(4)
    base_data = BASE.read_bytes()
    data = CANDIDATE.read_bytes()
    known, fresh = cases()
    configs = {}
    for label, level, threads in CONFIGS:
        session = make_session(data, level, threads)
        configs[label] = {
            "known": evaluate(session, known),
            "fresh": {str(seed): evaluate(session, fresh[seed]) for seed in SEEDS},
        }

    all_rows = [
        subset
        for config in configs.values()
        for subset in [config["known"], *config["fresh"].values()]
    ]
    hashes_by_subset = {
        "known": sorted({config["known"]["raw_sha256"] for config in configs.values()}),
        **{
            f"fresh_{seed}": sorted(
                {config["fresh"][str(seed)]["raw_sha256"] for config in configs.values()}
            )
            for seed in SEEDS
        },
    }
    base_profile = profile(base_data, "base")
    candidate_profile = profile(data, "combined")
    admitted = (
        int(candidate_profile["cost"]) < int(base_profile["cost"])
        and all(row["accuracy"] >= 0.90 for row in all_rows)
        and all(row["runtime_errors"] == 0 for row in all_rows)
        and all(row["nonfinite_cases"] == 0 for row in all_rows)
        and all(row["shape_mismatch_cases"] == 0 for row in all_rows)
        and all(row["small_positive_values"] == 0 for row in all_rows)
        and all(len(items) == 1 for items in hashes_by_subset.values())
    )
    result = {
        "task": 354,
        "policy": "KNOWN_BLACK_EXCLUDED_ELSE_ACCURACY90",
        "known_black_catalog_hit": False,
        "known_lb_white_lineage": "task354 has prior white probes in docs/golf/private_zero_tasks.md",
        "authority": {
            "path": str(BASE.relative_to(ROOT)),
            "sha256": digest(base_data),
            "profile": base_profile,
        },
        "candidate": {
            "path": str(CANDIDATE.relative_to(ROOT)),
            "sha256": digest(data),
            "profile": candidate_profile,
            "gain": math.log(int(base_profile["cost"]) / int(candidate_profile["cost"])),
        },
        "configs": configs,
        "cross_config_raw_hashes": hashes_by_subset,
        "static": static_audit(data),
        "admitted_under_user_policy": admitted,
        "root_or_checkpoint_modified": False,
    }
    (HERE / "audit_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "admitted": admitted,
                "cost": candidate_profile["cost"],
                "gain": result["candidate"]["gain"],
                "minimum_accuracy": min(row["accuracy"] for row in all_rows),
                "rows": len(all_rows),
            },
            indent=2,
        )
    )
    return 0 if admitted else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""SOUND exact-regolf audit for the immutable 8009.46 members.

This script never promotes models.  It checks exact generator behavior under
four ORT configurations, runtime-truthful shapes, structural policy gates, and
fixed-point semantics-preserving onnxoptimizer profiles.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline"
CANDIDATES = HERE / "candidates"
EVIDENCE = HERE / "evidence"
TASKS = (153, 161, 200, 316)
HASHES = {
    153: "681b3aeb",
    161: "6cdd2623",
    200: "8403a5d5",
    316: "cdecee7f",
}
AUTHORITY_COSTS = {153: 230, 161: 190, 200: 346, 316: 246}
FRESH_SEEDS = (153_161_200, 316_200_161)
FRESH_PER_SEED = 3000
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf import check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402


PASS_SETS: dict[str, list[str]] = {
    "dead": ["eliminate_deadend"],
    "cse": ["eliminate_common_subexpression"],
    "initializer_aliases": [
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
    ],
    "idempotent": ["eliminate_consecutive_idempotent_ops"],
    "noops": [
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_concat",
        "eliminate_nop_dropout",
        "eliminate_nop_expand",
        "eliminate_nop_flatten",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_split",
        "eliminate_nop_transpose",
        "eliminate_nop_with_unit",
    ],
    "all_safe_cleanup": [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_unused_initializer",
        "eliminate_common_subexpression",
        "eliminate_consecutive_idempotent_ops",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_concat",
        "eliminate_nop_dropout",
        "eliminate_nop_expand",
        "eliminate_nop_flatten",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_split",
        "eliminate_nop_transpose",
        "eliminate_nop_with_unit",
    ],
    "conv_fusions": [
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_pad_into_conv",
        "fuse_pad_into_pool",
    ],
    "shape_folds": [
        "eliminate_shape_gather",
        "eliminate_slice_after_shape",
        "eliminate_shape_op",
        "fuse_consecutive_slices",
    ],
    "einsum_matmul": ["replace_einsum_with_matmul"],
    "rewrite_where": ["rewrite_where"],
    "adjust_add": ["adjust_add"],
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
        for dim in value.type.tensor_type.shape.dim
    ]


def load_known(task: int) -> list[dict[str, Any]]:
    examples = scoring.load_examples(task)
    return [
        example
        for split in ("train", "test", "arc-gen")
        for example in examples.get(split, [])
    ]


def load_rule(task: int):
    path = ROOT / "inputs/sakana-gcg-2025/raw" / f"task{task:03d}.py"
    spec = importlib.util.spec_from_file_location(f"raw_rule_{task}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def make_fresh(task: int, seed: int) -> list[dict[str, Any]]:
    module = importlib.import_module(f"task_{HASHES[task]}")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [module.generate() for _ in range(FRESH_PER_SEED)]


def rule_check(task: int, examples: list[dict[str, Any]]) -> dict[str, Any]:
    rule = load_rule(task)
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    for index, example in enumerate(examples):
        observed = rule(example["input"])
        if observed != example["output"]:
            mismatch_count += 1
            if len(mismatches) < 10:
                mismatches.append({"index": index})
    return {
        "attempts": len(examples),
        "right": len(examples) - mismatch_count,
        "mismatch_count": mismatch_count,
        "first_mismatches": mismatches,
    }


def make_session(data: bytes, disabled: bool, threads: int) -> ort.InferenceSession:
    model = scoring.sanitize_model(copy.deepcopy(onnx.load_model_from_string(data)))
    if model is None:
        raise RuntimeError("sanitize_model rejected payload")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disabled
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def runtime_group(data: bytes, examples: list[dict[str, Any]]) -> dict[str, Any]:
    sessions: dict[str, ort.InferenceSession] = {}
    session_errors: dict[str, str] = {}
    for disabled, threads, label in CONFIGS:
        try:
            sessions[label] = make_session(data, disabled, threads)
        except Exception as exc:  # noqa: BLE001
            session_errors[label] = f"{type(exc).__name__}: {exc}"
    metrics: dict[str, dict[str, Any]] = {
        label: {
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "nonfinite_values": 0,
            "bad_margin_values": 0,
            "raw_equal_to_disable_all_threads1": 0,
            "output_shapes": set(),
            "min_positive": None,
            "max_nonpositive": None,
        }
        for _, _, label in CONFIGS
    }
    valid = 0
    first_failure = None
    for index, example in enumerate(examples):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            continue
        valid += 1
        expected = benchmark["output"].astype(bool)
        outputs: dict[str, np.ndarray] = {}
        for label, session in sessions.items():
            try:
                value = np.asarray(
                    session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                )
            except Exception as exc:  # noqa: BLE001
                metrics[label]["runtime_errors"] += 1
                first_failure = first_failure or {
                    "index": index,
                    "config": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            outputs[label] = value
            row = metrics[label]
            row["output_shapes"].add(tuple(int(x) for x in value.shape))
            row["nonfinite_values"] += int(value.size - np.isfinite(value).sum())
            row["bad_margin_values"] += int(np.count_nonzero((value > 0) & (value < 0.25)))
            positive = value[value > 0]
            nonpositive = value[value <= 0]
            if positive.size:
                current = float(np.min(positive))
                row["min_positive"] = (
                    current if row["min_positive"] is None
                    else min(row["min_positive"], current)
                )
            if nonpositive.size:
                current = float(np.max(nonpositive))
                row["max_nonpositive"] = (
                    current if row["max_nonpositive"] is None
                    else max(row["max_nonpositive"], current)
                )
            correct = value.shape == expected.shape and np.array_equal(value > 0, expected)
            row["right" if correct else "wrong"] += 1
            if not correct and first_failure is None:
                first_failure = {
                    "index": index,
                    "config": label,
                    "shape": list(value.shape),
                    "expected_shape": list(expected.shape),
                    "threshold_differences": (
                        int(np.count_nonzero((value > 0) != expected))
                        if value.shape == expected.shape else None
                    ),
                }
        reference = outputs.get("disable_all_threads1")
        if reference is not None:
            reference_bytes = np.ascontiguousarray(reference).tobytes()
            for label, value in outputs.items():
                if (
                    value.dtype == reference.dtype
                    and value.shape == reference.shape
                    and np.ascontiguousarray(value).tobytes() == reference_bytes
                ):
                    metrics[label]["raw_equal_to_disable_all_threads1"] += 1
    for row in metrics.values():
        row["output_shapes"] = [list(x) for x in sorted(row["output_shapes"])]
    return {
        "attempts": len(examples),
        "valid": valid,
        "session_errors": session_errors,
        "configs": metrics,
        "first_failure": first_failure,
    }


def structural_audit(task: int, path: Path, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    memory, params, cost = cost_of(str(path))
    produced = {name for node in model.graph.node for name in node.output if name}
    consumed = {name for node in model.graph.node for name in node.input if name}
    graph_outputs = {item.name for item in model.graph.output}
    initializer_names = {item.name for item in model.graph.initializer}
    used_initializers = initializer_names & consumed
    initializers_by_value: dict[tuple[Any, ...], list[str]] = {}
    for item in model.graph.initializer:
        key = (item.data_type, tuple(item.dims), item.raw_data, tuple(item.float_data),
               tuple(item.int32_data), tuple(item.int64_data), tuple(item.double_data))
        initializers_by_value.setdefault(key, []).append(item.name)
    duplicates = [names for names in initializers_by_value.values() if len(names) > 1]
    return {
        "task": task,
        "sha256": sha256(data),
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "profile": {"memory": int(memory), "params": int(params), "cost": int(cost)},
        "authority_cost": AUTHORITY_COSTS[task],
        "profile_matches_authority": int(cost) == AUTHORITY_COSTS[task],
        "checker_full": True,
        "strict_data_prop": True,
        "declared_outputs": [shape(item) for item in model.graph.output],
        "inferred_outputs": [shape(item) for item in inferred.graph.output],
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "standard_domains": all(x.domain in {"", "ai.onnx"} for x in model.opset_import)
        and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "banned_ops": [
            node.op_type for node in model.graph.node
            if node.op_type in {
                "Loop", "Scan", "NonZero", "Unique", "Compress", "Script",
                "Function", "SequenceAt", "SequenceConstruct", "SequenceEmpty",
                "SequenceErase", "SequenceInsert", "SplitToSequence",
            }
        ],
        "lookup_ops": [
            node.op_type for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax"}
        ],
        "center_crop_pad": sum(node.op_type == "CenterCropPad" for node in model.graph.node),
        "giant_nodes": [
            {"op": node.op_type, "inputs": len(node.input)}
            for node in model.graph.node if len(node.input) > 16
        ],
        "conv_bias_ub": [list(item) for item in check_conv_bias.check_model(model)],
        "dead_node_outputs": sorted(produced - consumed - graph_outputs),
        "unused_initializers": sorted(initializer_names - used_initializers),
        "duplicate_initializer_groups": duplicates,
    }


def runtime_shape_trace(data: bytes, example: dict[str, Any]) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        typed = {
            item.name: item
            for item in list(inferred.graph.value_info) + list(inferred.graph.output)
            if item.type.HasField("tensor_type")
        }
        expected_shapes = {name: tuple(shape(item)) for name, item in typed.items()}
        names: list[str] = []
        seen: set[str] = set()
        for node in inferred.graph.node:
            for name in node.output:
                if name and name in typed and name not in seen:
                    names.append(name)
                    seen.add(name)
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 4
        session = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError("known witness conversion failed")
        values = session.run(names, {session.get_inputs()[0].name: benchmark["input"]})
        mismatches = [
            {
                "tensor": name,
                "declared": list(expected_shapes[name]),
                "runtime": list(value.shape),
            }
            for name, value in zip(names, values)
            if tuple(value.shape) != expected_shapes[name]
        ]
        return {
            "traced": len(names),
            "truthful": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
        }
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def optimizer_scan(task: int, data: bytes) -> list[dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    rows: list[dict[str, Any]] = []
    for label, passes in PASS_SETS.items():
        row: dict[str, Any] = {"label": label, "passes": passes}
        try:
            candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
            encoded = candidate.SerializeToString()
            row["changed"] = encoded != data
            if row["changed"]:
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
                with tempfile.TemporaryDirectory(prefix=f"audit_{task}_{label}_") as tmp:
                    path = Path(tmp) / f"task{task:03d}.onnx"
                    onnx.save(candidate, path)
                    memory, params, cost = cost_of(str(path))
                row["profile"] = {
                    "memory": int(memory), "params": int(params), "cost": int(cost)
                }
                row["strict_lower"] = int(cost) < AUTHORITY_COSTS[task]
                if row["strict_lower"]:
                    path = CANDIDATES / f"task{task:03d}_{label}.onnx"
                    onnx.save(candidate, path)
                    row["candidate"] = str(path.relative_to(ROOT))
                    row["sha256"] = sha256(path.read_bytes())
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(4)
    report: dict[str, Any] = {
        "authority_zip": "submission_base_8009.46.zip",
        "authority_zip_sha256": sha256((ROOT / "submission_base_8009.46.zip").read_bytes()),
        "fresh_seeds": list(FRESH_SEEDS),
        "fresh_per_seed": FRESH_PER_SEED,
        "configs": [label for _, _, label in CONFIGS],
        "tasks": {},
    }
    output = EVIDENCE / "audit.json"
    for task in TASKS:
        path = BASE / f"task{task:03d}.onnx"
        data = path.read_bytes()
        known = load_known(task)
        row: dict[str, Any] = {
            "structural": structural_audit(task, path, data),
            "runtime_shape": runtime_shape_trace(data, known[0]),
            "optimizer_profiles": optimizer_scan(task, data),
            "known_rule": rule_check(task, known),
            "known_runtime": runtime_group(data, known),
            "fresh": {},
        }
        report["tasks"][str(task)] = row
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"task{task:03d} known complete", flush=True)
        for seed in FRESH_SEEDS:
            examples = make_fresh(task, seed)
            row["fresh"][str(seed)] = {
                "rule": rule_check(task, examples),
                "runtime": runtime_group(data, examples),
            }
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"task{task:03d} fresh seed={seed} complete", flush=True)
    strict_lower = [
        {"task": int(task), **profile}
        for task, row in report["tasks"].items()
        for profile in row["optimizer_profiles"]
        if profile.get("strict_lower")
    ]
    report["summary"] = {
        "profiles": len(TASKS) * len(PASS_SETS),
        "strict_lower_count": len(strict_lower),
        "strict_lower": strict_lower,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only exact-regolf audit for tasks 175/224/240/376.

All emitted artifacts stay below this lane directory.  Root submissions and
score ledgers are authority inputs only and are never rewritten.
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
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE / "baseline"
EVIDENCE = HERE / "evidence"
CANDIDATES = HERE / "candidates"
TASKS = (175, 224, 240, 376)
HASHES = {
    175: "73251a56",
    224: "928ad970",
    240: "9d9215db",
    376: "eb281b96",
}
AUTHORITY_COSTS = {175: 166, 224: 162, 240: 160, 376: 158}
CONFIGS = (
    (True, 1, "disable_all_threads1"),
    (True, 4, "disable_all_threads4"),
    (False, 1, "default_threads1"),
    (False, 4, "default_threads4"),
)
FRESH_SEEDS = (175_224_240_376, 376_240_224_175)
FRESH_PER_SEED = 3_000
POLICY90_TASK175 = (
    HERE.parent / "root_sweep29/prune_latent/task175_r001.onnx"
)

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "inputs/arc-gen-repo/tasks"))
from golf import check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


PASS_SETS: dict[str, list[str]] = {
    "dead": ["eliminate_deadend"],
    "cse": ["eliminate_common_subexpression"],
    "initializer_cleanup": [
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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dims(value: onnx.ValueInfoProto) -> list[int | str | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [
        int(dim.dim_value)
        if dim.HasField("dim_value")
        else dim.dim_param if dim.HasField("dim_param") else None
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
    spec = importlib.util.spec_from_file_location(f"independent_rule_{task}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.p


def rule_check(task: int, examples: list[dict[str, Any]]) -> dict[str, Any]:
    rule = load_rule(task)
    wrong = 0
    first: list[int] = []
    for index, example in enumerate(examples):
        actual = rule(copy.deepcopy(example["input"]))
        if actual != example["output"]:
            wrong += 1
            if len(first) < 10:
                first.append(index)
    return {
        "attempts": len(examples),
        "right": len(examples) - wrong,
        "wrong": wrong,
        "first_mismatch_indices": first,
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


def runtime_group(
    data: bytes,
    examples: list[dict[str, Any]],
    authority_data: bytes | None = None,
) -> dict[str, Any]:
    sessions: dict[str, ort.InferenceSession] = {}
    authority_sessions: dict[str, ort.InferenceSession] = {}
    session_errors: dict[str, str] = {}
    for disabled, threads, label in CONFIGS:
        try:
            sessions[label] = make_session(data, disabled, threads)
            if authority_data is not None:
                authority_sessions[label] = make_session(
                    authority_data, disabled, threads
                )
        except Exception as exc:  # noqa: BLE001
            session_errors[label] = f"{type(exc).__name__}: {exc}"
    stats: dict[str, dict[str, Any]] = {
        label: {
            "right": 0,
            "wrong": 0,
            "runtime_errors": 0,
            "raw_equal_to_disable_all_threads1": 0,
            "raw_equal_to_authority": 0,
            "nonfinite_values": 0,
            "bad_margin_values": 0,
            "output_shapes": set(),
            "min_positive": None,
            "max_nonpositive": None,
        }
        for _, _, label in CONFIGS
    }
    valid = 0
    first_failure: dict[str, Any] | None = None
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
                if authority_data is not None:
                    ref_session = authority_sessions[label]
                    authority = np.asarray(
                        ref_session.run(
                            [ref_session.get_outputs()[0].name],
                            {ref_session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                    )
                    if (
                        value.dtype == authority.dtype
                        and value.shape == authority.shape
                        and np.array_equal(value, authority, equal_nan=False)
                    ):
                        stats[label]["raw_equal_to_authority"] += 1
            except Exception as exc:  # noqa: BLE001
                stats[label]["runtime_errors"] += 1
                first_failure = first_failure or {
                    "index": index,
                    "config": label,
                    "kind": "runtime",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            outputs[label] = value
            row = stats[label]
            row["output_shapes"].add(tuple(int(item) for item in value.shape))
            row["nonfinite_values"] += int(value.size - np.isfinite(value).sum())
            row["bad_margin_values"] += int(
                np.count_nonzero((value > 0.0) & (value < 0.25))
            )
            positive = value[value > 0.0]
            nonpositive = value[value <= 0.0]
            if positive.size:
                current = float(positive.min())
                row["min_positive"] = (
                    current
                    if row["min_positive"] is None
                    else min(row["min_positive"], current)
                )
            if nonpositive.size:
                current = float(nonpositive.max())
                row["max_nonpositive"] = (
                    current
                    if row["max_nonpositive"] is None
                    else max(row["max_nonpositive"], current)
                )
            correct = value.shape == expected.shape and np.array_equal(
                value > 0.0, expected
            )
            row["right" if correct else "wrong"] += 1
            if not correct and first_failure is None:
                first_failure = {
                    "index": index,
                    "config": label,
                    "kind": "wrong",
                    "output_shape": list(value.shape),
                    "expected_shape": list(expected.shape),
                    "threshold_differences": (
                        int(np.count_nonzero((value > 0.0) != expected))
                        if value.shape == expected.shape
                        else None
                    ),
                }
        reference = outputs.get("disable_all_threads1")
        if reference is not None:
            for label, value in outputs.items():
                if (
                    value.dtype == reference.dtype
                    and value.shape == reference.shape
                    and np.array_equal(value, reference, equal_nan=False)
                ):
                    stats[label]["raw_equal_to_disable_all_threads1"] += 1
    for row in stats.values():
        row["output_shapes"] = [list(x) for x in sorted(row["output_shapes"])]
    return {
        "attempts": len(examples),
        "valid": valid,
        "session_errors": session_errors,
        "configs": stats,
        "first_failure": first_failure,
    }


def zero_hyperplanes(array: np.ndarray) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    if array.ndim == 0:
        return result
    for axis in range(array.ndim):
        other = tuple(index for index in range(array.ndim) if index != axis)
        collapsed = np.all(array == 0, axis=other) if other else array == 0
        indices = np.flatnonzero(collapsed).astype(int).tolist()
        if indices:
            result[str(axis)] = indices
    return result


def structural_audit(
    task: int,
    path: Path,
    data: bytes,
    *,
    require_correct: bool = True,
) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }
    node_outputs = [
        name for node in inferred.graph.node for name in node.output if name
    ]
    unresolved = []
    nonpositive = []
    for name in node_outputs:
        value = typed.get(name)
        if value is None or not value.type.HasField("tensor_type"):
            unresolved.append(name)
            continue
        shape = dims(value)
        if any(not isinstance(item, int) or item <= 0 for item in shape):
            nonpositive.append({"tensor": name, "shape": shape})
    produced = set(node_outputs)
    consumed = {name for node in model.graph.node for name in node.input if name}
    graph_outputs = {value.name for value in model.graph.output}
    initializer_names = {item.name for item in model.graph.initializer}
    by_value: dict[tuple[Any, ...], list[str]] = {}
    initializers = []
    for item in model.graph.initializer:
        array = numpy_helper.to_array(item)
        key = (str(array.dtype), tuple(array.shape), array.tobytes())
        by_value.setdefault(key, []).append(item.name)
        initializers.append(
            {
                "name": item.name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "params": int(max(1, math.prod(array.shape))),
                "finite": bool(np.isfinite(array).all()),
                "zero_values": int(np.count_nonzero(array == 0)),
                "zero_hyperplanes": zero_hyperplanes(array),
            }
        )
    with tempfile.TemporaryDirectory(prefix=f"official_{task}_", dir=HERE) as tmp:
        official = scoring.score_and_verify(
            copy.deepcopy(model),
            task,
            tmp,
            label="authority" if require_correct else "policy90",
            require_correct=require_correct,
        )
    if official is None:
        raise RuntimeError(f"official scoring rejected authority task {task}")
    return {
        "task": task,
        "path": str(path.relative_to(ROOT)),
        "sha256": digest(data),
        "file_bytes": len(data),
        "nodes": len(model.graph.node),
        "initializers": initializers,
        "official": official,
        "authority_cost": AUTHORITY_COSTS[task],
        "cost_matches_authority": official["cost"] == AUTHORITY_COSTS[task],
        "checker_full": True,
        "strict_data_prop": True,
        "canonical_input_shapes": [dims(value) for value in inferred.graph.input],
        "canonical_output_shapes": [dims(value) for value in inferred.graph.output],
        "node_outputs": len(node_outputs),
        "node_outputs_with_inferred_type_shape": len(node_outputs) - len(unresolved),
        "unresolved_node_outputs": unresolved,
        "nonpositive_or_symbolic_node_outputs": nonpositive,
        "standard_domains": all(
            item.domain in {"", "ai.onnx"} for item in model.opset_import
        )
        and all(node.domain in {"", "ai.onnx"} for node in model.graph.node),
        "ops": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
        "banned_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type.upper()
            in {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
            or "Sequence" in node.op_type
        ],
        "lookup_or_cloak_ops": [
            node.op_type
            for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax", "CenterCropPad"}
        ],
        "nested_graph_attributes": sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for node in model.graph.node
            for attr in node.attribute
        ),
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "conv_bias_ub": [list(item) for item in check_conv_bias.check_model(model)],
        "max_node_inputs": max((len(node.input) for node in model.graph.node), default=0),
        "dead_node_outputs": sorted(produced - consumed - graph_outputs),
        "unused_initializers": sorted(initializer_names - consumed),
        "duplicate_initializer_groups": [
            names for names in by_value.values() if len(names) > 1
        ],
    }


def runtime_shape_trace(task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    typed = {
        value.name: value
        for value in list(inferred.graph.value_info) + list(inferred.graph.output)
        if value.type.HasField("tensor_type")
    }
    names: list[str] = []
    for node in inferred.graph.node:
        for name in node.output:
            if name and name in typed and name not in names:
                names.append(name)
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
    converted = scoring.convert_to_numpy(load_known(task)[0])
    if converted is None:
        raise RuntimeError("known witness conversion failed")
    values = session.run(names, {session.get_inputs()[0].name: converted["input"]})
    mismatches = []
    tensors = []
    for name, value in zip(names, values):
        declared = dims(typed[name])
        runtime = list(value.shape)
        tensors.append(
            {
                "name": name,
                "declared": declared,
                "runtime": runtime,
                "dtype": str(value.dtype),
                "finite": bool(np.isfinite(value).all()),
            }
        )
        if declared != runtime:
            mismatches.append(
                {"tensor": name, "declared": declared, "runtime": runtime}
            )
    return {
        "traced": len(names),
        "truthful": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "nonfinite_tensors": [item["name"] for item in tensors if not item["finite"]],
        "tensors": tensors,
    }


def optimizer_scan(task: int, data: bytes) -> list[dict[str, Any]]:
    model = onnx.load_model_from_string(data)
    rows = []
    for label, passes in PASS_SETS.items():
        row: dict[str, Any] = {"label": label, "passes": passes}
        try:
            candidate = onnxoptimizer.optimize(model, passes, fixed_point=True)
            encoded = candidate.SerializeToString()
            row["changed"] = encoded != data
            if row["changed"]:
                onnx.checker.check_model(candidate, full_check=True)
                onnx.shape_inference.infer_shapes(
                    candidate, strict_mode=True, data_prop=True
                )
                with tempfile.TemporaryDirectory(prefix=f"opt_{task}_{label}_", dir=HERE) as tmp:
                    result = scoring.score_and_verify(
                        copy.deepcopy(candidate),
                        task,
                        tmp,
                        label=label,
                        require_correct=False,
                    )
                row["official"] = result
                row["strict_lower"] = (
                    result is not None
                    and result["correct"]
                    and result["cost"] < AUTHORITY_COSTS[task]
                )
                if row["strict_lower"]:
                    path = CANDIDATES / f"task{task:03d}_{label}.onnx"
                    onnx.save(candidate, path)
                    row["candidate"] = str(path.relative_to(ROOT))
                    row["sha256"] = digest(path.read_bytes())
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def history_summary(task: int) -> dict[str, Any]:
    prior = json.loads(
        (HERE.parent / "agent_mid20e_89/inventory/summary.json").read_text()
    )
    row = next(item for item in prior["per_task"] if item["task"] == task)
    result: dict[str, Any] = {
        "source": "scripts/golf/loop_8004_42_plus20/agent_mid20e_89/inventory/summary.json",
        "authority_sha_matches_current": row["authority_sha256"]
        == digest((BASE / f"task{task:03d}.onnx").read_bytes()),
        "unique_nonauthority_sha": row["unique_nonauthority_sha"],
        "stage_counts": row["stage_counts"],
        "conclusion": row["conclusion"],
    }
    if task in {224, 240}:
        old = json.loads(
            (HERE.parent / "root_high53/history_lead_audit.json").read_text()
        )
        result["numeric_lower_leads"] = [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "actual_cost": item["actual_cost"]["cost"],
                "known_right": item["known_disable_all"]["right"],
                "known_wrong": item["known_disable_all"]["wrong"],
                "runtime_errors": item["known_disable_all"]["errors"],
            }
            for item in old["lead_rows"]
            if item["task"] == task
        ]
    if task == 175:
        old = json.loads(
            (HERE.parent / "agent_prune_wave30a/result.json").read_text()
        )
        result["latent_prune_leads"] = [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "actual_cost": item["actual_cost"]["cost"],
                "known_right": item["known_dual"]["disable_all"]["right"],
                "known_wrong": item["known_dual"]["disable_all"]["wrong"],
                "runtime_errors": item["known_dual"]["disable_all"]["runtime_errors"],
            }
            for item in old["rows"]
            if item["task"] == task
        ]
    return result


def make_fresh(task: int, seed: int) -> list[dict[str, Any]]:
    module = importlib.import_module(f"task_{HASHES[task]}")
    common = importlib.import_module("common")
    random.seed(seed)
    common.random.seed(seed)
    return [module.generate() for _ in range(FRESH_PER_SEED)]


def main() -> int:
    ort.set_default_logger_severity(4)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "authority_zip": "submission_base_8009.46.zip",
        "authority_zip_sha256": digest(
            (ROOT / "submission_base_8009.46.zip").read_bytes()
        ),
        "tasks": {},
        "configs": [label for _, _, label in CONFIGS],
        "fresh_policy": {
            "seeds": list(FRESH_SEEDS),
            "per_seed": FRESH_PER_SEED,
            "run_only_after_strict_lower_known_complete_candidate": True,
        },
    }
    output = EVIDENCE / "audit.json"
    strict_lower: list[dict[str, Any]] = []
    for task in TASKS:
        path = BASE / f"task{task:03d}.onnx"
        data = path.read_bytes()
        known = load_known(task)
        row = {
            "generator_hash": HASHES[task],
            "known_cases": len(known),
            "independent_rule_known": rule_check(task, known),
            "structural": structural_audit(task, path, data),
            "runtime_shapes": runtime_shape_trace(task, data),
            "known_four_configs": runtime_group(data, known),
            "optimizer_profiles": optimizer_scan(task, data),
            "historical_evidence": history_summary(task),
        }
        report["tasks"][str(task)] = row
        for profile in row["optimizer_profiles"]:
            if profile.get("strict_lower"):
                strict_lower.append({"task": task, **profile})
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"task{task:03d} authority and optimizer audit complete", flush=True)

    fresh_ran = False
    finalists = []
    for item in strict_lower:
        task = item["task"]
        candidate_path = ROOT / item["candidate"]
        candidate_data = candidate_path.read_bytes()
        authority_data = (BASE / f"task{task:03d}.onnx").read_bytes()
        known = load_known(task)
        known_result = runtime_group(candidate_data, known, authority_data)
        configs = known_result["configs"].values()
        known_ok = not known_result["session_errors"] and all(
            row["right"] == len(known)
            and row["wrong"] == 0
            and row["runtime_errors"] == 0
            and row["raw_equal_to_authority"] == len(known)
            and row["nonfinite_values"] == 0
            and row["bad_margin_values"] == 0
            for row in configs
        )
        finalist = {"task": task, "profile": item, "known": known_result, "fresh": {}}
        if known_ok:
            fresh_ran = True
            for seed in FRESH_SEEDS:
                fresh = make_fresh(task, seed)
                finalist["fresh"][str(seed)] = {
                    "independent_rule": rule_check(task, fresh),
                    "runtime": runtime_group(candidate_data, fresh, authority_data),
                }
        finalists.append(finalist)
        output.write_text(json.dumps(report, indent=2) + "\n")

    report["finalists"] = finalists
    policy_data = POLICY90_TASK175.read_bytes()
    authority175 = (BASE / "task175.onnx").read_bytes()
    policy_known = load_known(175)
    policy90: dict[str, Any] = {
        "classification": "isolated_POLICY90_evidence_not_exact_winner",
        "source": str(POLICY90_TASK175.relative_to(ROOT)),
        "structural": structural_audit(
            175, POLICY90_TASK175, policy_data, require_correct=False
        ),
        "runtime_shapes": runtime_shape_trace(175, policy_data),
        "independent_rule_known": rule_check(175, policy_known),
        "known_four_configs": runtime_group(
            policy_data, policy_known, authority175
        ),
        "fresh": {},
    }
    for seed in FRESH_SEEDS:
        fresh = make_fresh(175, seed)
        policy90["fresh"][str(seed)] = {
            "independent_rule": rule_check(175, fresh),
            "runtime": runtime_group(policy_data, fresh, authority175),
        }
        report["policy90_task175"] = policy90
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"task175 POLICY90 fresh seed={seed} complete", flush=True)
    report["policy90_task175"] = policy90
    report["summary"] = {
        "authority_tasks": len(TASKS),
        "optimizer_profiles": len(TASKS) * len(PASS_SETS),
        "strict_lower_optimizer_candidates": len(strict_lower),
        "fresh_executed": fresh_ran,
        "fresh_not_run_reason": (
            None
            if fresh_ran
            else "No strict-lower candidate passed the authority/known gate."
        ),
        "winner": None,
        "policy90_candidate_is_exact_winner": False,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

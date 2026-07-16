#!/usr/bin/env python3
"""Fail-closed audit for the cheapest task012 retraining result.

The retained model is evidence only: it is cheaper than the authority, but it
fails both the exact-known gate and the requested POLICY90 generator gate.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TASK_DIR = ROOT / "inputs/arc-gen-repo/tasks"
KNOWN_PATH = ROOT / "inputs/neurogolf-2026/task012.json"
SOURCE = ROOT / (
    "scripts/golf/loop_8004_42_plus20/agent_high47/candidates/"
    "task012_history_r01_static500_a3640a1525.onnx"
)
AUTHORITY = ROOT / "artifacts/handcrafted/task012.onnx"
CANDIDATE = HERE / "task012_h7w8_known_opt_rejected.onnx"
EXPECTED_IO = [1, 10, 30, 30]
FRESH_SEEDS = (250012001, 250112001)
FRESH_PER_SEED = 5000
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Compress", "Script", "Function"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASK_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from golf.rank_dir import cost_of  # noqa: E402
from lib import scoring  # noqa: E402

GEN = importlib.import_module("task_0962bcdd")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | None]:
    result: list[int | None] = []
    for dim in value.type.tensor_type.shape.dim:
        result.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
    return result


def known_cases() -> list[dict[str, Any]]:
    data = json.loads(KNOWN_PATH.read_text())
    return [example for subset in ("train", "test", "arc-gen") for example in data[subset]]


def domain_cases() -> list[dict[str, Any]]:
    return [
        GEN.generate(colors=[1, 2], cols=[col0, col1], gravity=gravity)
        for col0 in range(3, 10)
        for col1 in range(3, 10)
        for gravity in range(4)
    ]


def session(model: onnx.ModelProto, disable: bool, threads: int) -> ort.InferenceSession:
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize_model rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if disable
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(sanitized.SerializeToString(), options)


def evaluate_cases(
    sess: ort.InferenceSession, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    right = wrong = errors = nonfinite = shape_mismatch = 0
    observed: set[tuple[int, ...]] = set()
    for example in cases:
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            errors += 1
            continue
        try:
            raw = sess.run(["output"], {"input": converted["input"]})[0]
            shape = tuple(int(value) for value in raw.shape)
            observed.add(shape)
            if shape != tuple(EXPECTED_IO):
                shape_mismatch += 1
            if not np.isfinite(raw).all():
                nonfinite += 1
            if np.array_equal(raw > 0.0, converted["output"] > 0.0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {
        "total": len(cases),
        "right": right,
        "wrong": wrong,
        "errors": errors,
        "nonfinite": nonfinite,
        "output_shape_mismatches": shape_mismatch,
        "observed_output_shapes": [list(shape) for shape in sorted(observed)],
    }


def evaluate_fresh(sess: ort.InferenceSession, seed: int, count: int) -> dict[str, Any]:
    random.seed(seed)
    right = wrong = errors = nonfinite = shape_mismatch = 0
    observed: set[tuple[int, ...]] = set()
    for _ in range(count):
        example = GEN.generate()
        converted = scoring.convert_to_numpy(example)
        if converted is None:
            errors += 1
            continue
        try:
            raw = sess.run(["output"], {"input": converted["input"]})[0]
            shape = tuple(int(value) for value in raw.shape)
            observed.add(shape)
            if shape != tuple(EXPECTED_IO):
                shape_mismatch += 1
            if not np.isfinite(raw).all():
                nonfinite += 1
            if np.array_equal(raw > 0.0, converted["output"] > 0.0):
                right += 1
            else:
                wrong += 1
        except Exception:
            errors += 1
    return {
        "seed": seed,
        "total": count,
        "right": right,
        "wrong": wrong,
        "rate": right / count,
        "errors": errors,
        "nonfinite": nonfinite,
        "output_shape_mismatches": shape_mismatch,
        "observed_output_shapes": [list(shape) for shape in sorted(observed)],
    }


def structural(model: onnx.ModelProto) -> dict[str, Any]:
    checker = strict = True
    checker_error = strict_error = None
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
    except Exception as exc:
        checker = False
        checker_error = f"{type(exc).__name__}: {exc}"
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception as exc:
        strict = False
        strict_error = f"{type(exc).__name__}: {exc}"
        inferred = model
    graph_values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    shapes = {value.name: tensor_shape(value) for value in graph_values}
    static_positive = all(
        shape and all(isinstance(dim, int) and dim > 0 for dim in shape)
        for shape in shapes.values()
    )
    domains = sorted(
        {item.domain for item in model.opset_import if item.domain not in ("", "ai.onnx")}
        | {node.domain for node in model.graph.node if node.domain not in ("", "ai.onnx")}
    )
    histogram = Counter(node.op_type for node in model.graph.node)
    initializer_arrays = {
        item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer
    }
    conv_bias_lengths = []
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 3 and node.input[2] in initializer_arrays:
            conv_bias_lengths.append(int(initializer_arrays[node.input[2]].size))
    return {
        "checker_full": checker,
        "checker_error": checker_error,
        "strict_shape_inference_data_prop": strict,
        "strict_error": strict_error,
        "all_declared_and_inferred_shapes_static_positive": static_positive,
        "shapes": shapes,
        "canonical_input_shape": shapes.get("input") == EXPECTED_IO,
        "canonical_output_shape": shapes.get("output") == EXPECTED_IO,
        "standard_domains": not domains,
        "nonstandard_domains": domains,
        "functions": len(model.functions),
        "sparse_initializers": len(model.graph.sparse_initializer),
        "banned_ops": sorted(
            {
                node.op_type
                for node in model.graph.node
                if node.op_type in BANNED or "Sequence" in node.op_type
            }
        ),
        "op_histogram": dict(sorted(histogram.items())),
        "conv_bias_lengths": conv_bias_lengths,
        "conv_bias_len10": bool(conv_bias_lengths) and all(length == 10 for length in conv_bias_lengths),
        "conv_bias_findings": check_conv_bias(model),
        "lookup_or_fixture_correction": False,
        "shape_cloak": False,
    }


def model_summary(path: Path) -> dict[str, Any]:
    memory, params, cost = cost_of(str(path))
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "file_bytes": path.stat().st_size,
        "memory": int(memory),
        "params": int(params),
        "cost": int(cost),
    }


def milp_summaries() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(HERE.glob("milp_*.json")):
        item = json.loads(path.read_text())
        rows.append(
            {
                "file": path.name,
                "dataset": item["dataset"],
                "kernel": [item["kh"], item["kw"]],
                "pad_top": item["pad_top"],
                "pad_left": item["pad_left"],
                "state_count": item["solver"]["state_count"],
                "best_exact_cases": item["solver"]["best_exact_cases"],
                "mip_gap": item["solver"]["mip_gap"],
                "success": item["solver"]["success"],
            }
        )
    return rows


def main() -> None:
    with zipfile.ZipFile(ROOT / "submission.zip") as archive:
        authority_bytes = archive.read("task012.onnx")
    if authority_bytes != AUTHORITY.read_bytes():
        raise RuntimeError("submission.zip task012 does not match handcrafted authority")

    model = onnx.load(CANDIDATE)
    structure = structural(model)
    if not all(
        (
            structure["checker_full"],
            structure["strict_shape_inference_data_prop"],
            structure["all_declared_and_inferred_shapes_static_positive"],
            structure["canonical_input_shape"],
            structure["canonical_output_shape"],
            structure["standard_domains"],
            structure["conv_bias_len10"],
            not structure["functions"],
            not structure["sparse_initializers"],
            not structure["banned_ops"],
            not structure["conv_bias_findings"],
        )
    ):
        raise RuntimeError(f"candidate structural audit failed: {structure}")

    configs = [
        ("disable_threads1", True, 1),
        ("default_threads1", False, 1),
        ("disable_threads4", True, 4),
        ("default_threads4", False, 4),
    ]
    known = known_cases()
    domain = domain_cases()
    runtime: dict[str, Any] = {}
    for name, disable, threads in configs:
        sess = session(model, disable, threads)
        runtime[name] = {
            "optimization": "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL",
            "threads": threads,
            "known": evaluate_cases(sess, known),
            "default_generator_domain": evaluate_cases(sess, domain),
            "fresh": [evaluate_fresh(sess, seed, FRESH_PER_SEED) for seed in FRESH_SEEDS],
        }

    reference_runtime: dict[str, Any] = {}
    for reference_name, reference_path, include_fresh in (
        ("source_cost500", SOURCE, False),
        ("authority_cost710", AUTHORITY, True),
    ):
        reference_model = onnx.load(reference_path)
        reference_runtime[reference_name] = {}
        for name, disable, threads in configs:
            sess = session(reference_model, disable, threads)
            row = {
                "optimization": "ORT_DISABLE_ALL" if disable else "ORT_ENABLE_ALL",
                "threads": threads,
                "known": evaluate_cases(sess, known),
                "default_generator_domain": evaluate_cases(sess, domain),
            }
            if include_fresh:
                row["fresh"] = [
                    evaluate_fresh(sess, seed, FRESH_PER_SEED) for seed in FRESH_SEEDS
                ]
            reference_runtime[reference_name][name] = row

    dense = json.loads((HERE / "dense_census.json").read_text())
    candidate_summary = model_summary(CANDIDATE)
    source_summary = model_summary(SOURCE)
    authority_summary = model_summary(AUTHORITY)
    if candidate_summary["cost"] != 570 or source_summary["cost"] != 500 or authority_summary["cost"] != 710:
        raise RuntimeError("unexpected official cost")

    all_known = [item["known"] for item in runtime.values()]
    all_domain = [item["default_generator_domain"] for item in runtime.values()]
    all_fresh = [item for config in runtime.values() for item in config["fresh"]]
    authority_known = [
        item["known"] for item in reference_runtime["authority_cost710"].values()
    ]
    authority_fresh = [
        fresh
        for item in reference_runtime["authority_cost710"].values()
        for fresh in item["fresh"]
    ]
    no_runtime_faults = all(
        item["errors"] == 0
        and item["nonfinite"] == 0
        and item["output_shape_mismatches"] == 0
        for item in all_known + all_domain + all_fresh
    )
    known_exact = all(item["right"] == item["total"] for item in all_known)
    policy90 = all(item["right"] / item["total"] >= 0.90 for item in all_domain)
    authority_exact = all(
        item["right"] == item["total"]
        and item["errors"] == 0
        and item["nonfinite"] == 0
        and item["output_shape_mismatches"] == 0
        for item in authority_known + authority_fresh
    )
    if not authority_exact:
        raise RuntimeError("authority did not reproduce known/fresh exact")

    payload = {
        "task": 12,
        "decision": "REJECT_KNOWN_AND_POLICY90",
        "winner": None,
        "authority": {
            **authority_summary,
            "submission_member": "task012.onnx",
            "submission_member_sha256": sha256_bytes(authority_bytes),
            "matches_artifacts_handcrafted": True,
        },
        "source": source_summary,
        "best_rejected_candidate": candidate_summary,
        "structure": structure,
        "runtime": runtime,
        "reference_runtime": reference_runtime,
        "gates": {
            "strict_structure": True,
            "runtime_fault_free": no_runtime_faults,
            "known_exact_all_four_configs": known_exact,
            "policy90_default_domain_all_four_configs": policy90,
            "candidate_cost_below_authority": candidate_summary["cost"] < authority_summary["cost"],
            "authority_known_and_fresh_exact_all_four_configs": authority_exact,
            "eligible": False,
        },
        "dense_hard_lp_census": {
            "known_total": dense["known_total"],
            "dense_layout_count": dense["dense_layout_count"],
            "dense_all_channel_feasible": dense["dense_all_channel_feasible"],
            "best_feasible_channel_count": dense["best_feasible_channel_count"],
            "dilated_geometry_census": dense["dilated_geometry_census"],
            "historical_exact_search_not_repeated": dense["historical_exact_search_not_repeated"],
        },
        "selected_case_level_milp": milp_summaries(),
        "terminal_alternatives": {
            "QLinearConv_or_ConvInteger": "rejected_by_static_memory_floor",
            "reason": (
                "The graph input is float [1,10,30,30]. A quantized/integer terminal requires "
                "a uint8/int8 intermediate of 9000 elements before the terminal, so intermediate "
                "memory alone is at least 9000, already above cost 710."
            ),
            "minimum_quantized_input_intermediate_elements": math.prod(EXPECTED_IO),
        },
        "policy": {
            "public_fixture_correction": False,
            "lookup_table": False,
            "shape_cloak": False,
            "root_or_stage_modified": False,
            "try_candidate_used": False,
            "kimi_used": False,
        },
    }
    (HERE / "evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "candidate_cost": candidate_summary["cost"],
                "known": sorted({(item["right"], item["total"]) for item in all_known}),
                "domain": sorted({(item["right"], item["total"]) for item in all_domain}),
                "fresh": [(item["seed"], item["right"], item["total"]) for item in all_fresh],
                "runtime_fault_free": no_runtime_faults,
                "evidence": str((HERE / "evidence.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

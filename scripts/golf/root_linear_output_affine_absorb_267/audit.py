#!/usr/bin/env python3
"""Fail-closed audit for lane 267's strict-lower affine-absorption candidates."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
SCAN = HERE / "scan.json"
OUTPUT = HERE / "audit.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
FRESH_BASE_SEEDS = (267_000_001, 267_000_002)
FRESH_CASES = 1000
CONFIGS = (
    ("disable_t1", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 1),
    ("disable_t4", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, 4),
    ("enable_t1", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 1),
    ("enable_t4", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, 4),
)
BANNED = {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
from golf.check_conv_bias import check_model as check_conv_bias  # noqa: E402
from lib import scoring  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def raw_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
    )


def make_session(data: bytes, level: Any, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    options.graph_optimization_level = level
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_cases(task: int) -> tuple[list[dict[str, np.ndarray]], int]:
    result = []
    skipped = 0
    for examples in scoring.load_examples(task).values():
        for example in examples:
            converted = scoring.convert_to_numpy(example)
            if converted is None:
                skipped += 1
            else:
                result.append(converted)
    if not result:
        raise RuntimeError(f"task{task:03d} has no convertible known cases")
    return result, skipped


def fresh_cases(task: int, seed: int, count: int) -> tuple[list[dict[str, np.ndarray]], int, int]:
    mapping = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    generator = importlib.import_module(f"task_{mapping[f'{task:03d}']}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    result = []
    attempts = 0
    skips = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"generator stalled: task{task:03d} {len(result)}/{count}")
        converted = scoring.convert_to_numpy(generator.generate())
        if converted is None:
            skips += 1
        else:
            result.append(converted)
    return result, attempts, skips


def compare(authority: bytes, candidate: bytes, cases: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, level, threads in CONFIGS:
        row: dict[str, Any] = {
            "cases": len(cases),
            "raw_equal": 0,
            "threshold_equal": 0,
            "authority_correct": 0,
            "candidate_correct": 0,
            "authority_errors": 0,
            "candidate_errors": 0,
            "authority_nonfinite": 0,
            "candidate_nonfinite": 0,
            "first_failure": None,
        }
        try:
            authority_session = make_session(authority, level, threads)
            candidate_session = make_session(candidate, level, threads)
        except Exception as exc:  # noqa: BLE001
            row["session_error"] = f"{type(exc).__name__}: {exc}"
            result[label] = row
            continue
        a_input = authority_session.get_inputs()[0].name
        c_input = candidate_session.get_inputs()[0].name
        a_output = authority_session.get_outputs()[0].name
        c_output = candidate_session.get_outputs()[0].name
        for case_index, case in enumerate(cases):
            try:
                base = authority_session.run([a_output], {a_input: case["input"]})[0]
            except Exception as exc:  # noqa: BLE001
                row["authority_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "case": case_index,
                    "authority_error": f"{type(exc).__name__}: {exc}",
                }
                continue
            try:
                got = candidate_session.run([c_output], {c_input: case["input"]})[0]
            except Exception as exc:  # noqa: BLE001
                row["candidate_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "case": case_index,
                    "candidate_error": f"{type(exc).__name__}: {exc}",
                }
                continue
            equal = raw_equal(base, got)
            threshold = np.array_equal(base > 0, got > 0)
            base_correct = np.array_equal(base > 0, case["output"] > 0)
            got_correct = np.array_equal(got > 0, case["output"] > 0)
            row["raw_equal"] += int(equal)
            row["threshold_equal"] += int(threshold)
            row["authority_correct"] += int(base_correct)
            row["candidate_correct"] += int(got_correct)
            if np.issubdtype(base.dtype, np.floating):
                row["authority_nonfinite"] += int(np.count_nonzero(~np.isfinite(base)))
            if np.issubdtype(got.dtype, np.floating):
                row["candidate_nonfinite"] += int(np.count_nonzero(~np.isfinite(got)))
            if row["first_failure"] is None and not equal:
                row["first_failure"] = {
                    "case": case_index,
                    "different_elements": int(np.count_nonzero(base != got)),
                    "threshold_equal": bool(threshold),
                    "authority_correct": bool(base_correct),
                    "candidate_correct": bool(got_correct),
                }
        row["raw_pass"] = bool(
            row["raw_equal"] == row["cases"]
            and row["threshold_equal"] == row["cases"]
            and row["authority_errors"] == 0
            and row["candidate_errors"] == 0
            and row["authority_nonfinite"] == 0
            and row["candidate_nonfinite"] == 0
            and "session_error" not in row
        )
        result[label] = row
    return result


def comparison_pass(report: dict[str, Any], require_gold: bool) -> bool:
    return all(
        row.get("raw_pass") is True
        and (not require_gold or row["candidate_correct"] == row["cases"])
        for row in report.values()
    )


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                count += 1
                pending.extend(attr.g.node)
            elif attr.type == AttributeProto.GRAPHS:
                count += len(attr.graphs)
                for graph in attr.graphs:
                    pending.extend(graph.node)
    return count


def static_audit(model: onnx.ModelProto) -> dict[str, Any]:
    row: dict[str, Any] = {}
    try:
        onnx.checker.check_model(copy.deepcopy(model), full_check=True)
        row["full_check"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(full_check=False, checker_error=f"{type(exc).__name__}: {exc}")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        nonstatic = []
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
            if value.type.HasField("tensor_type") and any(
                not dim.HasField("dim_value") or dim.dim_value <= 0
                for dim in value.type.tensor_type.shape.dim
            ):
                nonstatic.append(value.name)
        row["strict_data_prop"] = True
        row["nonstatic"] = nonstatic
    except Exception as exc:  # noqa: BLE001
        row.update(strict_data_prop=False, strict_error=f"{type(exc).__name__}: {exc}", nonstatic=[])
    row["banned_ops"] = sorted({
        node.op_type for node in model.graph.node
        if node.op_type in BANNED or "Sequence" in node.op_type
    })
    row["nested_graphs"] = nested_graph_count(model)
    row["nonstandard_domains"] = sorted({item.domain for item in model.opset_import if item.domain})
    row["functions"] = len(model.functions)
    row["sparse_initializers"] = len(model.graph.sparse_initializer)
    row["external_initializers"] = [
        item.name for item in model.graph.initializer
        if item.data_location == TensorProto.EXTERNAL or item.external_data
    ]
    try:
        row["conv_bias_findings"] = check_conv_bias(model)
        row["conv_bias_ub0"] = not row["conv_bias_findings"]
    except Exception as exc:  # noqa: BLE001
        row.update(conv_bias_ub0=False, conv_bias_error=f"{type(exc).__name__}: {exc}")
    row["pass"] = bool(
        row.get("full_check") and row.get("strict_data_prop") and not row.get("nonstatic")
        and not row["banned_ops"] and row["nested_graphs"] == 0
        and not row["nonstandard_domains"] and row["functions"] == 0
        and row["sparse_initializers"] == 0 and not row["external_initializers"]
        and row.get("conv_bias_ub0")
    )
    return row


def declared_shape(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def truthful_shapes(data: bytes, case: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run the unsanitized model with every statically typed node value exposed."""
    try:
        model = onnx.load_model_from_string(data)
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
        typed = {
            value.name: value
            for value in [*inferred.graph.value_info, *inferred.graph.output]
            if value.type.HasField("tensor_type")
            and all(
                dim.HasField("dim_value") and dim.dim_value > 0
                for dim in value.type.tensor_type.shape.dim
            )
        }
        names = []
        for node in inferred.graph.node:
            for name in node.output:
                if name and name in typed and name not in names:
                    names.append(name)
        exposed = copy.deepcopy(inferred)
        del exposed.graph.output[:]
        exposed.graph.output.extend(copy.deepcopy(typed[name]) for name in names)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.log_severity_level = 4
        runtime = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = runtime.run(names, {runtime.get_inputs()[0].name: case["input"]})
        mismatches = [
            {
                "tensor": name,
                "declared": declared_shape(typed[name]),
                "actual": list(value.shape),
            }
            for name, value in zip(names, actual)
            if declared_shape(typed[name]) != list(value.shape)
        ]
        nonfinite = sum(
            int(np.count_nonzero(~np.isfinite(value)))
            for value in actual if np.issubdtype(value.dtype, np.floating)
        )
        return {
            "traced_outputs": len(names),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "nonfinite_elements": nonfinite,
            "truthful": not mismatches and nonfinite == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def official(task: int, model: onnx.ModelProto) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"linear267_{task:03d}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, work,
            label=f"linear267_task{task:03d}", require_correct=True,
        )


def initializer_audit(authority: onnx.ModelProto, candidate: onnx.ModelProto) -> dict[str, Any]:
    def table(model: onnx.ModelProto) -> dict[str, bytes]:
        return {item.name: item.SerializeToString() for item in model.graph.initializer}

    before = table(authority)
    after = table(candidate)
    changed = sorted(
        name for name in before.keys() & after.keys() if before[name] != after[name]
    )
    return {
        "authority_count": len(before),
        "candidate_count": len(after),
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": changed,
        "all_serialized_initializers_identical": (
            before.keys() == after.keys() and not changed
        ),
        "serialized_dtype_roundtrip_required": False,
        "shared_weight_or_scalar_mutated": False,
    }


def strict_lower_rows() -> list[dict[str, Any]]:
    payload = json.loads(SCAN.read_text(encoding="utf-8"))
    return [row for row in payload["candidates"] if row.get("strict_lower")]


def main() -> None:
    if digest(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority ZIP changed")
    output: dict[str, Any] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "ort_version": ort.__version__,
        "fresh_base_seeds": list(FRESH_BASE_SEEDS),
        "fresh_cases_per_seed": FRESH_CASES,
        "configs": [label for label, _level, _threads in CONFIGS],
        "tasks": [],
        "known_shape_cloak_authorities": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for selected in strict_lower_rows():
            task = int(selected["task"])
            authority = archive.read(f"task{task:03d}.onnx")
            candidate_path = ROOT / selected["path"]
            candidate = candidate_path.read_bytes()
            authority_model = onnx.load_model_from_string(authority)
            candidate_model = onnx.load_model_from_string(candidate)
            cases, known_skips = known_cases(task)
            static = static_audit(candidate_model)
            official_result = official(task, candidate_model) if static.get("pass") else None
            truth = (
                truthful_shapes(candidate, cases[0])
                if static.get("strict_data_prop") else {"truthful": False, "not_run": "strict inference failed"}
            )
            known = (
                compare(authority, candidate, cases)
                if static.get("pass") and truth.get("truthful") else {}
            )
            known_pass = bool(known) and comparison_pass(known, require_gold=True)
            official_ok = bool(
                official_result and official_result.get("correct")
                and int(official_result.get("cost", -1))
                == int(selected["candidate_profile"]["cost"])
            )
            row: dict[str, Any] = {
                "task": task,
                "path": selected["path"],
                "authority_sha256": digest(authority),
                "candidate_sha256": digest(candidate),
                "baseline_profile": selected["baseline_profile"],
                "candidate_profile": selected["candidate_profile"],
                "activation_bytes_removed": selected["activation_bytes_removed"],
                "static": static,
                "initializer_audit": initializer_audit(authority_model, candidate_model),
                "official": official_result,
                "official_ok": official_ok,
                "truthful_shapes": truth,
                "known_cases": len(cases),
                "known_conversion_skips": known_skips,
                "known_four_config": known,
                "known_pass": known_pass,
                "known_not_run_reason": (
                    None if known else "truthful runtime-shape gate failed"
                ),
                "fresh": [],
            }
            if known_pass and official_ok:
                for base_seed in FRESH_BASE_SEEDS:
                    seed = base_seed + task
                    fresh, attempts, skips = fresh_cases(task, seed, FRESH_CASES)
                    report = compare(authority, candidate, fresh)
                    row["fresh"].append({
                        "seed": seed,
                        "attempts": attempts,
                        "conversion_skips": skips,
                        "four_config": report,
                        "raw_pass": comparison_pass(report, require_gold=False),
                    })
            row["fresh_not_run_reason"] = (
                None if row["fresh"] else "known4 raw gate was not reached/passed"
            )
            row["accepted"] = bool(
                static.get("pass") and truth.get("truthful") and official_ok and known_pass
                and len(row["fresh"]) == len(FRESH_BASE_SEEDS)
                and all(item["raw_pass"] for item in row["fresh"])
            )
            output["tasks"].append(row)
            print(json.dumps({
                "task": task,
                "static": static.get("pass"),
                "official": official_ok,
                "truthful": truth.get("truthful"),
                "runtime_shape_mismatches": truth.get("mismatch_count"),
                "known": known_pass,
                "fresh_runs": len(row["fresh"]),
                "accepted": row["accepted"],
            }), flush=True)
        scan = json.loads(SCAN.read_text(encoding="utf-8"))
        task367_exclusion = scan["task367_single_use_exclusion"][0]
        for task in (54, 367):
            authority = archive.read(f"task{task:03d}.onnx")
            cases, skips = known_cases(task)
            authority_model = onnx.load_model_from_string(authority)
            static = static_audit(authority_model)
            trace = (
                truthful_shapes(authority, cases[0])
                if static.get("strict_data_prop") else {"truthful": False, "not_run": "strict inference failed"}
            )
            output["known_shape_cloak_authorities"].append({
                "task": task,
                "authority_sha256": digest(authority),
                "known_cases": len(cases),
                "known_conversion_skips": skips,
                "static": static,
                "truthful_shapes": trace,
                "single_use_exclusion": task367_exclusion if task == 367 else None,
                "fail_closed": not trace.get("truthful", False),
            })
            print(json.dumps({
                "shape_cloak_task": task,
                "strict": static.get("strict_data_prop"),
                "truthful": trace.get("truthful"),
                "runtime_shape_mismatches": trace.get("mismatch_count"),
            }), flush=True)
    output["accepted_tasks"] = [row["task"] for row in output["tasks"] if row["accepted"]]
    output["rejected_tasks"] = [row["task"] for row in output["tasks"] if not row["accepted"]]
    output["all_errors"] = sum(
        int(config.get("authority_errors", 0)) + int(config.get("candidate_errors", 0))
        for row in output["tasks"]
        for group in [row.get("known_four_config", {})]
        + [item["four_config"] for item in row.get("fresh", [])]
        for config in group.values()
    )
    output["all_nonfinite"] = sum(
        int(config.get("authority_nonfinite", 0)) + int(config.get("candidate_nonfinite", 0))
        for row in output["tasks"]
        for group in [row.get("known_four_config", {})]
        + [item["four_config"] for item in row.get("fresh", [])]
        for config in group.values()
    )
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted_tasks": output["accepted_tasks"],
        "rejected_tasks": output["rejected_tasks"],
        "all_errors": output["all_errors"],
        "all_nonfinite": output["all_nonfinite"],
    }, indent=2))


if __name__ == "__main__":
    main()

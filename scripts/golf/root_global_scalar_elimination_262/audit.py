#!/usr/bin/env python3
"""Fail-closed audit for global scalar-elimination lower survivors."""

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
SCAN = HERE / "scan.json"
OUTPUT = HERE / "audit.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
FRESH_BASE_SEEDS = (262_000_001, 262_000_002)
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
from golf.rank_dir import cost_of  # noqa: E402
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
    for split, examples in scoring.load_examples(task).items():
        for index, example in enumerate(examples):
            converted = scoring.convert_to_numpy(example)
            if converted is None:
                skipped += 1
                continue
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
    conversion_skips = 0
    while len(result) < count:
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"generator stalled: task{task:03d} {len(result)}/{count}")
        converted = scoring.convert_to_numpy(generator.generate())
        if converted is None:
            conversion_skips += 1
            continue
        result.append(converted)
    return result, attempts, conversion_skips


def compare(
    authority: bytes,
    candidate: bytes,
    cases: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    result = {}
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
                    "case": case_index, "authority_error": f"{type(exc).__name__}: {exc}",
                }
                continue
            try:
                got = candidate_session.run([c_output], {c_input: case["input"]})[0]
            except Exception as exc:  # noqa: BLE001
                row["candidate_errors"] += 1
                row["first_failure"] = row["first_failure"] or {
                    "case": case_index, "candidate_error": f"{type(exc).__name__}: {exc}",
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
                    "raw_equal": False,
                    "threshold_equal": threshold,
                    "authority_correct": base_correct,
                    "candidate_correct": got_correct,
                    "different_elements": int(np.count_nonzero(base != got)),
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
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
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
    ops = [node.op_type for node in model.graph.node]
    row["banned_ops"] = sorted({op for op in ops if op in BANNED or "Sequence" in op})
    row["nested_graphs"] = nested_graph_count(model)
    row["nonstandard_domains"] = sorted({op.domain for op in model.opset_import if op.domain})
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


def shape_list(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def truthful_shapes(data: bytes, case: dict[str, np.ndarray]) -> dict[str, Any]:
    try:
        model = onnx.load_model_from_string(data)
        inferred = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=True)
        typed = {
            value.name: value
            for value in [*inferred.graph.value_info, *inferred.graph.output]
            if value.type.HasField("tensor_type")
            and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)
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
        options.log_severity_level = 4
        runtime = ort.InferenceSession(
            exposed.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        actual = runtime.run(names, {runtime.get_inputs()[0].name: case["input"]})
        mismatches = [
            {"tensor": name, "declared": shape_list(typed[name]), "actual": list(value.shape)}
            for name, value in zip(names, actual)
            if shape_list(typed[name]) != list(value.shape)
        ]
        nonfinite = sum(
            int(np.count_nonzero(~np.isfinite(value)))
            for value in actual if np.issubdtype(value.dtype, np.floating)
        )
        return {
            "traced_outputs": len(names),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20],
            "nonfinite_elements": nonfinite,
            "truthful": not mismatches and nonfinite == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def official(task: int, model: onnx.ModelProto) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"scalar262_official_{task:03d}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            copy.deepcopy(model), task, work,
            label=f"scalar262_task{task:03d}", require_correct=True,
        )


def best_rows() -> list[dict[str, Any]]:
    payload = json.loads(SCAN.read_text(encoding="utf-8"))
    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in payload["rows"]:
        if row.get("strict_lower"):
            by_task.setdefault(int(row["task"]), []).append(row)
    return [
        min(
            rows,
            key=lambda row: (
                int(row["candidate_profile"]["cost"]),
                -len(row["initializers"]),
                row["label"] != "combined",
            ),
        )
        for _task, rows in sorted(by_task.items())
    ]


def main() -> None:
    output: dict[str, Any] = {
        "ort_version": ort.__version__,
        "fresh_base_seeds": list(FRESH_BASE_SEEDS),
        "fresh_cases_per_seed": FRESH_CASES,
        "configs": [label for label, _level, _threads in CONFIGS],
        "tasks": [],
    }
    with zipfile.ZipFile(AUTHORITY) as archive:
        for selected in best_rows():
            task = int(selected["task"])
            authority = archive.read(f"task{task:03d}.onnx")
            candidate_path = ROOT / selected["path"]
            candidate = candidate_path.read_bytes()
            candidate_model = onnx.load_model_from_string(candidate)
            cases, known_skips = known_cases(task)
            static = static_audit(candidate_model)
            truth = truthful_shapes(candidate, cases[0]) if static.get("strict_data_prop") else {"truthful": False}
            official_result = official(task, candidate_model) if static.get("pass") else None
            known = compare(authority, candidate, cases) if static.get("pass") and truth.get("truthful") else {}
            known_pass = bool(known) and comparison_pass(known, require_gold=True)
            row: dict[str, Any] = {
                "task": task,
                "label": selected["label"],
                "initializers_removed": selected["initializers"],
                "path": selected["path"],
                "authority_sha256": digest(authority),
                "candidate_sha256": digest(candidate),
                "baseline_profile": selected["baseline_profile"],
                "candidate_profile": selected["candidate_profile"],
                "static": static,
                "truthful_shapes": truth,
                "official": official_result,
                "known_cases": len(cases),
                "known_conversion_skips": known_skips,
                "known_four_config": known,
                "known_pass": known_pass,
                "fresh": [],
            }
            official_ok = bool(
                official_result
                and official_result.get("correct")
                and int(official_result.get("cost", -1)) == int(selected["candidate_profile"]["cost"])
            )
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
                    print(json.dumps({
                        "task": task, "seed": seed,
                        "fresh_raw_pass": row["fresh"][-1]["raw_pass"],
                    }), flush=True)
            row["accepted"] = bool(
                static.get("pass")
                and truth.get("truthful")
                and official_ok
                and known_pass
                and len(row["fresh"]) == len(FRESH_BASE_SEEDS)
                and all(item["raw_pass"] for item in row["fresh"])
            )
            output["tasks"].append(row)
            print(json.dumps({
                "task": task,
                "static": static.get("pass"),
                "truthful": truth.get("truthful"),
                "official": official_ok,
                "known": known_pass,
                "accepted": row["accepted"],
            }), flush=True)
    output["accepted_tasks"] = [row["task"] for row in output["tasks"] if row["accepted"]]
    output["rejected_tasks"] = [row["task"] for row in output["tasks"] if not row["accepted"]]
    output["all_errors"] = sum(
        int(config.get("authority_errors", 0)) + int(config.get("candidate_errors", 0))
        for row in output["tasks"]
        for group in [row.get("known_four_config", {})] + [item["four_config"] for item in row.get("fresh", [])]
        for config in group.values()
    )
    output["all_nonfinite"] = sum(
        int(config.get("authority_nonfinite", 0)) + int(config.get("candidate_nonfinite", 0))
        for row in output["tasks"]
        for group in [row.get("known_four_config", {})] + [item["four_config"] for item in row.get("fresh", [])]
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

#!/usr/bin/env python3
"""Screen every strict-lower history model under the user's POLICY95 rule."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = HERE / "strict_history_evidence.json"
OUT = HERE / "policy95_history_evidence.json"
THRESHOLD = 0.95


def load_support():
    path = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
    spec = importlib.util.spec_from_file_location("policy95_307_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(source: str) -> bytes:
    if "!" not in source:
        return (ROOT / source).read_bytes()
    archive, member = source.rsplit("!", 1)
    with zipfile.ZipFile(ROOT / archive) as handle:
        return handle.read(member)


def compact(row):
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "sign_sha256", "raw_sha256", "first_wrong", "first_error", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def row_pass(row, threshold=THRESHOLD):
    return bool(
        float(row.get("accuracy", 0.0)) >= threshold
        and row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0 and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def structure_audit(support, task, model, data):
    reasons = []
    full_error = strict_error = None
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception as exc:
        full_error = f"{type(exc).__name__}: {exc}"
        reasons.append("full_checker")
    try:
        inferred = onnx.shape_inference.infer_shapes(
            copy.deepcopy(model), strict_mode=True, data_prop=True
        )
    except Exception as exc:
        inferred = None
        strict_error = f"{type(exc).__name__}: {exc}"
        reasons.append("strict_shape")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        reasons.append("io_count")
    else:
        def shape(value):
            return [int(dim.dim_value) if dim.HasField("dim_value") else None
                    for dim in value.type.tensor_type.shape.dim]
        if model.graph.input[0].name != "input" or shape(model.graph.input[0]) != [1, 10, 30, 30]:
            reasons.append("input_io")
        if model.graph.output[0].name != "output" or shape(model.graph.output[0]) != [1, 10, 30, 30]:
            reasons.append("output_io")
    for value in ([] if inferred is None else
                  list(inferred.graph.input) + list(inferred.graph.output) + list(inferred.graph.value_info)):
        if not value.type.HasField("tensor_type"):
            continue
        if any(dim.HasField("dim_param") or not dim.HasField("dim_value") or dim.dim_value <= 0
               for dim in value.type.tensor_type.shape.dim):
            reasons.append("nonstatic_shape")
    nonfinite = []
    for tensor in model.graph.initializer:
        try:
            if not np.all(np.isfinite(onnx.numpy_helper.to_array(tensor))):
                nonfinite.append(tensor.name)
        except Exception:
            nonfinite.append(tensor.name)
    if nonfinite:
        reasons.append("nonfinite_initializer")
    banned = []
    nested = 0
    domains = []
    for node in model.graph.node:
        if node.op_type in {"Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress"} or "Sequence" in node.op_type:
            banned.append(node.op_type)
        if node.domain not in ("", "ai.onnx"):
            domains.append(node.domain)
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested += 1
    if banned or nested or domains or model.functions or model.graph.sparse_initializer:
        reasons.append("banned_or_nonstandard")
    bias = list(support.check_conv_bias(copy.deepcopy(model)))
    if bias:
        reasons.append("conv_bias_ub")
    trace = None
    if not reasons:
        try:
            trace = support.runtime_shape_trace(task, copy.deepcopy(model))
            if trace.get("error") or trace.get("declared_actual_mismatches"):
                reasons.append("runtime_shape_cloak")
        except Exception as exc:
            trace = {"error": f"{type(exc).__name__}: {exc}"}
            reasons.append("runtime_shape_cloak")
    return {
        "pass": not reasons, "reasons": sorted(set(reasons)),
        "full_check": full_error is None, "full_error": full_error,
        "strict_shape": strict_error is None, "strict_error": strict_error,
        "nonfinite_initializers": nonfinite, "banned_ops": sorted(set(banned)),
        "nonstandard_domains": sorted(set(domains)), "nested_graphs": nested,
        "functions": len(model.functions), "sparse_initializers": len(model.graph.sparse_initializer),
        "conv_bias_ub": bias, "runtime_shape_trace": trace,
        "max_einsum_fanin": max([len(node.input) for node in model.graph.node
                                  if node.op_type == "Einsum"] or [0]),
    }


def fast_profile(support, task, model, first_case):
    """Official cost from one profiled inference; correctness is audited separately."""
    sanitized = support.scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        return None
    with tempfile.TemporaryDirectory(prefix=f"policy95_cost_{task:03d}_", dir="/tmp") as tmp:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.profile_file_prefix = str(Path(tmp) / "trace")
        try:
            runtime = ort.InferenceSession(
                sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            benchmark = support.scoring.convert_to_numpy(first_case)
            runtime.run(["output"], {"input": benchmark["input"]})
            trace = runtime.end_profiling()
            memory = support.scoring.calculate_memory(sanitized, trace)
            params = support.scoring.calculate_params(sanitized)
        except Exception:
            return None
        if memory is None or params is None or memory < 0 or params < 0:
            return None
        cost = int(memory + params)
        return {
            "memory": int(memory), "params": int(params), "cost": cost,
            "score": max(1.0, 25.0 - math.log(max(1, cost))), "correct": None,
        }


def main():
    started = time.monotonic()
    support = load_support()
    support.POLICY_THRESHOLD = THRESHOLD
    ledger = json.loads(EVIDENCE.read_text())
    records = [row for row in ledger["results"] if not row["structural_reasons"]]
    known_cache = {}
    known_screen = []
    eligible = []
    for index, row in enumerate(records, 1):
        task = int(row["task"])
        item = dict(row)
        existing_profile = row.get("profile")
        # The strict pass already did full known verification and official
        # profiling for exact rows.  All exact non-winners can be excluded
        # without rerunning hundreds of examples.
        if row.get("known_exact") and existing_profile:
            item["known_disable_threads1"] = {
                "total": int(row.get("checked", 0)), "right": int(row.get("checked", 0)),
                "wrong": 0, "accuracy": 1.0, "errors": 0,
                "nonfinite_cases": 0, "nonfinite_elements": 0,
                "runtime_shape_mismatches": 0, "small_positive_elements_0_to_0_25": 0,
                "sign_mismatch_cases_vs_disable_threads1": 0,
                "sign_mismatch_cells_vs_disable_threads1": 0,
            }
            item["known_policy95"] = True
            known_screen.append(item)
            if int(existing_profile["cost"]) >= int(row["authority_cost"]):
                if index % 50 == 0:
                    print(json.dumps({"i": index, "n": len(records), "task": task,
                                      "known_accuracy": 1.0, "eligible": len(eligible)}), flush=True)
                continue
            data = resolve(str(row["source"]))
            model = onnx.load_model_from_string(data)
            structure = structure_audit(support, task, model, data)
            item["structure"] = structure
            if structure["pass"]:
                item["data"] = data
                eligible.append(item)
            continue
        try:
            data = resolve(str(row["source"]))
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("sha mismatch")
            runtime = support.make_session(data, True, 1)
            if task not in known_cache:
                known_cache[task] = support.known_cases(task)
            cases, counts = known_cache[task]
            known_row, _ = support.evaluate_config(runtime, cases, None)
            item["known_counts"] = counts
            item["known_disable_threads1"] = compact(known_row)
        except Exception as exc:
            item["known_disable_threads1"] = {
                "total": 0, "right": 0, "accuracy": 0.0, "errors": 1,
                "session_error": f"{type(exc).__name__}: {exc}",
            }
            known_row = item["known_disable_threads1"]
            data = None
        item["known_policy95"] = row_pass(known_row)
        known_screen.append(item)
        if item["known_policy95"] and data is not None:
            model = onnx.load_model_from_string(data)
            cases, _ = known_cache[task]
            profile = fast_profile(support, task, model, cases[0])
            item["profile"] = profile
            if profile and int(profile["cost"]) < int(row["authority_cost"]):
                structure = structure_audit(support, task, model, data)
                item["structure"] = structure
                if structure["pass"]:
                    item["data"] = data
                    eligible.append(item)
        if index % 50 == 0 or item["known_policy95"]:
            print(json.dumps({"i": index, "n": len(records), "task": task,
                              "known_accuracy": known_row.get("accuracy"),
                              "eligible": len(eligible)}), flush=True)

    grouped = defaultdict(list)
    for row in eligible:
        grouped[int(row["task"])].append(row)
    fresh_results = []
    finalists = []
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    outdir = HERE / "policy95_history_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    for task, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (int(row["profile"]["cost"]), row["sha256"]))
        for rank, item in enumerate(rows[:5], 1):
            data = item.pop("data")
            known_cases, _ = known_cache[task]
            known_four = support.evaluate_four(data, known_cases)
            fresh_runs = []
            for seed in (307_200_000 + task, 307_300_000 + task):
                cases, generation = support.fresh_cases(task, seed, task_map)
                runtime = support.evaluate_four(data, cases)
                fresh_runs.append({
                    "seed": seed, "generation": generation,
                    "runtime": {name: compact(row) for name, row in runtime.items()},
                    "pass": all(row_pass(row) for row in runtime.values()),
                })
            result = {key: value for key, value in item.items() if key != "data"}
            result["known_four"] = {name: compact(row) for name, row in known_four.items()}
            result["known_four_pass"] = all(row_pass(row) for row in known_four.values())
            result["fresh"] = fresh_runs
            result["policy95_pass"] = bool(
                result["known_four_pass"] and all(run["pass"] for run in fresh_runs)
            )
            result["meets_half"] = int(result["profile"]["cost"]) * 2 <= int(result["authority_cost"])
            fresh_results.append(result)
            print(json.dumps({"task": task, "rank": rank, "cost": result["profile"]["cost"],
                              "fresh": [run["runtime"]["disable_threads1"]["accuracy"]
                                        for run in fresh_runs],
                              "pass": result["policy95_pass"]}), flush=True)
            if result["policy95_pass"]:
                path = outdir / f"task{task:03d}_cost{result['profile']['cost']}_{result['sha256'][:12]}_POLICY95.onnx"
                path.write_bytes(data)
                result["saved_path"] = str(path.relative_to(ROOT))
                finalists.append(result)
                break

    for row in eligible:
        row.pop("data", None)
    payload = {
        "threshold": THRESHOLD, "history_records": len(records),
        "known_policy95_count": sum(row["known_policy95"] for row in known_screen),
        "profiled_structure_eligible_count": len(eligible),
        "fresh_audited_count": len(fresh_results), "finalist_count": len(finalists),
        "half_finalists": [row for row in finalists if row["meets_half"]],
        "strict_nonhalf_finalists": [row for row in finalists if not row["meets_half"]],
        "finalists": finalists, "fresh_results": fresh_results,
        "known_screen": known_screen,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"history_records": len(records), "known_policy95": payload["known_policy95_count"],
                      "eligible": len(eligible), "finalists": [
                          {"task": r["task"], "cost": r["profile"]["cost"],
                           "authority_cost": r["authority_cost"], "meets_half": r["meets_half"]}
                          for r in finalists], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

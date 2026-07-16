#!/usr/bin/env python3
"""Full screen for the incremental SHAs found by lane 94.

Policy/private/nonfinite/giant markers are diagnostics, not automatic rejects.
Schema, unsupported runtime, Conv-family UB, known correctness, and truthful
runtime shapes remain fail-closed.  Fresh results rank LB probes only.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUTHORITY = ROOT / "submission.zip"
AUTHORITY_SHA256 = "9085e2f795c0a73d44d27d712ad8fbaad67a3f37b1d8363a0f33305fbafa4118"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
INVENTORY = HERE / "inventory_delta.json"
TARGETS = (102, 25, 250, 62, 324, 308, 8, 275, 338, 333, 268, 184, 377, 109, 160, 99, 279, 345, 170, 245)
CURRENT_COSTS = {
    102: 493, 25: 474, 250: 468, 62: 465, 324: 439, 308: 434,
    8: 431, 275: 428, 338: 426, 333: 423, 268: 422, 184: 421,
    377: 409, 109: 405, 160: 404, 99: 398, 279: 397, 345: 389,
    170: 387, 245: 387,
}
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
KNOWN_BLACK_TASKS = {18, 48, 112, 134, 168, 198, 233, 251, 277, 286, 365, 366}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from harvest import actual_screen, exact_conv_bias_gate, known_score  # noqa: E402
from screen_all import resolve_source  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"])


def structural_audit(data: bytes) -> dict[str, Any]:
    report: dict[str, Any] = {"pass": False, "hard_failures": [], "policy_markers": []}
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        report["hard_failures"].append(f"checker_or_strict_shape:{type(exc).__name__}:{exc}")
        return report
    if model.functions:
        report["hard_failures"].append("local_functions")
    if model.graph.sparse_initializer:
        report["hard_failures"].append("sparse_initializer")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        report["hard_failures"].append("noncanonical_io_count")
    elif model.graph.input[0].name != "input" or model.graph.output[0].name != "output":
        report["hard_failures"].append("noncanonical_io_names")
    negative_pads = []
    op_hist = Counter()
    nested = []
    for node in model.graph.node:
        op_hist[node.op_type] += 1
        upper = node.op_type.upper()
        if node.domain not in ("", "ai.onnx"):
            report["hard_failures"].append(f"custom_domain:{node.domain}")
        if upper in BANNED or "SEQUENCE" in upper:
            report["hard_failures"].append(f"banned_op:{node.op_type}")
        if node.op_type == "Einsum" and len(node.input) >= 15:
            report["policy_markers"].append(f"giant_einsum:{len(node.input)}")
        if node.op_type in {"TfIdfVectorizer", "Hardmax"}:
            report["policy_markers"].append(f"lookup_op:{node.op_type}")
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                nested.append(node.output[0] if node.output else node.name)
            if node.op_type in {"Conv", "ConvTranspose"} and attr.name == "pads" and any(v < 0 for v in attr.ints):
                negative_pads.append({"output": node.output[0], "pads": list(attr.ints)})
    if nested:
        report["hard_failures"].append("nested_graph")
    if negative_pads:
        report["hard_failures"].append("negative_conv_pads")
    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            report["hard_failures"].append(f"custom_opset:{opset.domain}")
    nonstatic = []
    initializer_names = {item.name for item in inferred.graph.initializer}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type") or any(
            not dim.HasField("dim_value") or dim.dim_value <= 0
            for dim in value.type.tensor_type.shape.dim
        ):
            nonstatic.append(value.name)
    if nonstatic:
        report["hard_failures"].append("nonstatic_shapes")
    nonfinite = []
    giant_initializers = []
    for item in model.graph.initializer:
        if item.data_location == onnx.TensorProto.EXTERNAL or item.external_data:
            report["hard_failures"].append("external_initializer")
        try:
            array = numpy_helper.to_array(item)
            if array.size >= 10_000:
                giant_initializers.append(item.name)
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                nonfinite.append(item.name)
        except Exception as exc:  # noqa: BLE001
            report["hard_failures"].append(f"initializer_read:{item.name}:{type(exc).__name__}")
    if giant_initializers:
        report["policy_markers"].append("giant_initializer_lookup")
    if nonfinite:
        report["policy_markers"].append("nonfinite_initializer")
    bias_ok, bias_reason, bias_findings = exact_conv_bias_gate(model)
    if not bias_ok:
        report["hard_failures"].append(bias_reason or "conv_bias_ub")
    report.update({
        "pass": not report["hard_failures"],
        "op_histogram": dict(op_hist),
        "negative_conv_pads": negative_pads,
        "nonstatic_shapes": nonstatic,
        "nonfinite_initializers": nonfinite,
        "giant_initializers": giant_initializers,
        "conv_bias_ub0": bias_ok,
        "conv_bias_findings": bias_findings,
    })
    report["hard_failures"] = sorted(set(report["hard_failures"]))
    report["policy_markers"] = sorted(set(report["policy_markers"]))
    return report


def known_four(task: int, data: bytes) -> dict[str, Any]:
    output = {}
    examples = scoring.load_examples(task)
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        for threads in (1, 4):
            key = f"{mode}_threads{threads}"
            stats = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
            try:
                session = make_session(data, disabled, threads)
            except Exception as exc:  # noqa: BLE001
                stats["session_error"] = f"{type(exc).__name__}: {exc}"
                output[key] = stats
                continue
            for subset in ("train", "test", "arc-gen"):
                for example in examples[subset]:
                    benchmark = scoring.convert_to_numpy(example)
                    if benchmark is None:
                        continue
                    try:
                        raw = session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                        if np.array_equal(raw > 0, benchmark["output"] > 0):
                            stats["right"] += 1
                        else:
                            stats["wrong"] += 1
                            if stats["first_failure"] is None:
                                stats["first_failure"] = {"subset": subset}
                    except Exception as exc:  # noqa: BLE001
                        stats["errors"] += 1
                        if stats["first_failure"] is None:
                            stats["first_failure"] = {"error": f"{type(exc).__name__}: {exc}"}
            output[key] = stats
    return output


def known_complete(quad: dict[str, Any]) -> bool:
    return len(quad) == 4 and all(
        item.get("right", 0) > 0
        and item.get("wrong", 0) == 0
        and item.get("errors", 0) == 0
        and not item.get("session_error")
        for item in quad.values()
    )


def fresh_two_seed(task: int, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    task_map = json.loads(TASK_MAP.read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    seeds = (94_000_000 + task, 94_100_000 + task)
    configs = ((True, 1, "disable_all_threads1"), (True, 4, "disable_all_threads4"),
               (False, 1, "default_threads1"), (False, 4, "default_threads4"))
    sessions = {}
    for row in candidates:
        for disabled, threads, name in configs:
            sessions[(row["sha256"], name)] = make_session(row["data"], disabled, threads)
    reports = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        stats = {
            row["sha256"]: {
                name: {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
                for _, _, name in configs
            } for row in candidates
        }
        valid = attempts = generation_errors = conversion_skips = 0
        while valid < 500:
            attempts += 1
            try:
                benchmark = scoring.convert_to_numpy(generator.generate())
            except Exception:  # noqa: BLE001
                generation_errors += 1
                continue
            if benchmark is None:
                conversion_skips += 1
                continue
            valid += 1
            want = benchmark["output"] > 0
            for row in candidates:
                digest = row["sha256"]
                for _, _, name in configs:
                    item = stats[digest][name]
                    try:
                        session = sessions[(digest, name)]
                        raw = session.run(
                            [session.get_outputs()[0].name],
                            {session.get_inputs()[0].name: benchmark["input"]},
                        )[0]
                        if np.array_equal(raw > 0, want):
                            item["right"] += 1
                        else:
                            item["wrong"] += 1
                            if item["first_failure"] is None:
                                item["first_failure"] = {"valid_case": valid, "different_cells": int(np.count_nonzero((raw > 0) != want))}
                    except Exception as exc:  # noqa: BLE001
                        item["errors"] += 1
                        if item["first_failure"] is None:
                            item["first_failure"] = {"valid_case": valid, "error": f"{type(exc).__name__}: {exc}"}
        reports.append({
            "seed": seed, "valid": valid, "attempts": attempts,
            "generation_errors": generation_errors, "conversion_skips": conversion_skips,
            "candidates": stats,
        })
    return {"task": task, "count_per_seed": 500, "seeds": list(seeds), "runs": reports}


def resolve(row: dict[str, Any]) -> tuple[bytes, str]:
    for source in row["sources"]:
        data = resolve_source(source, int(row["task"]))
        if data is not None and sha256(data) == row["sha256"]:
            return data, source
    raise RuntimeError(f"unresolved task{row['task']:03d} {row['sha256']}")


def main() -> int:
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("submission.zip drifted from approved 8006.61 authority")
    inventory = json.loads(INVENTORY.read_text())
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TARGETS}
    # Every target member is unchanged from the previously full-profiled
    # 8005.17 authority, so reusing those official profiles is exact.
    with zipfile.ZipFile(ROOT / "submission_base_8005.17.zip") as old:
        rebase = {
            str(task): {
                "authority_sha256": sha256(authority_payloads[task]),
                "old_sha256": sha256(old.read(f"task{task:03d}.onnx")),
                "byte_identical": authority_payloads[task] == old.read(f"task{task:03d}.onnx"),
                "authority_cost": CURRENT_COSTS[task],
            } for task in TARGETS
        }
    if not all(item["byte_identical"] for item in rebase.values()):
        raise RuntimeError("target authority member changed since predecessor full scans")
    (HERE / "audit/authority_rebase.json").write_text(json.dumps({
        "authority_zip_sha256": AUTHORITY_SHA256,
        "predecessor_zip_sha256": sha256((ROOT / "submission_base_8005.17.zip").read_bytes()),
        "all_target_members_byte_identical": True,
        "tasks": rebase,
    }, indent=2) + "\n")

    rows = []
    qualified = []
    for item in inventory["new_rows"]:
        row = {key: value for key, value in item.items() if key != "data"}
        task = int(row["task"])
        data, source = resolve(row)
        row["resolved_source"] = source
        row["authority_cost"] = CURRENT_COSTS[task]
        row["authority_sha256"] = sha256(authority_payloads[task])
        row["structural_audit"] = structural_audit(data)
        if not row["structural_audit"]["pass"]:
            row["classification"] = "HARD_REJECT_STRUCTURE_OR_UB"
            rows.append(row)
            continue
        row["actual_screen_cost"] = actual_screen(data, task)
        try:
            row["official_profile"] = known_score(data, task, False, f"expand20i94_{task}_{row['sha256'][:8]}")
        except Exception as exc:  # noqa: BLE001
            row["official_profile_error"] = f"{type(exc).__name__}: {exc}"
            row["official_profile"] = None
        profile = row["official_profile"]
        if not profile:
            row["classification"] = "HARD_REJECT_OFFICIAL_RUNTIME"
            rows.append(row)
            continue
        row["candidate_cost"] = int(profile["cost"])
        if row["candidate_cost"] >= CURRENT_COSTS[task]:
            row["classification"] = "REJECT_NOT_STRICTLY_LOWER"
            rows.append(row)
            continue
        row["gain"] = math.log(CURRENT_COSTS[task] / row["candidate_cost"])
        row["known_four"] = known_four(task, data)
        if not profile.get("correct") or not known_complete(row["known_four"]):
            runtime_failure = any(
                item.get("session_error") or item.get("errors", 0)
                for item in row["known_four"].values()
            )
            row["classification"] = (
                "HARD_REJECT_RUNTIME_CONFIG" if runtime_failure else "HARD_REJECT_KNOWN"
            )
            rows.append(row)
            continue
        try:
            row["runtime_shape_trace"] = runtime_shape_trace(task, onnx.load_model_from_string(data))
        except Exception as exc:  # noqa: BLE001
            row["runtime_shape_trace_error"] = f"{type(exc).__name__}: {exc}"
            row["classification"] = "HARD_REJECT_RUNTIME_SHAPE_TRACE"
            rows.append(row)
            continue
        if row["runtime_shape_trace"]["declared_actual_mismatches"]:
            row["classification"] = "HARD_REJECT_SHAPE_CLOAK"
            rows.append(row)
            continue
        row["classification"] = "FRESH_PENDING"
        row["exact_sha_lb_history"] = "NO_EXACT_SHA_LB_RECORD_FOUND"
        row["data"] = data
        qualified.append(row)
        rows.append(row)
        (HERE / "candidates" / f"task{task:03d}_{row['sha256'][:12]}_cost{row['candidate_cost']}.onnx").write_bytes(data)

    fresh_reports = {}
    for task in sorted({int(row["task"]) for row in qualified}):
        task_rows = [row for row in qualified if int(row["task"]) == task]
        report = fresh_two_seed(task, task_rows)
        fresh_reports[str(task)] = report
        for row in task_rows:
            per_config = []
            per_seed = []
            for run in report["runs"]:
                items = run["candidates"][row["sha256"]]
                rates = {key: value["right"] / 500 for key, value in items.items()}
                per_seed.append({"seed": run["seed"], "rates": rates})
                per_config.extend(rates.values())
            row["fresh_two_seed"] = {
                "count_per_seed": 500,
                "seeds": per_seed,
                "minimum_config_rate": min(per_config),
                "maximum_config_rate": max(per_config),
            }
            row["classification"] = "LB_PROBE_REQUIRED"
            row.pop("data", None)
    (HERE / "audit/fresh_two_seed.json").write_text(json.dumps(fresh_reports, indent=2) + "\n")

    probes = sorted(
        [
            {
                "task": row["task"], "sha256": row["sha256"],
                "path": str((HERE / "candidates" / f"task{int(row['task']):03d}_{row['sha256'][:12]}_cost{row['candidate_cost']}.onnx").relative_to(ROOT)),
                "source": row["resolved_source"],
                "authority_cost": row["authority_cost"], "candidate_cost": row["candidate_cost"],
                "gain": row["gain"], "fresh_two_seed": row["fresh_two_seed"],
                "policy_markers": row["structural_audit"]["policy_markers"],
                "exact_sha_lb_history": row["exact_sha_lb_history"],
                "decision": "LB_PROBE_REQUIRED_NOT_ADOPTED",
            }
            for row in rows if row["classification"] == "LB_PROBE_REQUIRED"
        ],
        key=lambda item: (item["task"], item["candidate_cost"], item["sha256"]),
    )
    report = {
        "authority_zip": "submission.zip",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "targets": list(TARGETS),
        "direct_known_black_task_intersection": sorted(set(TARGETS) & KNOWN_BLACK_TASKS),
        "incremental_new_sha_total": inventory["incremental_new_sha_total"],
        "screened_rows": len(rows),
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "lb_probe_required_count": len(probes),
        "rows": rows,
    }
    (HERE / "audit/incremental_screen.json").write_text(json.dumps(report, indent=2) + "\n")
    (HERE / "probe_manifest.json").write_text(json.dumps({
        "status": "LB_PROBE_REQUIRED_CANDIDATES" if probes else "NO_LB_PROBE_REQUIRED_CANDIDATE",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "count": len(probes),
        "candidates": probes,
    }, indent=2) + "\n")
    (HERE / "winner_manifest.json").write_text(json.dumps({
        "status": "NO_FIXED_WINNER",
        "authority_zip_sha256": AUTHORITY_SHA256,
        "count": 0,
        "candidates": [],
        "reason": "No candidate has exact LB-white evidence; local gates only create isolated LB probes.",
    }, indent=2) + "\n")
    print(json.dumps({
        "incremental_new_sha_total": report["incremental_new_sha_total"],
        "classification_counts": report["classification_counts"],
        "lb_probe_required_count": len(probes),
        "probe_tasks": dict(Counter(item["task"] for item in probes)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

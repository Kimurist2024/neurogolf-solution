#!/usr/bin/env python3
"""Three-worker POLICY90 archive/exact scan for authority costs 1001..2500.

The script is intentionally evidence-only: every generated artifact is kept
under this lane.  The root submission, score ledger, and others/ are read-only.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.15.zip"
AUTHORITY_SHA256 = "1dce2bc9bd65497daf59575512972309a3d71b87ee406a97c21fe2ee16010231"
AUTHORITY_LB = 8012.15
THRESHOLD = 0.90
FRESH_PER_SEED = 2_000
MAX_FRESH_CANDIDATES_PER_TASK = 5
RANGE_MIN = 1_001
RANGE_MAX = 2_500
LATEST_LB_BLACK = {70, 134, 202, 343}
# Operational set maintained from docs/golf/private_zero_tasks.md.  The final
# two entries are the separately documented cross-task contamination class.
PRIVATE_ZERO_OR_UNSOUND = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396, 182, 204,
}
SAFE_OPTIMIZER_PASSES = (
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_unused_initializer",
    "eliminate_duplicate_initializer",
    "eliminate_nop_cast",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_concat",
    "eliminate_nop_split",
    "eliminate_nop_expand",
    "eliminate_nop_transpose",
    "eliminate_nop_reshape",
    "eliminate_consecutive_idempotent_ops",
    "fuse_consecutive_transposes",
    "fuse_consecutive_slices",
    "fuse_consecutive_unsqueezes",
    "fuse_consecutive_squeezes",
)

sys.path.insert(0, str(ROOT / "scripts"))
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_support():
    module = load_module(
        f"restart409_support_{os.getpid()}",
        ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py",
    )
    module.POLICY_THRESHOLD = THRESHOLD
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


def task_from_path(path: str) -> int | None:
    match = re.search(r"task[_-]?(\d{3})(?!\d)", path, re.IGNORECASE)
    if match is None:
        match = re.search(r"task(\d{3})", path, re.IGNORECASE)
    return int(match.group(1)) if match else None


def parameter_count(model: onnx.ModelProto) -> int:
    value = scoring.calculate_params(model)
    return int(value) if value is not None else 10**18


def declared_lower_bound(model: onnx.ModelProto) -> int:
    """Safe cheap bound: params plus declared non-output node tensors."""
    total = parameter_count(model)
    graph_outputs = {value.name for value in model.graph.output}
    values = {
        value.name: value
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }
    counted: set[str] = set()
    for node in model.graph.node:
        for name in node.output:
            if not name or name in graph_outputs or name in counted:
                continue
            counted.add(name)
            value = values.get(name)
            if value is None or not value.type.HasField("tensor_type"):
                continue
            tensor = value.type.tensor_type
            dims = [int(dim.dim_value) for dim in tensor.shape.dim]
            if any(dim <= 0 for dim in dims):
                continue
            try:
                itemsize = onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type).itemsize
            except Exception:  # noqa: BLE001
                itemsize = 0
            total += math.prod(dims) * itemsize
    return int(total)


def score_census() -> tuple[list[dict[str, Any]], dict[int, int]]:
    scope: list[dict[str, Any]] = []
    eligible: dict[int, int] = {}
    with (ROOT / "all_scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = int(row["task"].removeprefix("task"))
            cost = int(row["cost"])
            score = float(row["score"])
            if not (RANGE_MIN <= cost <= RANGE_MAX) or score >= 25.0:
                continue
            reasons: list[str] = []
            if task in PRIVATE_ZERO_OR_UNSOUND:
                reasons.append("docs_private_zero_or_unsound_or_contamination")
            if task in LATEST_LB_BLACK:
                reasons.append("latest_lb_black")
            record = {
                "task": task,
                "cost": cost,
                "score": score,
                "excluded": bool(reasons),
                "exclusion_reasons": reasons,
            }
            scope.append(record)
            if not reasons:
                eligible[task] = cost
    scope.sort(key=lambda row: (-int(row["cost"]), int(row["task"])))
    return scope, eligible


def source_class(source: str, kind: str) -> str:
    lowered = source.lower()
    if kind.startswith("exact_"):
        return kind
    if any(token in lowered for token in ("/71502/", "white", "winner", "approved")):
        return "existing_white_lineage"
    return kind


def discover_candidates(
    eligible: dict[int, int], authority_members: dict[int, bytes]
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    started = time.monotonic()
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    raw_counts = defaultdict(int)

    def add(task: int, data: bytes, source: str, kind: str, detail: Any = None) -> None:
        if task not in eligible:
            return
        raw_counts[kind] += 1
        digest = sha256(data)
        key = (task, digest)
        if key in by_key:
            row = by_key[key]
            if source not in row["sources"]:
                row["sources"].append(source)
            label = source_class(source, kind)
            if label not in row["source_classes"]:
                row["source_classes"].append(label)
            return
        try:
            model = onnx.load_model_from_string(data)
            params = parameter_count(model)
            lower = declared_lower_bound(model)
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
            return
        by_key[key] = {
            "task": task,
            "sha256": digest,
            "data": data,
            "file_bytes": len(data),
            "sources": [source],
            "source_classes": [source_class(source, kind)],
            "detail": jsonable(detail),
            "params": params,
            "declared_lower_bound": lower,
            "node_count": len(model.graph.node),
            "op_histogram": dict(
                sorted(
                    {
                        op: sum(node.op_type == op for node in model.graph.node)
                        for op in {node.op_type for node in model.graph.node}
                    }.items()
                )
            ),
        }

    file_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.onnx"], cwd=ROOT, text=True
    ).splitlines()
    for path_text in file_paths:
        task = task_from_path(path_text)
        if task not in eligible:
            continue
        path = ROOT / path_text
        try:
            add(task, path.read_bytes(), path_text, "archive_file")
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": path_text, "error": f"{type(exc).__name__}: {exc}"})

    zip_paths = subprocess.check_output(
        ["rg", "--files", "-g", "*.zip"], cwd=ROOT, text=True
    ).splitlines()
    wanted = {f"task{task:03d}.onnx": task for task in eligible}
    for path_text in zip_paths:
        path = ROOT / path_text
        try:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    task = wanted.get(Path(member).name)
                    if task is None:
                        continue
                    add(task, archive.read(member), f"{path_text}!{member}", "archive_zip")
        except Exception as exc:  # noqa: BLE001
            errors.append({"source": path_text, "error": f"{type(exc).__name__}: {exc}"})

    exact = load_module(
        "restart409_exact_transforms",
        ROOT / "scripts/golf/agent_exact_noop_scan_285/scan.py",
    )
    for task, authority_data in authority_members.items():
        model = onnx.load_model_from_string(authority_data)
        for profile in exact.PROFILES:
            try:
                candidate, actions = exact.transform(model, profile)
                if not actions:
                    continue
                add(
                    task,
                    candidate.SerializeToString(),
                    f"authority_exact:{profile}",
                    "exact_graph_reduction",
                    {"profile": profile, "actions": actions},
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "source": f"authority_exact:task{task:03d}:{profile}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        for pass_name in SAFE_OPTIMIZER_PASSES:
            try:
                candidate = onnxoptimizer.optimize(copy.deepcopy(model), [pass_name])
                data = candidate.SerializeToString()
                if sha256(data) == sha256(authority_data):
                    continue
                add(
                    task,
                    data,
                    f"authority_optimizer:{pass_name}",
                    "exact_optimizer_reduction",
                    {"passes": [pass_name]},
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "source": f"authority_optimizer:task{task:03d}:{pass_name}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        try:
            candidate = onnxoptimizer.optimize(
                copy.deepcopy(model), list(SAFE_OPTIMIZER_PASSES)
            )
            data = candidate.SerializeToString()
            if sha256(data) != sha256(authority_data):
                add(
                    task,
                    data,
                    "authority_optimizer:combined",
                    "exact_optimizer_reduction",
                    {"passes": list(SAFE_OPTIMIZER_PASSES)},
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "source": f"authority_optimizer:task{task:03d}:combined",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    candidates: dict[int, list[dict[str, Any]]] = {task: [] for task in eligible}
    authority_shas = {task: sha256(data) for task, data in authority_members.items()}
    rejected_bounds = 0
    authority_duplicates = 0
    for (task, digest), row in by_key.items():
        if digest == authority_shas[task]:
            authority_duplicates += 1
            continue
        if (
            int(row["params"]) >= int(eligible[task])
            or int(row["declared_lower_bound"]) >= int(eligible[task])
        ):
            rejected_bounds += 1
            continue
        row["sources"].sort()
        row["source_classes"].sort()
        candidates[task].append(row)
    for rows in candidates.values():
        rows.sort(key=lambda row: (
            int(row["declared_lower_bound"]),
            int(row["params"]),
            str(row["sha256"]),
        ))
    meta = {
        "onnx_file_paths": len(file_paths),
        "zip_paths": len(zip_paths),
        "raw_source_counts": dict(sorted(raw_counts.items())),
        "unique_task_sha": len(by_key),
        "authority_duplicates": authority_duplicates,
        "rejected_by_safe_lower_bound": rejected_bounds,
        "screen_candidates": sum(len(rows) for rows in candidates.values()),
        "screen_candidates_by_task": {
            str(task): len(rows) for task, rows in sorted(candidates.items())
        },
        "source_errors": errors,
        "elapsed_seconds": time.monotonic() - started,
    }
    return candidates, meta


def runtime_row_pass(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("accuracy", 0.0)) >= THRESHOLD
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def compact_runtime(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive",
        "maximum_nonpositive", "sign_mismatch_cases_vs_disable_threads1",
        "sign_mismatch_cells_vs_disable_threads1", "sign_sha256", "raw_sha256",
        "first_wrong", "first_error", "first_shape_mismatch", "optimization", "threads",
    )
    result = {key: row.get(key) for key in keys if key in row}
    result["policy_threshold"] = THRESHOLD
    result["policy90"] = runtime_row_pass(row)
    if row.get("session_error"):
        result["session_error"] = row["session_error"]
    return result


def mandatory_structure(support, task: int, data: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(data)
    generic = support.structural_audit(task, model, data)
    try:
        trace = support.runtime_shape_trace(task, copy.deepcopy(model))
        trace["shape_cloak_findings"] = len(trace.get("declared_actual_mismatches", []))
        trace["truthful"] = bool(
            not trace.get("error") and trace["shape_cloak_findings"] == 0
        )
    except Exception as exc:  # noqa: BLE001
        trace = {
            "truthful": False,
            "shape_cloak_findings": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
    reasons: list[str] = []
    if not generic.get("full_check"):
        reasons.append("full_checker")
    if not generic.get("strict_shape_data_prop"):
        reasons.append("strict_shape")
    if not generic.get("canonical_io"):
        reasons.append("noncanonical_io")
    if generic.get("missing_node_outputs") or generic.get("nonstatic_node_outputs"):
        reasons.append("untyped_or_nonstatic_node_output")
    if generic.get("banned_ops"):
        reasons.append("banned_ops")
    if generic.get("lookup_ops"):
        reasons.append("suspicious_lookup_op")
    if (
        generic.get("nonstandard_domains")
        or generic.get("nested_graphs")
        or generic.get("functions")
        or generic.get("sparse_initializers")
        or generic.get("external_initializers")
    ):
        reasons.append("nonstandard_nested_or_external")
    if generic.get("nonfinite_initializers"):
        reasons.append("nonfinite_initializer")
    if generic.get("conv_bias_ub_findings"):
        reasons.append("conv_bias_ub")
    if not trace.get("truthful"):
        reasons.append("runtime_shape_cloak")
    return {
        "pass": not reasons,
        "reasons": sorted(set(reasons)),
        "full_check": generic.get("full_check"),
        "strict_shape_data_prop": generic.get("strict_shape_data_prop"),
        "canonical_io": generic.get("canonical_io"),
        "missing_node_outputs": generic.get("missing_node_outputs"),
        "nonstatic_node_outputs": generic.get("nonstatic_node_outputs"),
        "banned_ops": generic.get("banned_ops"),
        "lookup_ops": generic.get("lookup_ops"),
        "nonstandard_domains": generic.get("nonstandard_domains"),
        "nested_graphs": generic.get("nested_graphs"),
        "functions": generic.get("functions"),
        "sparse_initializers": generic.get("sparse_initializers"),
        "external_initializers": generic.get("external_initializers"),
        "nonfinite_initializers": generic.get("nonfinite_initializers"),
        "conv_bias_ub_findings": generic.get("conv_bias_ub_findings"),
        "runtime_shape_trace": trace,
        "max_einsum_inputs": generic.get("max_einsum_inputs"),
        "giant_einsum_advisory": generic.get("giant_einsum"),
        "file_bytes": generic.get("file_bytes"),
    }


def official_profile(support, task: int, data: bytes, label: str) -> dict[str, Any] | None:
    try:
        return support.official_profile(
            task, onnx.load_model_from_string(data), label
        )
    except Exception:  # noqa: BLE001
        return None


def known_base_screen(support, data: bytes, cases: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        runtime = support.make_session(data, True, 1)
        row, _ = support.evaluate_config(runtime, cases, None)
        return compact_runtime(row)
    except Exception as exc:  # noqa: BLE001
        return {
            "total": len(cases),
            "right": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "errors": len(cases),
            "session_error": f"{type(exc).__name__}: {exc}",
            "policy_threshold": THRESHOLD,
            "policy90": False,
        }


def summarize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "data"}


def evaluate_four(support, data: bytes, cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = support.evaluate_four(data, cases)
    return {name: compact_runtime(row) for name, row in rows.items()}


def four_pass(rows: dict[str, dict[str, Any]]) -> bool:
    return bool(len(rows) == 4 and all(row["policy90"] for row in rows.values()))


def worker_main(args: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    worker_index = int(args["worker_index"])
    tasks = [int(task) for task in args["tasks"]]
    costs = {int(task): int(cost) for task, cost in args["costs"].items()}
    candidates: dict[int, list[dict[str, Any]]] = {
        int(task): rows for task, rows in args["candidates"].items()
    }
    authority_members: dict[int, bytes] = {
        int(task): data for task, data in args["authority_members"].items()
    }
    support = load_support()
    results: list[dict[str, Any]] = []
    worker_finalists: list[dict[str, Any]] = []
    candidate_dir = HERE / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_started = time.monotonic()
        authority_data = authority_members[task]
        authority_profile = official_profile(
            support, task, authority_data, f"restart409_authority_{task:03d}"
        )
        if authority_profile is None:
            raise RuntimeError(f"authority task{task:03d} is unscorable")
        known_cases, known_counts = support.known_cases(task)
        screen_rows: list[dict[str, Any]] = []
        eligible_rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates[task], start=1):
            data = candidate["data"]
            row = summarize_candidate(candidate)
            known = known_base_screen(support, data, known_cases)
            row["known_disable_threads1"] = known
            row["status"] = "REJECT_KNOWN_POLICY90"
            if known["policy90"]:
                profile = official_profile(
                    support, task, data, f"restart409_w{worker_index}_{task:03d}_{index}"
                )
                row["profile"] = profile
                if profile is None:
                    row["status"] = "REJECT_UNSCORABLE"
                elif int(profile["cost"]) >= int(authority_profile["cost"]):
                    row["status"] = "REJECT_NOT_STRICT_LOWER_ACTUAL"
                else:
                    structure = mandatory_structure(support, task, data)
                    row["structure"] = structure
                    if not structure["pass"]:
                        row["status"] = "REJECT_STRUCTURE"
                    else:
                        row["status"] = "ELIGIBLE_FRESH"
                        row["data"] = data
                        eligible_rows.append(row)
            screen_rows.append({key: value for key, value in row.items() if key != "data"})
        eligible_rows.sort(
            key=lambda row: (
                int(row["profile"]["cost"]),
                -float(row["known_disable_threads1"]["accuracy"]),
                0 if "existing_white_lineage" in row["source_classes"] else 1,
                str(row["sha256"]),
            )
        )
        fresh_cases = []
        fresh_generation = []
        for seed in (409_200_000 + task, 409_300_000 + task):
            cases, generation = support.fresh_cases(
                task,
                seed,
                json.loads((ROOT / "docs/golf/task_hash_map.json").read_text()),
            )
            fresh_cases.append(cases)
            fresh_generation.append(generation)
        fresh_audits: list[dict[str, Any]] = []
        finalist = None
        for rank, row in enumerate(
            eligible_rows[:MAX_FRESH_CANDIDATES_PER_TASK], start=1
        ):
            data = row["data"]
            known_four = evaluate_four(support, data, known_cases)
            fresh_runs = []
            for cases, generation in zip(fresh_cases, fresh_generation):
                runtime = evaluate_four(support, data, cases)
                fresh_runs.append(
                    {
                        "generation": generation,
                        "runtime": runtime,
                        "pass": four_pass(runtime),
                    }
                )
            audit = {
                **{key: value for key, value in row.items() if key != "data"},
                "rank": rank,
                "known_four": known_four,
                "known_four_pass": four_pass(known_four),
                "fresh": fresh_runs,
            }
            audit["policy90_pass"] = bool(
                audit["known_four_pass"] and all(run["pass"] for run in fresh_runs)
            )
            fresh_audits.append(audit)
            print(
                json.dumps(
                    {
                        "worker": worker_index,
                        "pid": os.getpid(),
                        "task": task,
                        "rank": rank,
                        "cost": row["profile"]["cost"],
                        "known": known_four["disable_threads1"]["accuracy"],
                        "fresh": [
                            run["runtime"]["disable_threads1"]["accuracy"]
                            for run in fresh_runs
                        ],
                        "pass": audit["policy90_pass"],
                    }
                ),
                flush=True,
            )
            if audit["policy90_pass"]:
                all_rows = [*known_four.values()]
                for run in fresh_runs:
                    all_rows.extend(run["runtime"].values())
                exact_runtime = all(float(item["accuracy"]) == 1.0 for item in all_rows)
                classification = (
                    "KNOWN_EXACT_FRESH_EXACT"
                    if row["profile"].get("correct") is True and exact_runtime
                    else "POLICY90_NONEXACT"
                )
                output = candidate_dir / (
                    f"task{task:03d}_cost{int(row['profile']['cost'])}_"
                    f"{row['sha256'][:12]}_{classification}.onnx"
                )
                output.write_bytes(data)
                audit["classification"] = classification
                audit["saved_path"] = rel(output)
                audit["saved_sha256"] = sha256(output.read_bytes())
                audit["projected_gain"] = math.log(
                    int(authority_profile["cost"]) / int(row["profile"]["cost"])
                )
                finalist = audit
                worker_finalists.append(audit)
                break
        result = {
            "task": task,
            "ledger_cost": costs[task],
            "authority": {
                "member": f"task{task:03d}.onnx",
                "sha256": sha256(authority_data),
                "bytes": len(authority_data),
                "profile": authority_profile,
                "ledger_cost_matches_actual": int(authority_profile["cost"]) == costs[task],
            },
            "known_counts": known_counts,
            "screen_candidate_count": len(candidates[task]),
            "eligible_fresh_count": len(eligible_rows),
            "screen": screen_rows,
            "fresh_audits": fresh_audits,
            "finalist": finalist,
            "elapsed_seconds": time.monotonic() - task_started,
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "worker": worker_index,
                    "pid": os.getpid(),
                    "task_done": task,
                    "screened": len(candidates[task]),
                    "eligible": len(eligible_rows),
                    "winner": None if finalist is None else {
                        "cost": finalist["profile"]["cost"],
                        "class": finalist["classification"],
                    },
                }
            ),
            flush=True,
        )
    return {
        "worker_index": worker_index,
        "pid": os.getpid(),
        "tasks": tasks,
        "candidate_load": sum(len(candidates[task]) for task in tasks),
        "results": results,
        "finalists": worker_finalists,
        "elapsed_seconds": time.monotonic() - started,
    }


def balanced_groups(candidates: dict[int, list[dict[str, Any]]]) -> list[list[int]]:
    groups: list[list[int]] = [[], [], []]
    loads = [0, 0, 0]
    for task, rows in sorted(candidates.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(3), key=lambda index: (loads[index], index))
        groups[target].append(task)
        loads[target] += len(rows)
    for group in groups:
        group.sort()
    return groups


def main() -> None:
    started = time.monotonic()
    HERE.mkdir(parents=True, exist_ok=True)
    authority_bytes = AUTHORITY.read_bytes()
    if sha256(authority_bytes) != AUTHORITY_SHA256:
        raise RuntimeError("immutable authority changed")
    scope, eligible = score_census()
    if not eligible:
        raise RuntimeError("empty eligible range")
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_members = {
            task: archive.read(f"task{task:03d}.onnx") for task in eligible
        }
    candidates, discovery = discover_candidates(eligible, authority_members)
    groups = balanced_groups(candidates)
    loads = [sum(len(candidates[task]) for task in group) for group in groups]
    inventory = {
        "authority": {
            "zip": rel(AUTHORITY),
            "sha256": AUTHORITY_SHA256,
            "lb": AUTHORITY_LB,
        },
        "range": [RANGE_MIN, RANGE_MAX],
        "scope_count": len(scope),
        "scope": scope,
        "eligible_count": len(eligible),
        "eligible_costs": {str(task): cost for task, cost in sorted(eligible.items())},
        "excluded_count": sum(bool(row["excluded"]) for row in scope),
        "latest_lb_black": sorted(LATEST_LB_BLACK),
        "private_catalog": {
            "path": "docs/golf/private_zero_tasks.md",
            "sha256": sha256((ROOT / "docs/golf/private_zero_tasks.md").read_bytes()),
        },
        "discovery": discovery,
        "worker_groups": [
            {"worker_index": index, "tasks": group, "candidate_load": loads[index]}
            for index, group in enumerate(groups)
        ],
    }
    (HERE / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "scope": len(scope),
                "eligible": eligible,
                "screen_candidates": discovery["screen_candidates"],
                "groups": inventory["worker_groups"],
            },
            indent=2,
        ),
        flush=True,
    )

    worker_args = []
    for index, group in enumerate(groups):
        worker_args.append(
            {
                "worker_index": index,
                "tasks": group,
                "costs": {task: eligible[task] for task in group},
                "candidates": {task: candidates[task] for task in group},
                "authority_members": {task: authority_members[task] for task in group},
            }
        )
    context = mp.get_context("spawn")
    with context.Pool(processes=3) as pool:
        workers = pool.map(worker_main, worker_args)
    workers.sort(key=lambda row: int(row["worker_index"]))
    for worker in workers:
        (HERE / f"worker_{worker['worker_index']}_evidence.json").write_text(
            json.dumps(worker, indent=2) + "\n", encoding="utf-8"
        )
    task_results = sorted(
        [result for worker in workers for result in worker["results"]],
        key=lambda row: int(row["task"]),
    )
    finalists = [row["finalist"] for row in task_results if row["finalist"] is not None]
    total_gain = sum(float(row["projected_gain"]) for row in finalists)
    payload = {
        "lane": rel(HERE),
        "authority": inventory["authority"],
        "policy_threshold": THRESHOLD,
        "fresh_design": "2 independent seeds x 2000 cases x 4 ORT configs",
        "workers_requested": 3,
        "worker_pids": [int(worker["pid"]) for worker in workers],
        "worker_groups": inventory["worker_groups"],
        "tasks": task_results,
        "summary": {
            "scope_count": len(scope),
            "excluded_count": inventory["excluded_count"],
            "eligible_count": len(eligible),
            "screen_candidates": discovery["screen_candidates"],
            "fresh_audited": sum(len(row["fresh_audits"]) for row in task_results),
            "admitted": len(finalists),
            "exact_admitted": sum(
                row["classification"] == "KNOWN_EXACT_FRESH_EXACT" for row in finalists
            ),
            "policy90_nonexact_admitted": sum(
                row["classification"] == "POLICY90_NONEXACT" for row in finalists
            ),
            "projected_gain": total_gain,
            "projected_lb_if_all_admitted_hold": AUTHORITY_LB + total_gain,
        },
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "root submission/all_scores/others unchanged; lane only",
    }
    (HERE / "evidence.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "authority": inventory["authority"],
        "admission_policy": "POLICY90; runtime-clean and structurally mandatory; not an LB guarantee",
        "candidates": [
            {
                "task": row["task"],
                "path": row["saved_path"],
                "sha256": row["saved_sha256"],
                "authority_cost": next(
                    result["authority"]["profile"]["cost"]
                    for result in task_results if result["task"] == row["task"]
                ),
                "candidate_cost": row["profile"]["cost"],
                "projected_gain": row["projected_gain"],
                "classification": row["classification"],
                "sources": row["sources"],
            }
            for row in finalists
        ],
        "projected_gain": total_gain,
        "projected_lb_if_all_admitted_hold": AUTHORITY_LB + total_gain,
        "root_submission_all_scores_others_modified": False,
    }
    (HERE / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# cost 1001..2500 / 3-worker POLICY90 scan",
        "",
        f"Authority: `submission_base_8012.15.zip` (`{AUTHORITY_SHA256}`)",
        "",
        f"Scope: {len(scope)} non-score25 tasks; excluded: {inventory['excluded_count']}; scanned: {len(eligible)}.",
        "",
        f"Workers: PIDs {payload['worker_pids']}; candidate loads {loads}.",
        "",
        "| task | authority | candidate | gain | class | known/fresh minimum |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for result in task_results:
        row = result["finalist"]
        if row is None:
            report.append(
                f"| {result['task']:03d} | {result['authority']['profile']['cost']} | — | — | NO_ADMISSION | — |"
            )
            continue
        runtime_rows = [*row["known_four"].values()]
        for run in row["fresh"]:
            runtime_rows.extend(run["runtime"].values())
        minimum = min(float(item["accuracy"]) for item in runtime_rows)
        report.append(
            f"| {result['task']:03d} | {result['authority']['profile']['cost']} | "
            f"{row['profile']['cost']} | +{row['projected_gain']:.6f} | "
            f"{row['classification']} | {minimum:.2%} |"
        )
    report.extend(
        [
            "",
            f"Conditional total gain: **+{total_gain:.6f}**",
            f"Conditional projected LB: **{AUTHORITY_LB + total_gain:.6f}**",
            "",
            "The sole strict-lower structural lead was task023 at cost 1319 (authority",
            "1321). It was known-exact but fresh accuracy was 85.95% / 84.00%, below",
            "POLICY90, and was rejected.",
            "",
            "POLICY90 candidates may be non-exact and are not an LB guarantee. All admitted",
            "models pass checker, strict/static shapes, runtime-shape tracing, Conv-bias UB,",
            "banned-op, nonfinite, output-shape, small-positive, and four-config gates.",
            "",
            "The root submission, all_scores.csv, and others/ were not modified.",
        ]
    )
    (HERE / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

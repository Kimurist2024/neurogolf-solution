#!/usr/bin/env python3
"""Clean policy-threshold rescreen of two authoritative historical inventories.

This lane is non-promoting.  It SHA-deduplicates the accepted-history and old
loose-sweep inventories, rebases every candidate against the exact 8004.50
member, and applies strict structural, actual-cost, known, runtime-shape, and
fresh-generator gates.  It never writes a submission archive.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
COSTS_JSON = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
HISTORY_JSON = ROOT / "scripts/golf/loop_8004_42_plus20/agent_history_miner/history_inventory.json"
LOOSE_JSON = ROOT / "scripts/golf/loop_7999_13/lane_archive_loose_sweep/inventory.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
MAX_BYTES = 1_440_000
BANNED = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
PRIVATE_TERMS = ("private0", "private_zero", "private-zero", "quarantine", "nonadopt")

# docs/golf/private_zero_tasks.md: all confirmed/high-risk/unsound-incumbent
# tasks.  A cheap model from these tasks is deliberately ineligible in this
# clean lane even if its copied path no longer carries the quarantine name.
PRIVATE_ZERO_CATALOG = {
    9, 15, 35, 44, 48, 66, 72, 77, 86, 90, 96, 101, 102, 133, 134,
    138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187, 192, 196,
    202, 205, 209, 216, 219, 222, 233, 246, 255, 277, 285, 286, 302,
    325, 346, 361, 365, 366, 372, 377, 379, 393, 396,
    # unresolved 7614 pair and documented monitoring additions
    70, 112, 198, 208, 319, 333, 343, 344, 391,
}
EXPLICIT_EXCLUDED_SHA = {
    # Already adopted by the root task009 policy95 wave.  Do not rediscover it.
    "b265f7f83d8fbf66c9388b9edfe0111d2b77a4b610377a3994a9c483fb445d28",
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from harvest import actual_screen, exact_conv_bias_gate  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def score_gain(before: int, after: int) -> float:
    return math.log(before / after)


def resolve_source(source: str, task: int) -> bytes | None:
    """Resolve a loose ONNX, ZIP member, or ZIP containing taskNNN.onnx."""
    if "::" in source:
        archive_text, member = source.split("::", 1)
        archive = Path(archive_text)
        if not archive.is_absolute():
            archive = ROOT / archive
        try:
            with zipfile.ZipFile(archive) as handle:
                return handle.read(member)
        except Exception:
            return None
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    try:
        if path.suffix.lower() == ".onnx":
            return path.read_bytes()
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as handle:
                return handle.read(f"task{task:03d}.onnx")
    except Exception:
        return None
    return None


def load_inventory() -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        baseline = {
            task: archive.read(f"task{task:03d}.onnx") for task in range(1, 401)
        }
    base_sha = {task: sha256(data) for task, data in baseline.items()}
    records: dict[tuple[int, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    resolution_errors: list[dict[str, Any]] = []

    def add(task: int, expected_sha: str | None, sources: list[str], origin: str) -> None:
        counts[f"{origin}_rows"] += 1
        data = None
        resolved = None
        for source in sources:
            data = resolve_source(source, task)
            if data is not None:
                resolved = source
                break
        if data is None:
            counts["unresolved"] += 1
            resolution_errors.append({"task": task, "sha256": expected_sha, "sources": sources})
            return
        digest = sha256(data)
        if expected_sha and digest != expected_sha:
            counts["sha_mismatch"] += 1
            resolution_errors.append(
                {"task": task, "expected_sha256": expected_sha, "actual_sha256": digest, "source": resolved}
            )
            return
        if digest == base_sha[task]:
            counts["current_duplicates"] += 1
            return
        key = (task, digest)
        slot = records.setdefault(
            key,
            {
                "task": task,
                "sha256": digest,
                "data": data,
                "sources": [],
                "origins": [],
                "resolved_source": resolved,
                "current_sha256": base_sha[task],
            },
        )
        slot["sources"].extend(sources)
        slot["origins"].append(origin)

    loose = json.loads(LOOSE_JSON.read_text())
    for task_text, rows in loose["retained"].items():
        task = int(task_text)
        for row in rows:
            sources = [str(row["path"]), *[str(item) for item in row.get("sources", [])]]
            add(task, row.get("sha256"), sources, "loose_sweep")

    history = json.loads(HISTORY_JSON.read_text())
    for row in history["all_records"]:
        task = int(row["task"])
        path = row.get("candidate_path")
        sources = ([str(path)] if path else []) + [str(item) for item in row.get("lineage", [])]
        add(task, row.get("sha256"), sources, "accepted_history")

    counts["unique_noncurrent_task_sha"] = len(records)
    report = {
        "input_inventories": [rel(HISTORY_JSON), rel(LOOSE_JSON)],
        "baseline": rel(BASE_ZIP),
        "baseline_sha256": sha256(BASE_ZIP.read_bytes()),
        "counts": dict(counts),
        "tasks_with_unique_candidates": len({task for task, _ in records}),
        "unique_by_task": dict(Counter(str(task) for task, _ in records)),
        "resolution_errors": resolution_errors,
    }
    return records, report


def tensor_dims(value: onnx.ValueInfoProto) -> list[int | None]:
    if not value.type.HasField("tensor_type"):
        return []
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def static_audit(data: bytes, sources: list[str], task: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pass": False,
        "reasons": [],
        "static_cost": None,
        "memory": None,
        "params": None,
    }
    if len(data) > MAX_BYTES:
        out["reasons"].append("file_too_large")
        return out
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model, full_check=True)
        inferred = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    except Exception as exc:  # noqa: BLE001
        out["reasons"].append(f"checker_or_strict_data_prop:{type(exc).__name__}")
        out["error"] = str(exc)
        return out

    if task in PRIVATE_ZERO_CATALOG:
        out["reasons"].append("private_zero_catalog_task")
    if any(term in source.lower() for source in sources for term in PRIVATE_TERMS):
        out["reasons"].append("private_zero_lineage")
    if model.functions:
        out["reasons"].append("local_functions")
    if model.graph.sparse_initializer:
        out["reasons"].append("sparse_initializer")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        out["reasons"].append("noncanonical_io_count")
    elif model.graph.input[0].name != "input" or model.graph.output[0].name != "output":
        out["reasons"].append("noncanonical_io_names")
    if any(item.domain not in {"", "ai.onnx"} for item in model.opset_import):
        out["reasons"].append("custom_opset")
    if any(item.external_data or item.data_location == onnx.TensorProto.EXTERNAL for item in model.graph.initializer):
        out["reasons"].append("external_data")

    ops = Counter(node.op_type for node in model.graph.node)
    max_einsum = 0
    nested = 0
    custom_nodes = 0
    banned: list[str] = []
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in BANNED or "SEQUENCE" in upper:
            banned.append(node.op_type)
        if node.domain not in {"", "ai.onnx"}:
            custom_nodes += 1
        if node.op_type == "Einsum":
            max_einsum = max(max_einsum, len(node.input))
        nested += sum(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for attr in node.attribute
        )
    if banned:
        out["reasons"].append("banned_op")
    if custom_nodes:
        out["reasons"].append("custom_node_domain")
    if nested:
        out["reasons"].append("nested_graph")
    if max_einsum >= 15:
        out["reasons"].append("giant_einsum")

    giant_initializers = []
    for item in model.graph.initializer:
        try:
            if int(numpy_helper.to_array(item).size) >= 10_000:
                giant_initializers.append(item.name)
        except Exception:  # noqa: BLE001
            giant_initializers.append(item.name)
    if ops.get("TfIdfVectorizer") or ops.get("Hardmax") or giant_initializers:
        out["reasons"].append("lookup_or_giant_initializer")

    bias_ok, bias_reason, bias_findings = exact_conv_bias_gate(model)
    if not bias_ok:
        out["reasons"].append("conv_family_bias_ub")

    values = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    init_names = {item.name for item in inferred.graph.initializer}
    node_outputs = {name for node in inferred.graph.node for name in node.output if name}
    memory = 0
    nonstatic = []
    unknown = []
    for name in node_outputs:
        if name in {"input", "output"} or name in init_names:
            continue
        value = values.get(name)
        dims = [] if value is None else tensor_dims(value)
        if value is None or not dims or any(dim is None or dim <= 0 for dim in dims):
            nonstatic.append(name)
            continue
        try:
            itemsize = np.dtype(helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)).itemsize
        except Exception:  # noqa: BLE001
            unknown.append(name)
            continue
        memory += int(np.prod(dims, dtype=np.int64)) * itemsize
    if nonstatic:
        out["reasons"].append("nonstatic_shape")
    if unknown:
        out["reasons"].append("unknown_dtype")
    params = int(sum(np.prod(item.dims, dtype=np.int64) for item in inferred.graph.initializer))
    out.update(
        {
            "ops": dict(ops),
            "max_einsum_inputs": max_einsum,
            "giant_initializers": giant_initializers,
            "nested_graph_count": nested,
            "custom_node_count": custom_nodes,
            "conv_bias_reason": bias_reason,
            "conv_bias_findings": bias_findings,
            "nonstatic": nonstatic,
            "unknown_dtype": unknown,
            "memory": memory,
            "params": params,
            "static_cost": memory + params,
        }
    )
    out["reasons"] = sorted(set(out["reasons"]))
    out["pass"] = not out["reasons"]
    return out


def make_session(data: bytes, disable_all: bool) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 4
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def known_dual(task: int, data: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        stats = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
        try:
            session = make_session(data, disabled)
        except Exception as exc:  # noqa: BLE001
            stats["session_error"] = f"{type(exc).__name__}: {exc}"
            result[mode] = stats
            continue
        for subset in ("train", "test", "arc-gen"):
            for example in scoring.load_examples(task)[subset]:
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
                        stats["first_failure"] = {"subset": subset, "error": f"{type(exc).__name__}: {exc}"}
        result[mode] = stats
    return result


def fresh_dual(task: int, candidates: list[dict[str, Any]], count: int, seed: int) -> dict[str, Any]:
    mapping = json.loads(TASK_MAP.read_text())
    generator = importlib.import_module(f"task_{mapping[f'{task:03d}']}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    sessions: dict[tuple[str, str], ort.InferenceSession] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for row in candidates:
        digest = row["sha256"]
        stats[digest] = {}
        for disabled, mode in ((True, "disable_all"), (False, "default")):
            stats[digest][mode] = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
            try:
                sessions[(digest, mode)] = make_session(row["data"], disabled)
            except Exception as exc:  # noqa: BLE001
                stats[digest][mode]["session_error"] = f"{type(exc).__name__}: {exc}"
    valid = attempts = generation_errors = conversion_skips = 0
    while valid < count:
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
            for mode in ("disable_all", "default"):
                item = stats[digest][mode]
                session = sessions.get((digest, mode))
                if session is None:
                    item["errors"] += 1
                    continue
                try:
                    raw = session.run(
                        [session.get_outputs()[0].name],
                        {session.get_inputs()[0].name: benchmark["input"]},
                    )[0]
                    if np.array_equal(raw > 0, want):
                        item["right"] += 1
                    else:
                        item["wrong"] += 1
                        if item["first_failure"] is None:
                            item["first_failure"] = {
                                "valid_case": valid,
                                "different_cells": int(np.count_nonzero((raw > 0) != want)),
                            }
                except Exception as exc:  # noqa: BLE001
                    item["errors"] += 1
                    if item["first_failure"] is None:
                        item["first_failure"] = {"valid_case": valid, "error": f"{type(exc).__name__}: {exc}"}
        if valid % 100 == 0:
            print(f"FRESH task{task:03d} {valid}/{count} candidates={len(candidates)}", flush=True)
    return {
        "task": task,
        "seed": seed,
        "requested": count,
        "valid": valid,
        "attempts": attempts,
        "generation_errors": generation_errors,
        "conversion_skips": conversion_skips,
        "candidates": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=int, default=500)
    parser.add_argument("--confirm", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "candidates").mkdir(exist_ok=True)
    (HERE / "evidence").mkdir(exist_ok=True)
    ort.set_default_logger_severity(4)
    started = time.time()
    current_costs = {int(task): int(cost) for task, cost in json.loads(COSTS_JSON.read_text())["costs"].items()}
    records, inventory = load_inventory()
    (HERE / "inventory_union.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        f"INVENTORY unique={inventory['counts']['unique_noncurrent_task_sha']} "
        f"tasks={inventory['tasks_with_unique_candidates']}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    for (task, digest), item in sorted(records.items()):
        sources = sorted(set(item["sources"]))
        row: dict[str, Any] = {
            "task": task,
            "sha256": digest,
            "sources": sources,
            "origins": sorted(set(item["origins"])),
            "resolved_source": item["resolved_source"],
            "current_sha256": item["current_sha256"],
            "current_actual_cost": current_costs[task],
            "stage": "static_pending",
        }
        rows.append(row)
        if digest in EXPLICIT_EXCLUDED_SHA:
            row.update(stage="already_adopted_excluded", reasons=["task009_b265_already_adopted"])
            continue
        audit = static_audit(item["data"], sources, task)
        row["static_audit"] = audit
        if not audit["pass"]:
            row.update(stage="static_policy_reject", reasons=audit["reasons"])
            continue
        if audit["static_cost"] is None or int(audit["static_cost"]) >= current_costs[task]:
            row.update(stage="static_not_cheaper", reasons=["static_cost_not_strictly_cheaper"])
            continue
        row.update(stage="actual_pending", reasons=[])
        candidates[digest] = {**item, "row": row}

    def checkpoint(complete: bool = False) -> None:
        (HERE / "screen_results.json").write_text(
            json.dumps(
                {
                    "complete": complete,
                    "baseline": rel(BASE_ZIP),
                    "baseline_sha256": sha256(BASE_ZIP.read_bytes()),
                    "inventory": inventory,
                    "fresh_count": args.fresh,
                    "confirm_count_per_seed": args.confirm,
                    "fresh_accuracy_threshold": args.threshold,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )

    checkpoint()
    actual_pending = [item for item in candidates.values() if item["row"]["stage"] == "actual_pending"]
    print(f"STATIC actual_pending={len(actual_pending)}", flush=True)
    for index, item in enumerate(actual_pending, 1):
        row = item["row"]
        cost = actual_screen(item["data"], int(row["task"]))
        row["actual_cost"] = cost
        if cost is None or cost >= int(row["current_actual_cost"]):
            row.update(stage="actual_reject", reasons=["actual_cost_not_strictly_cheaper"])
        else:
            row.update(stage="known_pending", reasons=[], gain=score_gain(int(row["current_actual_cost"]), cost))
        if index % 20 == 0:
            print(f"ACTUAL {index}/{len(actual_pending)}", flush=True)
            checkpoint()

    known_pending = [item for item in candidates.values() if item["row"]["stage"] == "known_pending"]
    print(f"ACTUAL known_pending={len(known_pending)}", flush=True)
    pre_fresh: list[dict[str, Any]] = []
    for index, item in enumerate(known_pending, 1):
        row = item["row"]
        task = int(row["task"])
        print(f"KNOWN {index}/{len(known_pending)} task{task:03d} {row['sha256'][:12]}", flush=True)
        dual = known_dual(task, item["data"])
        row["known_dual"] = dual
        if any(
            mode.get("wrong") or mode.get("errors") or mode.get("session_error") or not mode.get("right")
            for mode in dual.values()
        ):
            row.update(stage="known_reject", reasons=["known_dual_not_100_percent"])
            continue
        try:
            trace = runtime_shape_trace(task, onnx.load_model_from_string(item["data"]))
            row["runtime_shape_trace"] = trace
        except Exception as exc:  # noqa: BLE001
            row.update(stage="shape_reject", reasons=["runtime_shape_trace_error"], shape_error=f"{type(exc).__name__}: {exc}")
            continue
        if trace["declared_actual_mismatches"]:
            row.update(stage="shape_reject", reasons=["shape_cloak"])
            continue
        row.update(stage="fresh_pending", reasons=[])
        pre_fresh.append(item)
        out = HERE / "candidates" / f"task{task:03d}_{row['sha256'][:12]}_cost{row['actual_cost']}.onnx"
        out.write_bytes(item["data"])
        row["isolated_candidate"] = rel(out)
        checkpoint()

    print(f"PRE_FRESH candidates={len(pre_fresh)}", flush=True)
    for task in sorted({int(item["row"]["task"]) for item in pre_fresh}):
        task_items = [item for item in pre_fresh if int(item["row"]["task"]) == task]
        report = fresh_dual(task, task_items, args.fresh, 81_000_000 + task)
        (HERE / "evidence" / f"task{task:03d}_fresh_dual_{args.fresh}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        for item in task_items:
            row = item["row"]
            row["fresh500_dual"] = report["candidates"][row["sha256"]]
            passed = all(
                mode["right"] / args.fresh >= args.threshold
                and mode["right"] + mode["wrong"] == args.fresh
                and mode["errors"] == 0
                and not mode.get("session_error")
                for mode in row["fresh500_dual"].values()
            )
            row.update(
                stage="fresh500_pass" if passed else "fresh500_reject",
                reasons=[] if passed else ["fresh_below_threshold_or_runtime_error"],
            )
        checkpoint()

    stage500 = [item for item in pre_fresh if item["row"]["stage"] == "fresh500_pass"]
    print(f"FRESH500 survivors={len(stage500)}", flush=True)
    # Two independent 5000-case confirmations, exactly as requested.  Both
    # seeds reuse the same candidates per task for generation efficiency.
    for seed_index, seed_base in enumerate((82_000_000, 83_000_000), 1):
        for task in sorted({int(item["row"]["task"]) for item in stage500}):
            task_items = [item for item in stage500 if int(item["row"]["task"]) == task]
            report = fresh_dual(task, task_items, args.confirm, seed_base + task)
            (HERE / "evidence" / f"task{task:03d}_fresh_dual_{args.confirm}_seed{seed_index}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            for item in task_items:
                row = item["row"]
                row.setdefault("fresh5000_seeds", {})[f"seed{seed_index}"] = report["candidates"][row["sha256"]]
            checkpoint()

    admitted: list[dict[str, Any]] = []
    for item in stage500:
        row = item["row"]
        seeds = row.get("fresh5000_seeds", {})
        passed = len(seeds) == 2 and all(
            all(
                mode["right"] / args.confirm >= args.threshold
                and mode["right"] + mode["wrong"] == args.confirm
                and mode["errors"] == 0
                and not mode.get("session_error")
                for mode in seed.values()
            )
            for seed in seeds.values()
        )
        row.update(
            stage="admit" if passed else "confirm_reject",
            reasons=[] if passed else ["independent_5000_below_threshold_or_runtime_error"],
        )
        if passed:
            admitted.append(row)

    admitted.sort(key=lambda row: (-float(row["gain"]), int(row["task"]), int(row["actual_cost"])))
    # One admissible candidate per task: cheapest actual-cost model wins.
    best: dict[int, dict[str, Any]] = {}
    for row in admitted:
        task = int(row["task"])
        if task not in best or int(row["actual_cost"]) < int(best[task]["actual_cost"]):
            best[task] = row
    manifest = []
    for task, row in sorted(best.items()):
        manifest.append(
            {
                "task": task,
                "current_cost": row["current_actual_cost"],
                "candidate_cost": row["actual_cost"],
                "gain": row["gain"],
                "fresh_seed1": row["fresh5000_seeds"]["seed1"],
                "fresh_seed2": row["fresh5000_seeds"]["seed2"],
                "sha256": row["sha256"],
                "path": row["isolated_candidate"],
            }
        )
    (HERE / "admit_manifest.json").write_text(
        json.dumps(
            {
                "baseline": rel(BASE_ZIP),
                "status": "ADMIT_CLEAN_POLICY90" if manifest else "NO_CLEAN_POLICY90_CANDIDATE",
                "fresh_accuracy_threshold": args.threshold,
                "count": len(manifest),
                "total_projected_gain": sum(float(row["gain"]) for row in manifest),
                "candidates": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    checkpoint(complete=True)
    print(
        f"DONE admitted={len(manifest)} gain={sum(float(row['gain']) for row in manifest):.9f} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

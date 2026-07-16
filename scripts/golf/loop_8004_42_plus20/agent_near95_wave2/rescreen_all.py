#!/usr/bin/env python3
"""Exhaustively rescreen selected historical task models against the 8004.50 base.

This lane is intentionally non-promoting: it inventories every loose ONNX and
submission ZIP member for the requested tasks, SHA-deduplicates them, and then
applies the campaign structure, actual-cost, known-gold, shape-truth, dual-ORT,
and fresh-generator gates.  Only evidence and isolated candidate files are
written under this directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import re
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
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TARGETS = (396, 255, 196, 365, 48, 96, 23, 9, 202, 205)
BASE_ZIP = ROOT / "submission_base_8004.50.zip"
CURRENT_COSTS_JSON = ROOT / "scripts/golf/loop_8004_42_plus20/current_costs_8004_50.json"
ALL400_INVENTORY = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json"
TASK_MAP_PATH = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
MAX_BYTES = 1_440_000
FILE_RE = re.compile(r"^task(\d{3})(?:[^0-9].*)?\.onnx$", re.IGNORECASE)
MEMBER_RE = re.compile(r"(?:^|/)task(\d{3})\.onnx$", re.IGNORECASE)
PRIVATE_LINEAGE_TERMS = ("private0", "private_zero", "private-zero", "quarantine")

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_harvest"))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from harvest import actual_screen, known_score, run_bounded, screen_worker, structure_gate  # noqa: E402
from lib import scoring  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_baseline() -> tuple[dict[int, bytes], dict[int, int]]:
    with zipfile.ZipFile(BASE_ZIP) as archive:
        payloads = {task: archive.read(f"task{task:03d}.onnx") for task in TARGETS}
    raw = json.loads(CURRENT_COSTS_JSON.read_text())["costs"]
    costs = {task: int(raw[str(task)]) for task in TARGETS}
    return payloads, costs


def inventory() -> tuple[dict[int, dict[str, dict[str, Any]]], dict[str, Any]]:
    base, _ = load_baseline()
    base_sha = {task: sha256(data) for task, data in base.items()}
    candidates: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []

    def add(task: int, data: bytes, source: str, kind: str) -> None:
        counts[f"{kind}_observations"] += 1
        if len(data) > MAX_BYTES:
            counts["oversize_observations"] += 1
            return
        digest = sha256(data)
        if digest == base_sha[task]:
            counts["baseline_duplicates"] += 1
            return
        slot = candidates[task].setdefault(
            digest,
            {"data": data, "sha256": digest, "sources": [], "source_kinds": []},
        )
        slot["sources"].append(source)
        slot["source_kinds"].append(kind)

    # os.walk is considerably faster than resolving hundreds of thousands of
    # Path objects in this artifact-heavy repository.
    for directory, dirs, files in os.walk(ROOT):
        path_dir = Path(directory)
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "__pycache__"}]
        for name in files:
            match = FILE_RE.match(name)
            if not match:
                continue
            task = int(match.group(1))
            if task not in TARGETS:
                continue
            path = path_dir / name
            source = rel(path)
            try:
                add(task, path.read_bytes(), source, "loose")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})

    for directory, dirs, files in os.walk(ROOT):
        path_dir = Path(directory)
        dirs[:] = [name for name in dirs if name not in {".git", ".venv", "__pycache__"}]
        for name in files:
            if not name.lower().endswith(".zip"):
                continue
            path = path_dir / name
            source = rel(path)
            counts["zip_files_seen"] += 1
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        match = MEMBER_RE.search(member)
                        if not match:
                            continue
                        task = int(match.group(1))
                        if task in TARGETS:
                            add(task, archive.read(member), f"{source}::{member}", "zip")
            except Exception as exc:  # noqa: BLE001
                errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})

    counts["unique_different_sha"] = sum(len(rows) for rows in candidates.values())
    report = {
        "targets": list(TARGETS),
        "baseline_zip": rel(BASE_ZIP),
        "baseline_zip_sha256": sha256(BASE_ZIP.read_bytes()),
        "counts": dict(counts),
        "unique_by_task": {str(task): len(candidates[task]) for task in TARGETS},
        "errors": errors,
    }
    return dict(candidates), report


def strict_extra(data: bytes, sources: list[str]) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    detail: dict[str, Any] = {}
    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:  # noqa: BLE001
        return False, ["onnx_load"], {"error": f"{type(exc).__name__}: {exc}"}
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        detail["strict_data_prop"] = True
        nonstatic = []
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            if any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in value.type.tensor_type.shape.dim):
                nonstatic.append(value.name)
        detail["nonstatic"] = nonstatic
        if nonstatic:
            reasons.append("nonstatic")
    except Exception as exc:  # noqa: BLE001
        detail.update(strict_data_prop=False, strict_data_prop_error=f"{type(exc).__name__}: {exc}")
        reasons.append("strict_data_prop")

    ops = Counter(node.op_type for node in model.graph.node)
    giant_initializers = []
    for item in model.graph.initializer:
        try:
            if int(numpy_helper.to_array(item).size) >= 10_000:
                giant_initializers.append(item.name)
        except Exception:  # noqa: BLE001
            giant_initializers.append(item.name)
    detail["ops"] = dict(ops)
    detail["max_einsum_inputs"] = max(
        (len(node.input) for node in model.graph.node if node.op_type == "Einsum"),
        default=0,
    )
    detail["giant_initializers"] = giant_initializers
    detail["private_lineage_sources"] = [
        source for source in sources if any(term in source.lower() for term in PRIVATE_LINEAGE_TERMS)
    ]
    if ops.get("TfIdfVectorizer") or ops.get("Hardmax") or giant_initializers:
        reasons.append("lookup")
    if detail["max_einsum_inputs"] >= 15:
        reasons.append("giant_einsum")
    # A SHA seen in an explicitly quarantined/private-zero source retains that
    # lineage even if it was later copied into an ordinary submission archive.
    if detail["private_lineage_sources"]:
        reasons.append("private_zero_lineage")
    return not reasons, sorted(set(reasons)), detail


def make_session(model: onnx.ModelProto, disable_all: bool) -> ort.InferenceSession:
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
    model = onnx.load_model_from_string(data)
    output: dict[str, Any] = {}
    for disabled, mode in ((True, "disable_all"), (False, "default")):
        stats = {"right": 0, "wrong": 0, "errors": 0, "first_error": None}
        try:
            session = make_session(model, disabled)
        except Exception as exc:  # noqa: BLE001
            stats["session_error"] = f"{type(exc).__name__}: {exc}"
            output[mode] = stats
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
                    stats["right" if np.array_equal(raw > 0, benchmark["output"] > 0) else "wrong"] += 1
                except Exception as exc:  # noqa: BLE001
                    stats["errors"] += 1
                    if stats["first_error"] is None:
                        stats["first_error"] = f"{type(exc).__name__}: {exc}"
        output[mode] = stats
    return output


def fresh_dual(task: int, candidates: list[dict[str, Any]], count: int, seed: int) -> dict[str, Any]:
    task_map = json.loads(TASK_MAP_PATH.read_text())
    generator = importlib.import_module(f"task_{task_map[f'{task:03d}']}")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    sessions: dict[tuple[str, str], ort.InferenceSession] = {}
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    for row in candidates:
        digest = row["sha256"]
        model = onnx.load_model_from_string(row["data"])
        stats[digest] = {}
        for disabled, mode in ((True, "disable_all"), (False, "default")):
            stats[digest][mode] = {"right": 0, "wrong": 0, "errors": 0, "first_failure": None}
            try:
                sessions[(digest, mode)] = make_session(model, disabled)
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
    args = parser.parse_args()
    ort.set_default_logger_severity(4)
    started = time.time()
    baseline_data, current_costs = load_baseline()
    candidates, inventory_report = inventory()
    (HERE / "inventory.json").write_text(json.dumps(inventory_report, indent=2) + "\n")
    print(
        f"INVENTORY unique={inventory_report['counts']['unique_different_sha']} "
        f"by_task={inventory_report['unique_by_task']}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    data_by_sha: dict[str, bytes] = {}
    actual_jobs: list[tuple[str, int, bytes]] = []
    for task in TARGETS:
        base_sha = sha256(baseline_data[task])
        for digest, item in sorted(candidates.get(task, {}).items()):
            data = item["data"]
            sources = sorted(set(item["sources"]))
            row: dict[str, Any] = {
                "task": task,
                "sha256": digest,
                "bytes": len(data),
                "sources": sources,
                "source_kinds": sorted(set(item["source_kinds"])),
                "current_sha256": base_sha,
                "current_actual_cost": current_costs[task],
            }
            rows.append(row)
            model, reason, floor = structure_gate(data)
            row["static_floor"] = floor
            if model is None:
                row.update(stage="structure_reject", reasons=[reason])
                continue
            extra_ok, extra_reasons, detail = strict_extra(data, sources)
            row["strict_detail"] = detail
            if not extra_ok:
                row.update(stage="policy_reject", reasons=extra_reasons)
                continue
            if floor is None or floor >= current_costs[task]:
                row.update(stage="static_reject", reasons=["static_floor_not_strictly_cheaper"])
                continue
            row.update(stage="actual_pending", reasons=[])
            data_by_sha[digest] = data
            actual_jobs.append((digest, task, data))

    (HERE / "rescreen.json").write_text(json.dumps({"complete": False, "rows": rows}, indent=2) + "\n")
    print(f"STATIC actual_jobs={len(actual_jobs)}", flush=True)
    actual_results = run_bounded(actual_jobs, screen_worker, max_workers=4, timeout=25.0, label="ACTUAL")
    row_by_sha = {row["sha256"]: row for row in rows}
    known_candidates: list[dict[str, Any]] = []
    for result in actual_results:
        digest = result.get("sha256")
        if digest not in row_by_sha:
            continue
        row = row_by_sha[digest]
        cost = result.get("cost")
        row["actual_screen_cost"] = cost
        if cost is None or int(cost) >= int(row["current_actual_cost"]):
            row.update(stage="actual_reject", reasons=["actual_cost_not_strictly_cheaper_or_timeout"])
            continue
        row["stage"] = "known_pending"
        known_candidates.append(row)
    print(f"ACTUAL known_candidates={len(known_candidates)}", flush=True)

    pre_fresh: list[dict[str, Any]] = []
    for index, row in enumerate(known_candidates, 1):
        task = int(row["task"])
        digest = row["sha256"]
        data = data_by_sha[digest]
        print(f"KNOWN {index}/{len(known_candidates)} task{task:03d} {digest[:12]}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix=f"near95_{task:03d}_", dir="/tmp") as workdir:
                profile = known_score(data, task, True, f"near95_{digest[:10]}")
            row["official_like_score"] = profile
        except Exception as exc:  # noqa: BLE001
            row.update(stage="known_reject", reasons=["known_or_profile_error"], known_error=f"{type(exc).__name__}: {exc}")
            continue
        if not profile or not profile.get("correct") or int(profile["cost"]) >= int(row["current_actual_cost"]):
            row.update(stage="known_reject", reasons=["known_not_complete_or_not_cheaper"])
            continue
        row["actual_cost"] = int(profile["cost"])
        row["gain"] = math.log(int(row["current_actual_cost"]) / int(profile["cost"]))
        dual = known_dual(task, data)
        row["known_dual"] = dual
        if any(
            item.get("wrong") or item.get("errors") or item.get("session_error") or not item.get("right")
            for item in dual.values()
        ):
            row.update(stage="known_dual_reject", reasons=["known_dual"])
            continue
        try:
            trace = runtime_shape_trace(task, onnx.load_model_from_string(data))
            row["runtime_shape_trace"] = trace
        except Exception as exc:  # noqa: BLE001
            row.update(stage="shape_reject", reasons=["runtime_shape_trace"], shape_error=f"{type(exc).__name__}: {exc}")
            continue
        if trace["declared_actual_mismatches"]:
            row.update(stage="shape_reject", reasons=["shape_cloak"])
            continue
        row.update(stage="fresh_pending", reasons=[])
        pre_fresh.append({**row, "data": data})
        out = HERE / "candidates" / f"task{task:03d}_{digest[:12]}_cost{profile['cost']}.onnx"
        out.write_bytes(data)
        row["isolated_candidate"] = rel(out)
        (HERE / "rescreen.json").write_text(json.dumps({"complete": False, "rows": rows}, indent=2) + "\n")

    print(f"PRE_FRESH candidates={len(pre_fresh)}", flush=True)
    fresh_reports: dict[str, Any] = {}
    for task in TARGETS:
        task_rows = [row for row in pre_fresh if int(row["task"]) == task]
        if not task_rows:
            continue
        report = fresh_dual(task, task_rows, args.fresh, 80_045_000 + task)
        fresh_reports[str(task)] = report
        (HERE / "evidence" / f"task{task:03d}_fresh_dual_{args.fresh}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        for pre in task_rows:
            row = row_by_sha[pre["sha256"]]
            row["fresh_dual"] = report["candidates"][pre["sha256"]]
            passed = all(
                item["right"] / args.fresh >= 0.95
                and item["wrong"] + item["right"] == args.fresh
                and item["errors"] == 0
                and not item.get("session_error")
                for item in row["fresh_dual"].values()
            )
            row["stage"] = "fresh500_pass" if passed else "fresh500_reject"
            row["reasons"] = [] if passed else ["fresh_below_95_or_runtime_error"]

    winners = sorted(
        [row for row in rows if row["stage"] == "fresh500_pass"],
        key=lambda row: (int(row["task"]), int(row["actual_cost"])),
    )
    report = {
        "complete": True,
        "baseline_zip": rel(BASE_ZIP),
        "baseline_zip_sha256": sha256(BASE_ZIP.read_bytes()),
        "targets": list(TARGETS),
        "fresh_count": args.fresh,
        "inventory": inventory_report,
        "rows": rows,
        "fresh500_survivors": winners,
        "elapsed_seconds": time.time() - started,
    }
    (HERE / "rescreen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"DONE survivors={len(winners)} elapsed={report['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

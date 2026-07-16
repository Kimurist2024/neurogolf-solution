#!/usr/bin/env python3
"""Inventory historical exact-only rejects and preliminarily screen POLICY90 leads."""

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
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, helper, numpy_helper


ort.set_default_logger_severity(4)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8009.46.zip"
AUTHORITY_SHA256 = "4eb324d7ee835d2d325d9fd8acff68ba257413e1b48be3fa55a43a5860c58927"
CANONICAL_COSTS = ROOT / "scripts/golf/loop_8004_42_plus20/root_mem_census_119/canonical_costs.json"
ACTIVE_MANIFEST = ROOT / "others/71407/MANIFEST.json"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
TASKS_DIR = ROOT / "inputs/arc-gen-repo/tasks"
INVENTORY_OUT = HERE / "inventory.json"
CANDIDATES_OUT = HERE / "candidates.json"
POLICY_THRESHOLD = 0.90
FRESH_PER_SEED = 1_000
EXPECTED_IO = (1, 10, 30, 30)
CONFIGS = (
    ("disable_threads1", True, 1),
    ("disable_threads4", True, 4),
    ("default_threads1", False, 1),
    ("default_threads4", False, 4),
)
BASE_CONFIG = "disable_threads1"
RESCREEN_FILES = (
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20f_90/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_expand20h_92/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20_84/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20b_86/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20c_87/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20d_88/rescreen.json",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_mid20e_89/rescreen.json",
)
EXPLICIT_EXCLUDE = {7, 12, 71, 161}
# Conservative clean-lane catalog used by the all-history scanner.  It includes
# the public private-zero/unsound list plus unresolved monitored additions.
PRIVATE_ZERO_CATALOG = {
    9, 15, 35, 44, 48, 66, 70, 72, 77, 86, 90, 96, 101, 102, 112,
    133, 134, 138, 145, 157, 158, 169, 170, 173, 174, 178, 185, 187,
    192, 196, 198, 202, 205, 208, 209, 216, 219, 222, 233, 246, 255,
    277, 285, 286, 302, 319, 325, 333, 343, 344, 346, 361, 365, 366,
    372, 377, 379, 391, 393, 396,
}

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(TASKS_DIR))
sys.path.insert(0, str(ROOT / "scripts/golf/loop_7999_13/lane_c11"))
from audit_candidates import runtime_shape_trace  # noqa: E402
from lib import scoring  # noqa: E402


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCREEN = load_module(
    "policy90_backlog_screen_helper",
    ROOT / "scripts/golf/loop_8004_42_plus20/agent_clean95_all/screen_all.py",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def case_id(example: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("input", "output"):
        array = np.asarray(example[key], dtype=np.uint8)
        digest.update(np.asarray(array.shape, dtype=np.int16).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_candidate(row: dict[str, Any]) -> tuple[bytes | None, str | None, list[dict[str, Any]]]:
    errors = []
    for source in row["sources"]:
        data = SCREEN.resolve_source(source, int(row["task"]))
        if data is None:
            errors.append({"source": source, "error": "unresolved"})
            continue
        digest = sha256(data)
        if digest != row["sha256"]:
            errors.append({"source": source, "error": "sha_mismatch", "actual_sha256": digest})
            continue
        return data, source, errors
    return None, None, errors


def nested_graph_count(model: onnx.ModelProto) -> int:
    count = 0
    pending = list(model.graph.node)
    while pending:
        node = pending.pop()
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                count += 1
                pending.extend(attribute.g.node)
            elif attribute.type == AttributeProto.GRAPHS:
                count += len(attribute.graphs)
                for graph in attribute.graphs:
                    pending.extend(graph.node)
    return count


def extended_structure(task: int, data: bytes, sources: list[str]) -> dict[str, Any]:
    audit = SCREEN.static_audit(data, sources, task)
    try:
        model = onnx.load_model_from_string(data)
        arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
        nonfinite = [
            name for name, array in arrays.items()
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
        ]
        # Keep the repository's established lookup definition: learned/table
        # vectorizers, Hardmax, or a giant initializer.  Standard semantic
        # OneHot/Gather/Scatter operations are not lookup by themselves.
        lookup_ops = sorted({
            node.op_type for node in model.graph.node
            if node.op_type in {"TfIdfVectorizer", "Hardmax", "CategoryMapper"}
        })
        audit.update({
            "nonfinite_initializers": nonfinite,
            "lookup_ops_extended": lookup_ops,
            "nested_graphs_recursive": nested_graph_count(model),
            "functions_exact": len(model.functions),
            "sparse_initializers_exact": len(model.graph.sparse_initializer),
            "external_initializers_exact": [
                item.name for item in model.graph.initializer
                if item.data_location == TensorProto.EXTERNAL or item.external_data
            ],
            "initializer_elements": int(sum(array.size for array in arrays.values())),
            "largest_initializer_elements": int(max((array.size for array in arrays.values()), default=0)),
            "file_bytes": len(data),
        })
        if nonfinite:
            audit["reasons"].append("nonfinite_initializer")
        if lookup_ops:
            audit["reasons"].append("extended_lookup_op")
        audit["reasons"] = sorted(set(audit["reasons"]))
        audit["pass"] = bool(audit["pass"] and not nonfinite and not lookup_ops)
    except Exception as exc:  # noqa: BLE001
        audit["pass"] = False
        audit.setdefault("reasons", []).append(f"extended_structure_exception:{type(exc).__name__}")
        audit["extended_structure_error"] = str(exc)
    return audit


def make_session(data: bytes, disable_all: bool, threads: int) -> ort.InferenceSession:
    model = onnx.load_model_from_string(data)
    sanitized = scoring.sanitize_model(copy.deepcopy(model))
    if sanitized is None:
        raise RuntimeError("official sanitizer rejected model")
    options = ort.SessionOptions()
    if disable_all:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.log_severity_level = 4
    return ort.InferenceSession(
        sanitized.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


def sign_difference(left: bytes, right: bytes) -> int:
    xor = np.bitwise_xor(
        np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)
    )
    return int(np.unpackbits(xor).sum())


def evaluate_config(
    runtime: ort.InferenceSession,
    cases: list[dict[str, Any]],
    locations: list[dict[str, Any]],
    baseline_signs: list[bytes | None] | None,
    baseline_raw_hashes: list[str | None] | None,
) -> tuple[dict[str, Any], list[bytes | None], list[str | None], list[bool | None]]:
    right = wrong = errors = 0
    nonfinite_cases = nonfinite_elements = shape_mismatches = 0
    near_positive = 0
    minimum_positive = math.inf
    maximum_nonpositive = -math.inf
    sign_mismatch_cases = sign_mismatch_cells = raw_mismatch_cases = 0
    first_wrong = first_error = first_shape = first_sign_mismatch = None
    signs: list[bytes | None] = []
    raw_hashes: list[str | None] = []
    correctness: list[bool | None] = []
    sign_digest = hashlib.sha256()
    raw_digest = hashlib.sha256()
    started = time.monotonic()
    for index, (example, location) in enumerate(zip(cases, locations)):
        benchmark = scoring.convert_to_numpy(example)
        if benchmark is None:
            raise RuntimeError(f"official converter rejected case: {location}")
        expected = benchmark["output"] > 0
        try:
            raw = np.asarray(runtime.run(["output"], {"input": benchmark["input"]})[0])
        except Exception as exc:  # noqa: BLE001
            errors += 1
            signs.append(None)
            raw_hashes.append(None)
            correctness.append(None)
            if first_error is None:
                first_error = {**location, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if tuple(raw.shape) != EXPECTED_IO:
            shape_mismatches += 1
            signs.append(None)
            raw_hashes.append(None)
            correctness.append(None)
            if first_shape is None:
                first_shape = {**location, "actual": list(raw.shape)}
            continue
        finite = np.isfinite(raw)
        current_nonfinite = int(np.count_nonzero(~finite))
        nonfinite_cases += int(current_nonfinite > 0)
        nonfinite_elements += current_nonfinite
        positive = raw > 0
        packed = np.packbits(positive.reshape(-1), bitorder="little").tobytes()
        raw_bytes = np.ascontiguousarray(raw).tobytes()
        raw_sha = sha256(raw_bytes)
        signs.append(packed)
        raw_hashes.append(raw_sha)
        sign_digest.update(packed)
        raw_digest.update(raw_bytes)
        correct = bool(np.array_equal(positive, expected))
        correctness.append(correct)
        right += int(correct)
        wrong += int(not correct)
        if not correct and first_wrong is None:
            first_wrong = {**location, "different_cells": int(np.count_nonzero(positive != expected))}
        if np.any(positive):
            minimum_positive = min(minimum_positive, float(raw[positive].min()))
            near_positive += int(np.count_nonzero((raw > 0) & (raw < 0.25)))
        nonpositive = finite & ~positive
        if np.any(nonpositive):
            maximum_nonpositive = max(maximum_nonpositive, float(raw[nonpositive].max()))
        if baseline_signs is not None:
            baseline = baseline_signs[index]
            difference = math.prod(EXPECTED_IO) if baseline is None else sign_difference(packed, baseline)
            sign_mismatch_cases += int(difference > 0)
            sign_mismatch_cells += difference
            raw_mismatch_cases += int(raw_sha != baseline_raw_hashes[index])
            if difference and first_sign_mismatch is None:
                first_sign_mismatch = {**location, "different_cells": difference}
    total = len(cases)
    return ({
        "total": total,
        "right": right,
        "wrong": wrong,
        "accuracy": right / total,
        "policy90": right / total >= POLICY_THRESHOLD,
        "errors": errors,
        "nonfinite_cases": nonfinite_cases,
        "nonfinite_elements": nonfinite_elements,
        "output_shape_mismatches": shape_mismatches,
        "near_positive_elements_0_to_0_25": near_positive,
        "minimum_positive": None if minimum_positive == math.inf else minimum_positive,
        "maximum_nonpositive": None if maximum_nonpositive == -math.inf else maximum_nonpositive,
        "sign_mismatch_cases_vs_disable_threads1": sign_mismatch_cases,
        "sign_mismatch_cells_vs_disable_threads1": sign_mismatch_cells,
        "raw_mismatch_cases_vs_disable_threads1": raw_mismatch_cases,
        "first_wrong": first_wrong,
        "first_error": first_error,
        "first_shape_mismatch": first_shape,
        "first_sign_mismatch": first_sign_mismatch,
        "sign_sha256": sign_digest.hexdigest(),
        "raw_sha256": raw_digest.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }, signs, raw_hashes, correctness)


def evaluate_four(data: bytes, cases: list[dict[str, Any]], locations: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: dict[str, Any] = {}
    auxiliary: dict[str, Any] = {}
    baseline_signs = baseline_raw_hashes = None
    for config_name, disable_all, threads in CONFIGS:
        try:
            runtime = make_session(data, disable_all, threads)
        except Exception as exc:  # noqa: BLE001
            row = {
                "total": len(cases), "right": 0, "wrong": 0, "accuracy": 0.0,
                "policy90": False, "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}",
                "nonfinite_cases": 0, "nonfinite_elements": 0, "output_shape_mismatches": 0,
                "near_positive_elements_0_to_0_25": 0,
                "sign_mismatch_cases_vs_disable_threads1": 0,
                "sign_mismatch_cells_vs_disable_threads1": 0,
                "raw_mismatch_cases_vs_disable_threads1": 0,
            }
            rows[config_name] = row
            auxiliary[config_name] = {"signs": [None] * len(cases), "raw_hashes": [None] * len(cases), "correctness": [None] * len(cases)}
            if config_name == BASE_CONFIG:
                baseline_signs = [None] * len(cases)
                baseline_raw_hashes = [None] * len(cases)
            continue
        row, signs, raw_hashes, correctness = evaluate_config(
            runtime, cases, locations,
            None if config_name == BASE_CONFIG else baseline_signs,
            None if config_name == BASE_CONFIG else baseline_raw_hashes,
        )
        row["disable_all"] = disable_all
        row["threads"] = threads
        rows[config_name] = row
        auxiliary[config_name] = {"signs": signs, "raw_hashes": raw_hashes, "correctness": correctness}
        if config_name == BASE_CONFIG:
            baseline_signs = signs
            baseline_raw_hashes = raw_hashes
    return rows, auxiliary


def runtime_clean(row: dict[str, Any]) -> bool:
    return bool(
        row.get("errors") == 0 and not row.get("session_error")
        and row.get("nonfinite_cases") == 0 and row.get("nonfinite_elements") == 0
        and row.get("output_shape_mismatches") == 0
    )


def config_stable(row: dict[str, Any]) -> bool:
    return bool(
        row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and row.get("raw_mismatch_cases_vs_disable_threads1") == 0
    )


def known_cases(task: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    examples = scoring.load_examples(task)
    cases = []
    locations = []
    counts: dict[str, Any] = {
        "raw": {subset: len(examples[subset]) for subset in ("train", "test", "arc-gen")},
        "converted": {subset: 0 for subset in ("train", "test", "arc-gen")},
        "conversion_skips": {subset: 0 for subset in ("train", "test", "arc-gen")},
    }
    for subset in ("train", "test", "arc-gen"):
        for index, example in enumerate(examples[subset]):
            if scoring.convert_to_numpy(example) is None:
                counts["conversion_skips"][subset] += 1
                continue
            counts["converted"][subset] += 1
            cases.append(example)
            locations.append({"subset": subset, "index": index, "case_id": case_id(example)})
    counts["raw_total"] = sum(counts["raw"].values())
    counts["converted_total"] = sum(counts["converted"].values())
    counts["conversion_skip_total"] = sum(counts["conversion_skips"].values())
    return cases, locations, counts


def official_measure(task: int, data: bytes, label: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory(prefix=f"backlog281_{task:03d}_{label}_", dir="/tmp") as work:
        return scoring.score_and_verify(
            onnx.load_model_from_string(data), task, work, label=label, require_correct=False
        )


def shape_trace(task: int, data: bytes) -> dict[str, Any]:
    try:
        result = runtime_shape_trace(task, onnx.load_model_from_string(data))
        mismatch = result.get("declared_actual_mismatches", [])
        result["truthful"] = not mismatch and not result.get("error")
        return result
    except Exception as exc:  # noqa: BLE001
        return {"truthful": False, "error": f"{type(exc).__name__}: {exc}"}


def fresh_cases(task: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    module_name = f"task_{task_map[f'{task:03d}']}"
    generator = importlib.import_module(module_name)
    common = importlib.import_module("common")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    common.random.seed(seed)
    cases = []
    locations = []
    digest = hashlib.sha256()
    seen = set()
    attempts = generation_errors = conversion_skips = 0
    while len(cases) < FRESH_PER_SEED:
        attempts += 1
        try:
            example = generator.generate()
        except Exception:  # noqa: BLE001
            generation_errors += 1
            continue
        if scoring.convert_to_numpy(example) is None:
            conversion_skips += 1
            continue
        identifier = case_id(example)
        digest.update(bytes.fromhex(identifier))
        seen.add(identifier)
        locations.append({"seed": seed, "index": len(cases), "case_id": identifier})
        cases.append(example)
    return cases, locations, {
        "task": task, "seed": seed, "module": module_name,
        "requested": FRESH_PER_SEED, "accepted": len(cases), "attempts": attempts,
        "generation_errors": generation_errors, "conversion_skips": conversion_skips,
        "unique_case_ids": len(seen), "case_stream_sha256": digest.hexdigest(),
    }


def comparison(
    candidate_aux: dict[str, Any], authority_aux: dict[str, Any]
) -> dict[str, Any]:
    rows = {}
    for config_name, _disable, _threads in CONFIGS:
        cand = candidate_aux[config_name]
        auth = authority_aux[config_name]
        out = {
            "total_comparable": 0, "both_correct": 0,
            "authority_correct_candidate_wrong": 0,
            "candidate_correct_authority_wrong": 0, "both_wrong": 0,
            "sign_mismatch_cases": 0, "sign_mismatch_cells": 0,
        }
        for cs, cr, aus, aur in zip(cand["signs"], cand["correctness"], auth["signs"], auth["correctness"]):
            if cs is None or aus is None or cr is None or aur is None:
                continue
            out["total_comparable"] += 1
            if cr and aur:
                out["both_correct"] += 1
            elif aur:
                out["authority_correct_candidate_wrong"] += 1
            elif cr:
                out["candidate_correct_authority_wrong"] += 1
            else:
                out["both_wrong"] += 1
            difference = sign_difference(cs, aus)
            out["sign_mismatch_cases"] += int(difference > 0)
            out["sign_mismatch_cells"] += difference
        rows[config_name] = out
    return rows


def load_inventory() -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, bytes], dict[int, int], set[int]]:
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("immutable 8009.46 authority changed")
    canonical = json.loads(CANONICAL_COSTS.read_text(encoding="utf-8"))
    if canonical.get("authority_zip") != "submission_base_8009.46.zip" or canonical.get("count") != 400:
        raise RuntimeError("canonical cost census does not describe 8009.46")
    authority_costs = {int(item["task"]): int(item["cost"]) for item in canonical["ranked"]}
    active_manifest = json.loads(ACTIVE_MANIFEST.read_text(encoding="utf-8"))
    active_tasks = {int(item["task"]) for item in active_manifest["active_candidates"]}
    effective_exclude = active_tasks | EXPLICIT_EXCLUDE
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority_data = {task: archive.read(f"task{task:03d}.onnx") for task in range(1, 401)}

    per_task: dict[int, dict[str, Any]] = {}
    preknown: dict[tuple[int, str], dict[str, Any]] = {}
    source_summaries = []
    all_targets = []
    for path in RESCREEN_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = [int(item) for item in payload["targets"]]
        all_targets.extend(targets)
        stage_counts = Counter(row.get("stage", "missing") for row in payload["rows"])
        source_summaries.append({
            "path": rel(path),
            "baseline_zip": payload.get("baseline_zip"),
            "baseline_zip_sha256": payload.get("baseline_zip_sha256"),
            "target_count": len(targets),
            "row_count": len(payload["rows"]),
            "stage_counts": dict(stage_counts),
        })
        rows_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in payload["rows"]:
            rows_by_task[int(row["task"])].append(row)
        for task in targets:
            rows = rows_by_task[task]
            task_stage = Counter(row.get("stage", "missing") for row in rows)
            strict_known_rows = []
            for row in rows:
                if row.get("stage") != "known_reject":
                    continue
                old_cost = row.get("actual_screen_cost")
                if old_cost is None or int(old_cost) >= authority_costs[task]:
                    continue
                candidate = {
                    "task": task,
                    "sha256": row["sha256"],
                    "sources": list(row.get("sources", [])),
                    "source_kinds": list(row.get("source_kinds", [])),
                    "rescreen_source": rel(path),
                    "historical_stage": row.get("stage"),
                    "historical_reasons": row.get("reasons", []),
                    "historical_static_floor": row.get("static_floor"),
                    "historical_actual_screen_cost": int(old_cost),
                    "historical_authority_cost": row.get("current_actual_cost"),
                    "authority_8009_46_cost": authority_costs[task],
                    "authority_8009_46_sha256": sha256(authority_data[task]),
                }
                strict_known_rows.append(row["sha256"])
                key = (task, row["sha256"])
                if key in preknown:
                    preknown[key]["sources"] = sorted(set(preknown[key]["sources"] + candidate["sources"]))
                else:
                    preknown[key] = candidate
            exclusion_reasons = []
            if task in active_tasks:
                exclusion_reasons.append("71407_active")
            if task in EXPLICIT_EXCLUDE:
                exclusion_reasons.append("explicit_task007_012_071_161")
            if task in PRIVATE_ZERO_CATALOG:
                exclusion_reasons.append("private_zero_or_unsound_monitor")
            per_task[task] = {
                "task": task,
                "authority_cost": authority_costs[task],
                "authority_sha256": sha256(authority_data[task]),
                "rescreen_source": rel(path),
                "historical_baseline_zip": payload.get("baseline_zip"),
                "observed_unique_sha_rows": len(rows),
                "historical_stage_counts": dict(task_stage),
                "strict_lower_known_reject_shas_vs_8009_46": strict_known_rows,
                "excluded": bool(exclusion_reasons),
                "exclusion_reasons": exclusion_reasons,
            }

    if len(all_targets) != len(set(all_targets)):
        raise RuntimeError("rescreen target waves overlap; inventory accounting is not disjoint")
    eligible_preknown = [
        row for (task, _digest), row in preknown.items()
        if task not in effective_exclude and task not in PRIVATE_ZERO_CATALOG
    ]
    inventory = {
        "lane": "agent_policy90_backlog_281",
        "authority": {
            "zip": rel(AUTHORITY), "sha256": AUTHORITY_SHA256,
            "canonical_cost_source": rel(CANONICAL_COSTS), "task_count": 400,
        },
        "source_rescreens": source_summaries,
        "coverage": {
            "target_observations": len(all_targets),
            "unique_tasks": len(set(all_targets)),
            "source_target_sets_disjoint": True,
            "tasks_after_active_and_explicit_exclusion": len(set(all_targets) - effective_exclude),
            "tasks_after_private_active_explicit_exclusion": len(set(all_targets) - effective_exclude - PRIVATE_ZERO_CATALOG),
            "historical_rows_total": sum(item["row_count"] for item in source_summaries),
            "strict_lower_known_reject_candidate_sha_before_policy_exclusion": len(preknown),
            "strict_lower_known_reject_candidate_sha_after_policy_exclusion": len(eligible_preknown),
            "strict_lower_known_reject_tasks_after_policy_exclusion": len({row["task"] for row in eligible_preknown}),
        },
        "exclusions": {
            "explicit": sorted(EXPLICIT_EXCLUDE),
            "active_71407": sorted(active_tasks),
            "effective_active_or_explicit": sorted(effective_exclude),
            "private_zero_or_unsound_monitor": sorted(PRIVATE_ZERO_CATALOG),
        },
        "tasks": [per_task[task] for task in sorted(per_task)],
    }
    return inventory, sorted(eligible_preknown, key=lambda row: (row["task"], row["historical_actual_screen_cost"], row["sha256"])), authority_data, authority_costs, active_tasks


def main() -> int:
    started = time.monotonic()
    inventory, preknown, authority_data, authority_costs, active_tasks = load_inventory()
    INVENTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_OUT.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"inventory_tasks": inventory["coverage"]["unique_tasks"], "preknown_candidates": len(preknown)}), flush=True)

    known_cache: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]] = {}
    rows = []
    qualified = []
    for index, source_row in enumerate(preknown, start=1):
        row = copy.deepcopy(source_row)
        task = int(row["task"])
        data, resolved_source, resolution_errors = resolve_candidate(row)
        row["resolved_source"] = resolved_source
        row["resolution_errors"] = resolution_errors
        if data is None:
            row.update(classification="REJECT_UNRESOLVED", qualified_for_fresh=False)
            rows.append(row)
            continue
        row["resolved_sha256"] = sha256(data)
        row["structure"] = extended_structure(task, data, row["sources"])
        if not row["structure"]["pass"]:
            row.update(classification="REJECT_CURRENT_STRUCTURE_POLICY", qualified_for_fresh=False)
            rows.append(row)
            continue
        if task not in known_cache:
            known_cache[task] = known_cases(task)
        cases, locations, split_counts = known_cache[task]
        row["known_split_counts"] = split_counts
        known, _aux = evaluate_four(data, cases, locations)
        row["known_four"] = known
        known_policy90 = all(item["policy90"] for item in known.values())
        runtime_safe = all(runtime_clean(item) for item in known.values())
        sign_stable = all(config_stable(item) for item in known.values())
        margin_clean = all(item["near_positive_elements_0_to_0_25"] == 0 for item in known.values())
        row["known_policy90_all_four"] = known_policy90
        row["known_runtime_safe_all_four"] = runtime_safe
        row["known_sign_and_raw_stable_all_four"] = sign_stable
        row["known_margin_clean_all_four"] = margin_clean
        if not known_policy90:
            row.update(classification="REJECT_KNOWN_BELOW_POLICY90", qualified_for_fresh=False)
        elif not runtime_safe:
            row.update(classification="REJECT_KNOWN_RUNTIME", qualified_for_fresh=False)
        elif not sign_stable:
            row.update(classification="REJECT_CONFIG_INSTABILITY", qualified_for_fresh=False)
        elif not margin_clean:
            row.update(classification="REJECT_MARGIN", qualified_for_fresh=False)
        else:
            row["runtime_shape_trace"] = shape_trace(task, data)
            row["official_profile"] = official_measure(task, data, row["sha256"][:10])
            profile = row["official_profile"]
            truthful = bool(row["runtime_shape_trace"].get("truthful"))
            lower = bool(profile and int(profile["cost"]) < authority_costs[task])
            row["official_profile_strict_lower"] = lower
            if not truthful:
                row.update(classification="REJECT_RUNTIME_SHAPE_CLOAK", qualified_for_fresh=False)
            elif not lower:
                row.update(classification="REJECT_REPROFILE_NOT_STRICT_LOWER", qualified_for_fresh=False)
            else:
                row.update(classification="QUALIFIED_KNOWN_POLICY90", qualified_for_fresh=True)
                row["projected_gain"] = math.log(authority_costs[task] / int(profile["cost"]))
                row["_data"] = data
                qualified.append(row)
        rows.append(row)
        if index % 10 == 0 or row.get("qualified_for_fresh"):
            print(json.dumps({
                "screened": index, "total": len(preknown), "task": task,
                "sha": row["sha256"][:12], "class": row["classification"],
                "minimum_known_accuracy": min(item["accuracy"] for item in known.values()),
            }), flush=True)

    # One independent best lead per task enters the preliminary fresh screen.
    selected = []
    for task in sorted({int(row["task"]) for row in qualified}):
        pool = [row for row in qualified if int(row["task"]) == task]
        pool.sort(key=lambda row: (
            -min(item["accuracy"] for item in row["known_four"].values()),
            int(row["official_profile"]["cost"]),
            row["sha256"],
        ))
        winner = pool[0]
        winner["fresh_selection_reason"] = "best minimum known accuracy, then lowest reprofiled cost, then SHA"
        winner["fresh_alternates_same_task"] = len(pool) - 1
        selected.append(winner)

    fresh_results = []
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    for selected_index, row in enumerate(selected, start=1):
        task = int(row["task"])
        candidate_data = row["_data"]
        seeds = (281_000_000 + task, 281_100_000 + task)
        seed_rows = []
        for seed in seeds:
            cases, locations, generation = fresh_cases(task, seed)
            candidate_four, candidate_aux = evaluate_four(candidate_data, cases, locations)
            authority_four, authority_aux = evaluate_four(authority_data[task], cases, locations)
            seed_rows.append({
                "seed": seed,
                "generation": generation,
                "candidate": candidate_four,
                "authority": authority_four,
                "candidate_vs_authority": comparison(candidate_aux, authority_aux),
            })
            print(json.dumps({
                "fresh_task": task, "seed": seed,
                "candidate_right": {name: item["right"] for name, item in candidate_four.items()},
                "authority_right": {name: item["right"] for name, item in authority_four.items()},
            }), flush=True)
        fresh_pass = all(
            item["candidate"][name]["policy90"]
            and runtime_clean(item["candidate"][name])
            and config_stable(item["candidate"][name])
            and item["candidate"][name]["near_positive_elements_0_to_0_25"] == 0
            for item in seed_rows for name, _disable, _threads in CONFIGS
        )
        fresh_results.append({
            "task": task,
            "sha256": row["sha256"],
            "generator_module": f"task_{task_map[f'{task:03d}']}",
            "seeds": list(seeds),
            "count_per_seed": FRESH_PER_SEED,
            "preliminary_fresh_policy90_pass": fresh_pass,
            "runs": seed_rows,
        })
        row["preliminary_fresh_policy90_pass"] = fresh_pass

    # Strip in-memory bytes and persist a complete audit ledger.
    for row in rows:
        row.pop("_data", None)
    for row in qualified:
        row.pop("_data", None)
    classifications = Counter(row["classification"] for row in rows)
    fresh_pass_rows = [row for row in fresh_results if row["preliminary_fresh_policy90_pass"]]
    payload = {
        "lane": "agent_policy90_backlog_281",
        "authority_zip": rel(AUTHORITY),
        "authority_sha256": AUTHORITY_SHA256,
        "policy_threshold": POLICY_THRESHOLD,
        "fresh_count_per_seed": FRESH_PER_SEED,
        "inventory_source": rel(INVENTORY_OUT),
        "source_candidate_count": len(preknown),
        "source_task_count": len({row["task"] for row in preknown}),
        "classification_counts": dict(classifications),
        "known_policy90_qualified_count": len(qualified),
        "known_policy90_qualified_tasks": sorted({int(row["task"]) for row in qualified}),
        "fresh_selected_count": len(selected),
        "fresh_selected_tasks": [int(row["task"]) for row in selected],
        "preliminary_fresh_pass_count": len(fresh_pass_rows),
        "preliminary_fresh_pass_tasks": [int(row["task"]) for row in fresh_pass_rows],
        "selection_policy": "one lead per task: maximum minimum known-four accuracy, then lowest reprofiled cost, then SHA",
        "candidates": rows,
        "fresh_results": fresh_results,
        "elapsed_seconds": time.monotonic() - started,
        "protected_writes": "none; only this lane inventory/candidates outputs",
    }
    CANDIDATES_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    inventory["screen_result"] = {
        "candidate_ledger": rel(CANDIDATES_OUT),
        "classification_counts": dict(classifications),
        "known_policy90_qualified_count": len(qualified),
        "known_policy90_qualified_tasks": sorted({int(row["task"]) for row in qualified}),
        "fresh_selected_count": len(selected),
        "preliminary_fresh_pass_count": len(fresh_pass_rows),
        "preliminary_fresh_pass_tasks": [int(row["task"]) for row in fresh_pass_rows],
    }
    INVENTORY_OUT.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "classifications": dict(classifications),
        "qualified": len(qualified),
        "qualified_tasks": sorted({int(row["task"]) for row in qualified}),
        "fresh_selected": len(selected),
        "fresh_pass_tasks": [int(row["task"]) for row in fresh_pass_rows],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

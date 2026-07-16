#!/usr/bin/env python3
"""Fail-closed full cost-11..25 scan against the immutable 8011.05 authority."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
SCORES = ROOT / "all_scores.csv"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
EXTRA_PATH = ROOT / "scripts/golf/extra15_cost25_scan_294/scan.py"
ARCHIVE_INDEX = ROOT / "scripts/golf/loop_7999_13/lane_archive_all400/inventory.json"
EVIDENCE = HERE / "evidence.json"
CANDIDATES = HERE / "candidates"
FRESH_PER_SEED = 2_000
EXPECTED = [1, 10, 30, 30]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SUPPORT = load_module("cost11_25_support", SUPPORT_PATH)
EXTRA = load_module("cost11_25_extra_templates", EXTRA_PATH)
SUPPORT.FRESH_PER_SEED = FRESH_PER_SEED


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_costs() -> dict[int, int]:
    with SCORES.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["task"][4:]): int(row["cost"])
            for row in rows if 11 <= int(row["cost"]) <= 25
        }


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0 and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0 and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("sign_mismatch_cases_vs_disable_threads1") == 0
        and row.get("sign_mismatch_cells_vs_disable_threads1") == 0
        and not row.get("session_error")
    )


def four_exact(rows: dict[str, Any]) -> bool:
    return bool(len(rows) == 4 and all(exact(row) for row in rows.values()))


def audited_cost(structure: dict[str, Any]) -> int | None:
    memory = structure.get("runtime_intermediate_trace", {}).get(
        "single_example_intermediate_bytes"
    )
    if not isinstance(memory, int):
        return None
    return int(structure["initializer_elements"]) + memory


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
        "nonfinite_elements", "runtime_shape_mismatches",
        "small_positive_elements_0_to_0_25", "minimum_positive", "maximum_nonpositive",
        "sign_mismatch_cases_vs_disable_threads1", "sign_mismatch_cells_vs_disable_threads1",
        "session_error", "optimization", "threads",
    )
    return {key: row.get(key) for key in keys if key in row}


def rank_shrink_variants(base: onnx.ModelProto) -> list[dict[str, Any]]:
    if not base.graph.node or base.graph.node[0].op_type != "Einsum":
        return []
    specs = []
    for init_index, item in enumerate(base.graph.initializer):
        array = numpy_helper.to_array(item)
        for axis, dimension in enumerate(array.shape):
            if dimension == 2:
                specs.append((init_index, axis))
    if not specs or len(specs) > 4:
        return []
    rows = []
    for choices in itertools.product((0, 1), repeat=len(specs)):
        model = copy.deepcopy(base)
        arrays = [numpy_helper.to_array(item) for item in model.graph.initializer]
        for (init_index, axis), choice in zip(specs, choices):
            arrays[init_index] = np.take(arrays[init_index], [choice], axis=axis)
        for index, array in enumerate(arrays):
            name = model.graph.initializer[index].name
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array, name))
        data = model.SerializeToString()
        rows.append({
            "name": "rank1_axes_" + "".join(map(str, choices)),
            "family": "einsum_rank1_slice", "proof": "shrink every latent dimension 2 to 1",
            "sha256": sha256(data), "_model": model, "_data": data,
        })
    return rows


def optional_conv_integer_variants(base: onnx.ModelProto) -> list[dict[str, Any]]:
    result = []
    node_index = next(
        (index for index, node in enumerate(base.graph.node) if node.op_type == "ConvInteger"),
        None,
    )
    if node_index is None:
        return result
    arity = len(base.graph.node[node_index].input)
    for keep in range(2, arity):
        model = copy.deepcopy(base)
        del model.graph.node[node_index].input[keep:]
        model = onnxoptimizer.optimize(
            model, ["eliminate_deadend", "eliminate_unused_initializer"]
        )
        data = model.SerializeToString()
        result.append({
            "name": f"conv_integer_keep{keep}", "family": "optional_input_removal",
            "proof": "remove optional zero-point input and dead producers",
            "sha256": sha256(data), "_model": model, "_data": data,
        })
    return result


def historical_variants(task: int, cost: int) -> list[dict[str, Any]]:
    index = json.loads(ARCHIVE_INDEX.read_text(encoding="utf-8"))["retained"]
    result = []
    for row in index.get(str(task), []):
        if int(row["static_cost"]) >= cost:
            continue
        path = ROOT / row["path"]
        data = path.read_bytes()
        result.append({
            "name": f"history_{path.stem}", "family": "historical_archive",
            "proof": f"archive exhaustive scan source_count={row['source_count']}",
            "sha256": sha256(data), "_model": onnx.load_from_string(data), "_data": data,
            "history": row,
        })
    return result


def task_variants(task: int, cost: int, base: onnx.ModelProto) -> list[dict[str, Any]]:
    rows = [dict(row) for row in EXTRA.generic_variants()]
    if task in set(EXTRA.TASKS):
        rows.extend(EXTRA.task_specific_variants(task, base))
    rows.extend(rank_shrink_variants(base))
    rows.extend(optional_conv_integer_variants(base))
    rows.extend(historical_variants(task, cost))
    unique = {}
    for row in rows:
        unique.setdefault(row["sha256"], row)
    return list(unique.values())


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("8011.05 authority hash mismatch")
    costs = load_costs()
    if len(costs) != 27:
        raise RuntimeError(f"cost11..25 inventory changed: {len(costs)}")
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    tasks = []
    finalists = []
    structure_cache = {}
    with zipfile.ZipFile(AUTHORITY) as archive:
        authority = {task: archive.read(f"task{task:03d}.onnx") for task in costs}
    for task, authority_cost in sorted(costs.items(), key=lambda item: (item[1], item[0])):
        base_data = authority[task]
        base = onnx.load_from_string(base_data)
        profile = SUPPORT.official_profile(task, base, "authority801105")
        if profile is None or int(profile["cost"]) != authority_cost:
            raise RuntimeError(f"authority profile mismatch task{task:03d}: {profile}")
        cases, known_counts = SUPPORT.known_cases(task)
        quick_cases = cases[: min(12, len(cases))]
        variants = task_variants(task, authority_cost, base)
        best = {"right": -1, "name": None, "screen_total": len(quick_cases)}
        accepted = []
        for source in variants:
            data = source["_data"]
            digest = source["sha256"]
            cache_key = (task, digest) if source["family"] == "historical_archive" else digest
            if cache_key not in structure_cache:
                structure_cache[cache_key] = SUPPORT.structural_audit(
                    task, source["_model"], data
                )
            structure = structure_cache[cache_key]
            cost = audited_cost(structure)
            row = {
                "task": task, "name": source["name"], "family": source["family"],
                "sha256": digest, "authority_cost": authority_cost, "audited_cost": cost,
                "structure_pass": structure["pass"], "structure_reasons": structure["reasons"],
            }
            if not structure["pass"]:
                row["classification"] = "REJECT_STRUCTURE"
                rows.append(row)
                continue
            if cost is None or cost >= authority_cost:
                row["classification"] = "REJECT_NOT_STRICT_LOWER"
                rows.append(row)
                continue
            try:
                run = SUPPORT.make_session(data, True, 1)
                quick, _ = SUPPORT.evaluate_config(run, quick_cases, None)
            except Exception as exc:  # noqa: BLE001
                quick = {
                    "total": len(quick_cases), "right": 0, "wrong": 0,
                    "errors": len(quick_cases), "session_error": f"{type(exc).__name__}: {exc}",
                    "nonfinite_cases": 0, "nonfinite_elements": 0,
                    "runtime_shape_mismatches": 0, "small_positive_elements_0_to_0_25": 0,
                }
            row["quick"] = compact(quick)
            if int(quick.get("right", 0)) > best["right"]:
                best = {
                    "right": int(quick.get("right", 0)), "name": source["name"],
                    "screen_total": len(quick_cases), "cost": cost,
                }
            if not exact(quick):
                row["classification"] = "REJECT_QUICK_KNOWN"
                rows.append(row)
                continue
            known = SUPPORT.evaluate_four(data, cases)
            row["known_four"] = {name: compact(value) for name, value in known.items()}
            if not four_exact(known):
                row["classification"] = "REJECT_FULL_KNOWN"
                rows.append(row)
                continue
            official = SUPPORT.official_profile(task, source["_model"], source["name"])
            row["official_profile"] = official
            if official is None or int(official["cost"]) != cost or not official["correct"]:
                row["classification"] = "REJECT_OFFICIAL_PROFILE"
                rows.append(row)
                continue
            fresh_runs = []
            for seed in (296_000_000 + task, 296_100_000 + task):
                fresh, generation = SUPPORT.fresh_cases(task, seed, task_map)
                four = SUPPORT.evaluate_four(data, fresh)
                fresh_runs.append({
                    "seed": seed, "generation": generation,
                    "four": {name: compact(value) for name, value in four.items()},
                    "exact": four_exact(four),
                })
            row["fresh"] = fresh_runs
            clean = all(
                item["exact"] and item["generation"]["accepted"] == FRESH_PER_SEED
                and item["generation"]["generation_errors"] == 0
                and item["generation"]["conversion_skips"] == 0
                for item in fresh_runs
            )
            if not clean:
                row["classification"] = "REJECT_FRESH"
                rows.append(row)
                continue
            row["classification"] = "PASS_SAFE_STRICT_LOWER"
            rows.append(row)
            accepted.append((cost, digest, source, row))
        if accepted:
            cost, digest, source, row = sorted(accepted, key=lambda item: (item[0], item[1]))[0]
            path = CANDIDATES / f"task{task:03d}_{source['name']}_cost{cost}.onnx"
            path.write_bytes(source["_data"])
            row["candidate_path"] = str(path.relative_to(ROOT))
            finalists.append({
                "task": task, "cost": cost, "authority_cost": authority_cost,
                "sha256": digest, "path": str(path.relative_to(ROOT)), "name": source["name"],
            })
        tasks.append({
            "task": task, "hash": task_map[f"{task:03d}"], "authority_cost": authority_cost,
            "authority_sha256": sha256(base_data), "known_counts": known_counts,
            "variant_count": len(variants), "best_quick": best,
            "decision": "PASS_SAFE_STRICT_LOWER" if accepted else "NO_SAFE_STRICT_LOWER",
        })
        print(json.dumps({"task": task, "variants": len(variants), "best": best,
                          "accepted": len(accepted)}), flush=True)
    payload = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": AUTHORITY_SHA256,
                      "lb": 8011.05},
        "scope": {"cost_min": 11, "cost_max": 25, "task_count": len(costs),
                  "tasks": sorted(costs), "fresh_per_seed": FRESH_PER_SEED},
        "policy": {"strict_lower_actual": True, "four_ort_configs": True,
                   "no_nonfinite_shape_cloak_sparse_banned_or_giant": True,
                   "root_or_stage_written": False},
        "task_results": tasks, "candidate_rows": rows, "reported_candidates": finalists,
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "elapsed_seconds": time.monotonic() - started,
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"reported": finalists, "elapsed": payload["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

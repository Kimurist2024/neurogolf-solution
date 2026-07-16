#!/usr/bin/env python3
"""Slice free latent dimensions in single-Einsum cost-26..50 models."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8011.05.zip"
AUTHORITY_SHA256 = "ad96519a07bcae72017bdb92855b361374f7a34c9f40db237cb2e854615bcd56"
TASK_MAP = ROOT / "docs/golf/task_hash_map.json"
SCORES = ROOT / "all_scores.csv"
SUPPORT_PATH = ROOT / "scripts/golf/agent_where_ablation_scan_287/scan.py"
OUTPUT = HERE / "latent_prune.json"
CANDIDATES = HERE / "candidates"
FRESH_PER_SEED = 2_000


def load_support() -> Any:
    spec = importlib.util.spec_from_file_location("cost297_latent_support", SUPPORT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SUPPORT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.FRESH_PER_SEED = FRESH_PER_SEED
    return module


SUPPORT = load_support()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def costs() -> dict[int, int]:
    result = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split(",")
        result[int(fields[1][4:])] = int(fields[3])
    return result


def exact(row: dict[str, Any]) -> bool:
    return bool(
        row.get("right") == row.get("total") and row.get("wrong") == 0
        and row.get("errors") == 0 and row.get("nonfinite_cases") == 0
        and row.get("nonfinite_elements") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and not row.get("session_error")
    )


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "total", "right", "wrong", "accuracy", "errors", "nonfinite_cases",
            "nonfinite_elements", "runtime_shape_mismatches",
            "small_positive_elements_0_to_0_25", "minimum_positive",
            "maximum_nonpositive", "sign_mismatch_cases_vs_disable_threads1",
            "sign_mismatch_cells_vs_disable_threads1", "sign_sha256", "raw_sha256",
            "first_wrong", "first_error", "optimization", "threads",
        )
        if key in row
    }


def equation(node: onnx.NodeProto) -> str:
    for attribute in node.attribute:
        if attribute.name == "equation":
            value = helper.get_attribute_value(attribute)
            return value.decode() if isinstance(value, bytes) else str(value)
    raise ValueError("Einsum equation missing")


class DSU:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def add(self, item: tuple[str, int]) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: tuple[str, int]) -> tuple[str, int]:
        root = self.parent[item]
        if root != item:
            self.parent[item] = self.find(root)
        return self.parent[item]

    def union(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def free_components(model: onnx.ModelProto) -> list[dict[str, Any]]:
    node = model.graph.node[0]
    lhs, rhs = equation(node).split("->", 1)
    terms = lhs.split(",")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    dsu = DSU()
    label_vertices: dict[str, list[tuple[str, int]]] = defaultdict(list)
    protected_labels = set(rhs)
    for name, term in zip(node.input, terms):
        if name not in initializers:
            protected_labels.update(term)
            continue
        array = initializers[name]
        if len(term) != array.ndim:
            return []
        for axis, label in enumerate(term):
            vertex = (name, axis)
            dsu.add(vertex)
            label_vertices[label].append(vertex)
    for vertices in label_vertices.values():
        for vertex in vertices[1:]:
            dsu.union(vertices[0], vertex)
    components: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    labels_by_root: dict[tuple[str, int], set[str]] = defaultdict(set)
    for label, vertices in label_vertices.items():
        for vertex in vertices:
            root = dsu.find(vertex)
            components[root].add(vertex)
            labels_by_root[root].add(label)
    result = []
    for root, vertices in components.items():
        dims = {int(initializers[name].shape[axis]) for name, axis in vertices}
        labels = labels_by_root[root]
        if len(dims) != 1 or labels & protected_labels:
            continue
        dim = next(iter(dims))
        if not 2 <= dim <= 6:
            continue
        result.append({
            "vertices": sorted(vertices),
            "labels": sorted(labels),
            "dim": dim,
        })
    return result


def sliced(model: onnx.ModelProto, component: dict[str, Any], keep: tuple[int, ...]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    arrays = {item.name: numpy_helper.to_array(item) for item in result.graph.initializer}
    axes_by_name: dict[str, list[int]] = defaultdict(list)
    for name, axis in component["vertices"]:
        axes_by_name[name].append(axis)
    replacements = {}
    for name, axes in axes_by_name.items():
        array = arrays[name]
        for axis in sorted(axes):
            array = np.take(array, keep, axis=axis)
        replacements[name] = numpy_helper.from_array(np.asarray(array), name)
    for index, item in enumerate(result.graph.initializer):
        if item.name in replacements:
            result.graph.initializer[index].CopyFrom(replacements[item.name])
    return result


def main() -> int:
    started = time.monotonic()
    if sha256(AUTHORITY.read_bytes()) != AUTHORITY_SHA256:
        raise RuntimeError("authority SHA mismatch")
    all_costs = costs()
    tasks = sorted(task for task, cost in all_costs.items() if 26 <= cost <= 50)
    task_map = json.loads(TASK_MAP.read_text(encoding="utf-8"))
    CANDIDATES.mkdir(parents=True, exist_ok=True)
    rows = []
    winners = []
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in tasks:
            data = archive.read(f"task{task:03d}.onnx")
            model = onnx.load_from_string(data)
            if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
                continue
            components = free_components(model)
            cases, known_counts = SUPPORT.known_cases(task)
            task_row: dict[str, Any] = {
                "task": task,
                "authority_cost": all_costs[task],
                "authority_sha256": sha256(data),
                "equation": equation(model.graph.node[0]),
                "components": components,
                "known_counts": known_counts,
                "attempts": [],
            }
            for component_index, component in enumerate(components):
                dim = component["dim"]
                selections = []
                for target_dim in sorted({1, dim - 1}):
                    selections.extend(itertools.combinations(range(dim), target_dim))
                for keep in selections:
                    candidate = sliced(model, component, keep)
                    candidate_data = candidate.SerializeToString()
                    profile = SUPPORT.official_profile(task, candidate, f"latent297_{component_index}_{keep}")
                    attempt: dict[str, Any] = {
                        "component_index": component_index,
                        "component": component,
                        "keep": list(keep),
                        "sha256": sha256(candidate_data),
                        "official_profile": profile,
                    }
                    if profile is None or int(profile["cost"]) >= all_costs[task]:
                        attempt["decision"] = "REJECT_NOT_STRICT_LOWER_OR_UNSCORABLE"
                        task_row["attempts"].append(attempt)
                        continue
                    try:
                        session = SUPPORT.make_session(candidate_data, True, 1)
                        known, _ = SUPPORT.evaluate_config(session, cases, None)
                    except Exception as exc:  # noqa: BLE001
                        known = {"total": len(cases), "right": 0, "wrong": 0, "errors": len(cases), "session_error": f"{type(exc).__name__}: {exc}"}
                    attempt["known"] = compact(known)
                    if not exact(known):
                        attempt["decision"] = "REJECT_KNOWN"
                        task_row["attempts"].append(attempt)
                        continue
                    four = SUPPORT.evaluate_four(candidate_data, cases)
                    attempt["known_four"] = {name: compact(value) for name, value in four.items()}
                    if not all(exact(value) for value in four.values()):
                        attempt["decision"] = "REJECT_KNOWN_FOUR"
                        task_row["attempts"].append(attempt)
                        continue
                    fresh_runs = []
                    for seed in (297_200_000 + task, 297_300_000 + task):
                        fresh, generation = SUPPORT.fresh_cases(task, seed, task_map)
                        fresh_four = SUPPORT.evaluate_four(candidate_data, fresh)
                        passed = bool(
                            generation["generation_errors"] == 0
                            and generation["conversion_skips"] == 0
                            and all(exact(value) for value in fresh_four.values())
                        )
                        fresh_runs.append({
                            "seed": seed,
                            "generation": generation,
                            "four": {name: compact(value) for name, value in fresh_four.items()},
                            "exact": passed,
                        })
                    attempt["fresh"] = fresh_runs
                    if not all(value["exact"] for value in fresh_runs):
                        attempt["decision"] = "REJECT_FRESH"
                        task_row["attempts"].append(attempt)
                        continue
                    attempt["decision"] = "ACCEPT"
                    path = CANDIDATES / f"task{task:03d}_latent_c{component_index}_{''.join(map(str, keep))}_cost{profile['cost']}.onnx"
                    path.write_bytes(candidate_data)
                    attempt["candidate_path"] = str(path.relative_to(ROOT))
                    winners.append({"task": task, **attempt})
                    task_row["attempts"].append(attempt)
            rows.append(task_row)
            best = max((attempt.get("known", {}).get("accuracy", 0.0) for attempt in task_row["attempts"]), default=0.0)
            print(json.dumps({"task": task, "components": len(components), "attempts": len(task_row["attempts"]), "best_known_accuracy": best}), flush=True)
    result = {
        "authority": {"path": str(AUTHORITY.relative_to(ROOT)), "sha256": AUTHORITY_SHA256, "lb": 8011.05},
        "task_rows": rows,
        "winners": winners,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "winners": len(winners)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

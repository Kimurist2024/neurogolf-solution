#!/usr/bin/env python3
"""Delete one internal tensor-network bond component from compact Einsum nets."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUTHORITY = ROOT / "submission_base_8012.23.zip"
TASKS = (132, 199, 212)


def import_worker():
    path = ROOT / "scripts/golf/restart8012_cost501_1000_3w_408/worker.py"
    spec = importlib.util.spec_from_file_location("einsum_rank_support", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.THRESHOLD = 1.0
    module.FRESH_PER_SEED = 2_000
    module.SUPPORT.FRESH_PER_SEED = 2_000
    return module


def equation(node: onnx.NodeProto) -> tuple[list[str], str]:
    value = next(
        helper.get_attribute_value(attr)
        for attr in node.attribute if attr.name == "equation"
    )
    lhs, rhs = value.decode().split("->")
    return lhs.split(","), rhs


def deletable_components(model: onnx.ModelProto) -> list[tuple[tuple[str, ...], int]]:
    node = model.graph.node[0]
    terms, output = equation(node)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    input_labels = {
        label for name, term in zip(node.input, terms)
        if name == "input" for label in term
    }
    labels = set("".join(terms))
    parent = {label: label for label in labels}

    def find(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    # Reusing one initializer with different index names ties those logical
    # bonds to the same physical axis.  Reduce the whole tied component.
    for name, array in arrays.items():
        uses = [term for item, term in zip(node.input, terms) if item == name]
        for axis in range(array.ndim):
            names = [term[axis] for term in uses]
            for label in names[1:]:
                union(names[0], label)

    groups: dict[str, set[str]] = {}
    for label in labels:
        groups.setdefault(find(label), set()).add(label)
    result: list[tuple[tuple[str, ...], int]] = []
    forbidden = set(output) | input_labels
    for group in groups.values():
        if group & forbidden:
            continue
        dimensions = {
            int(array.shape[axis])
            for name, array in arrays.items()
            for term in [term for item, term in zip(node.input, terms) if item == name]
            for axis, label in enumerate(term)
            if label in group
        }
        if len(dimensions) == 1 and next(iter(dimensions)) > 1:
            result.append((tuple(sorted(group)), next(iter(dimensions))))
    return sorted(result)


def make_variant(base: onnx.ModelProto, labels: tuple[str, ...], removed: int) -> onnx.ModelProto:
    model = copy.deepcopy(base)
    node = model.graph.node[0]
    terms, _ = equation(node)
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for item in model.graph.initializer:
        uses = [term for name, term in zip(node.input, terms) if name == item.name]
        array = arrays[item.name]
        axes = tuple(
            axis for axis in range(array.ndim)
            if uses and any(term[axis] in labels for term in uses)
        )
        if not axes:
            continue
        for axis in axes:
            array = np.delete(array, removed, axis=axis)
        item.CopyFrom(numpy_helper.from_array(array, item.name))
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def exact(row: dict[str, object]) -> bool:
    return bool(
        row.get("right") == row.get("total")
        and row.get("wrong") == 0
        and row.get("errors") == 0
        and row.get("nonfinite_cases") == 0
        and row.get("runtime_shape_mismatches") == 0
        and row.get("small_positive_elements_0_to_0_25") == 0
        and row.get("early_reject_reason") is None
    )


def main() -> int:
    support = import_worker()
    task_map = json.loads((ROOT / "docs/golf/task_hash_map.json").read_text())
    report: dict[str, object] = {
        "authority": str(AUTHORITY.relative_to(ROOT)),
        "method": "one-component deletion on initializer-only internal Einsum bonds",
        "gold_required": True,
        "fresh_required": 2_000,
        "tasks": [],
        "finalists": [],
        "counters": {},
    }
    counters: Counter[str] = Counter()
    with zipfile.ZipFile(AUTHORITY) as archive:
        for task in TASKS:
            base_data = archive.read(f"task{task:03d}.onnx")
            base = onnx.load_model_from_string(base_data)
            cases, counts = support.SUPPORT.known_cases(task)
            base_profile = support.POLICY.fast_profile(support.SUPPORT, task, base, cases[0])
            components = deletable_components(base)
            task_row = {
                "task": task,
                "authority_sha256": hashlib.sha256(base_data).hexdigest(),
                "authority_cost": int(base_profile["cost"]),
                "known_counts": counts,
                "components": [
                    {"labels": labels, "dimension": dimension}
                    for labels, dimension in components
                ],
                "attempts": [],
            }
            for component_index, (labels, dimension) in enumerate(components):
                label_key = "".join(labels)
                for removed in range(dimension):
                    counters["variants"] += 1
                    try:
                        model = make_variant(base, labels, removed)
                    except Exception as exc:
                        counters["build_reject"] += 1
                        task_row["attempts"].append({
                            "component": component_index, "labels": labels, "removed": removed,
                            "reject": f"build:{type(exc).__name__}:{exc}",
                        })
                        continue
                    data = model.SerializeToString()
                    profile = support.POLICY.fast_profile(support.SUPPORT, task, model, cases[0])
                    if profile is None or int(profile["cost"]) >= int(base_profile["cost"]):
                        counters["cost_reject"] += 1
                        continue
                    known = support.failfast_known(data, cases)
                    attempt = {
                        "component": component_index, "labels": labels, "removed": removed,
                        "candidate_cost": int(profile["cost"]),
                        "gain": math.log(int(base_profile["cost"]) / int(profile["cost"])),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "known": support.compact_runtime(known),
                    }
                    task_row["attempts"].append(attempt)
                    if not exact(known):
                        counters["known_reject"] += 1
                        continue
                    counters["known_exact"] += 1
                    fresh_cases, generation = support.SUPPORT.fresh_cases(
                        task, 425_000_000 + task * 100
                        + sum(ord(label) for label in labels) + removed, task_map
                    )
                    fresh_raw = support.SUPPORT.evaluate_four(data, fresh_cases)
                    fresh = {name: support.compact_runtime(value)
                             for name, value in fresh_raw.items()}
                    fresh_pass = len(fresh_cases) >= 2_000 and all(exact(value) for value in fresh_raw.values())
                    attempt["fresh_generation"] = generation
                    attempt["fresh"] = fresh
                    attempt["fresh_pass"] = fresh_pass
                    if not fresh_pass:
                        counters["fresh_reject"] += 1
                        continue
                    structure = support.POLICY.structure_audit(support.SUPPORT, task, model, data)
                    attempt["structure"] = structure
                    if not structure["pass"]:
                        counters["structure_reject"] += 1
                        continue
                    digest = attempt["sha256"]
                    path = HERE / "candidates" / (
                        f"task{task:03d}_einsum_drop_c{component_index}_{label_key}{removed}_cost{profile['cost']}_{digest[:12]}.onnx"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    check = subprocess.run(
                        [sys.executable, str(ROOT / "scripts/golf/try_candidate.py"),
                         "--task", str(task), "--onnx", str(path)],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    output = check.stdout + check.stderr
                    attempt["official_gold"] = check.returncode == 0 and "PASS gold:" in output
                    attempt["official_gold_output"] = output[-4000:]
                    if not attempt["official_gold"]:
                        counters["official_gold_reject"] += 1
                        path.unlink(missing_ok=True)
                        continue
                    attempt["path"] = str(path.relative_to(ROOT))
                    report["finalists"].append({"task": task, **attempt})
                    counters["finalists"] += 1
            report["tasks"].append(task_row)
            (HERE / "einsum_rank_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
    report["counters"] = dict(counters)
    (HERE / "einsum_rank_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counters": dict(counters), "finalists": report["finalists"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

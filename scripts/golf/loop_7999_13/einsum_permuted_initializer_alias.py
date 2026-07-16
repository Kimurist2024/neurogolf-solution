#!/usr/bin/env python3
"""Alias exactly permuted initializers used only by Einsum nodes.

If ``B == transpose(A, permutation)``, every Einsum use of ``B`` can consume
``A`` directly by applying the inverse permutation to that operand's equation
term.  This removes all parameters of ``B`` without allocating an intermediate
tensor or changing the contraction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum equation missing")


def set_equation(node: onnx.NodeProto, terms: list[str]) -> None:
    old = equation(node)
    suffix = "->" + old.split("->", 1)[1] if "->" in old else ""
    value = ",".join(terms) + suffix
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum equation missing")


def uses(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                locations[name].append((node_index, input_index))
    return counts, locations


def find_permutation(source: np.ndarray, target: np.ndarray) -> tuple[int, ...] | None:
    if source.dtype != target.dtype or source.ndim != target.ndim or source.ndim < 2:
        return None
    if source.size != target.size or source.ndim > 7:
        return None
    for permutation in itertools.permutations(range(source.ndim)):
        if tuple(source.shape[index] for index in permutation) != target.shape:
            continue
        if np.array_equal(source.transpose(permutation), target, equal_nan=True):
            return tuple(int(index) for index in permutation)
    return None


def candidate_plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: np.asarray(numpy_helper.to_array(item)) for name, item in initializers.items()}
    counts, locations = uses(model.graph)
    eligible_targets: list[str] = []
    for name, array in arrays.items():
        if array.ndim < 2 or counts[name] == 0:
            continue
        ok = True
        for node_index, input_index in locations[name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum":
                ok = False
                break
            terms = equation(node).split("->", 1)[0].split(",")
            if len(terms) != len(node.input) or len(terms[input_index]) != array.ndim:
                ok = False
                break
            if "..." in terms[input_index]:
                ok = False
                break
        if ok:
            eligible_targets.append(name)

    plans: list[dict[str, object]] = []
    for target_name in eligible_targets:
        target = arrays[target_name]
        target_elements = int(target.size)
        best: dict[str, object] | None = None
        for source_name, source in arrays.items():
            if source_name == target_name:
                continue
            permutation = find_permutation(source, target)
            if permutation is None:
                continue
            # Prefer an already-shared or larger canonical tensor, then name,
            # so two aliases do not accidentally form an unstable cycle.
            rank = (-counts[source_name], -int(source.size), source_name)
            row = {
                "source": source_name,
                "target": target_name,
                "permutation": list(permutation),
                "removed_elements": target_elements,
                "rank": rank,
            }
            if best is None or row["rank"] < best["rank"]:
                best = row
        if best is not None:
            best.pop("rank", None)
            plans.append(best)
    return plans


def build(model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    source = str(plan["source"])
    target = str(plan["target"])
    permutation = tuple(int(index) for index in plan["permutation"])
    _, locations = uses(result.graph)
    for node_index, input_index in locations[target]:
        node = result.graph.node[node_index]
        terms = equation(node).split("->", 1)[0].split(",")
        old_term = terms[input_index]
        new_term = [""] * len(old_term)
        for target_axis, source_axis in enumerate(permutation):
            new_term[source_axis] = old_term[target_axis]
        if any(not label for label in new_term):
            raise RuntimeError("invalid inverse permutation")
        node.input[input_index] = source
        terms[input_index] = "".join(new_term)
        set_equation(node, terms)
    kept = [item for item in result.graph.initializer if item.name != target]
    del result.graph.initializer[:]
    result.graph.initializer.extend(kept)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True, data_prop=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            with tempfile.TemporaryDirectory(prefix=f"epi_base_{task:03d}_") as tmp:
                base_path = Path(tmp) / member
                onnx.save(model, base_path)
                try:
                    base_memory, base_params, base_cost = cost_of(str(base_path))
                except Exception as exc:
                    errors.append({"task": task, "stage": "base_cost", "error": repr(exc)})
                    continue
            for ordinal, plan in enumerate(candidate_plans(model), 1):
                try:
                    candidate = build(model, plan)
                    with tempfile.TemporaryDirectory(prefix=f"epi_cand_{task:03d}_") as tmp:
                        probe = Path(tmp) / member
                        onnx.save(candidate, probe)
                        memory, params, cost = cost_of(str(probe))
                    if cost < 0 or cost >= base_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, path)
                    rows.append({
                        "task": task,
                        "path": str(path),
                        **plan,
                        "baseline_memory": int(base_memory),
                        "baseline_params": int(base_params),
                        "baseline_cost": int(base_cost),
                        "candidate_memory": int(memory),
                        "candidate_params": int(params),
                        "candidate_cost": int(cost),
                        "projected_gain": math.log(base_cost / cost),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    })
                except Exception as exc:
                    errors.append({"task": task, "stage": "build", **plan, "error": repr(exc)})
    payload = {"source_zip": str(args.zip), "rows": rows, "errors": errors}
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "error_count": len(errors),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

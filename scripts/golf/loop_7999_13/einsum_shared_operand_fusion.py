#!/usr/bin/env python3
"""Fuse every use of a shared constant Einsum operand into unique constants.

If a constant ``S[j]`` appears in several Einsum nodes, and every occurrence
can be multiplied into a node-local constant ``T[...,j,...]``, all occurrences
of ``S`` can be removed.  The node-local constants keep their original sizes,
so the exact parameter saving is the size of ``S``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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


def get_equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum equation missing")


def set_equation(node: onnx.NodeProto, terms: list[str], original: str) -> None:
    rhs = original.split("->", 1)[1] if "->" in original else None
    value = ",".join(terms) + (("->" + rhs) if rhs is not None else "")
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum equation missing")


def terms(node: onnx.NodeProto) -> list[str]:
    return get_equation(node).split("->", 1)[0].split(",")


def uses(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                locations[name].append((node_index, input_index))
    return counts, locations


def align_product(
    target: np.ndarray,
    target_term: str,
    source: np.ndarray,
    source_term: str,
) -> np.ndarray:
    source_order = [label for label in target_term if label in source_term]
    transpose = [source_term.index(label) for label in source_order]
    aligned = source.transpose(transpose) if transpose else source
    shape = [
        target.shape[index] if label in source_term else 1
        for index, label in enumerate(target_term)
    ]
    return np.multiply(target, aligned.reshape(shape), dtype=target.dtype)


def plan_source(
    model: onnx.ModelProto,
    source_name: str,
    counts: Counter[str],
    locations: dict[str, list[tuple[int, int]]],
) -> list[dict[str, object]] | None:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    source_array = numpy_helper.to_array(initializers[source_name])
    selected_targets: set[str] = set()
    plan: list[dict[str, object]] = []
    for node_index, source_index in locations[source_name]:
        node = graph.node[node_index]
        if node.op_type != "Einsum":
            return None
        node_terms = terms(node)
        if len(node_terms) != len(node.input):
            return None
        source_term = node_terms[source_index]
        if (
            "..." in source_term
            or len(source_term) != source_array.ndim
            or len(set(source_term)) != len(source_term)
        ):
            return None
        options: list[tuple[int, int, str, np.ndarray]] = []
        for target_index, target_name in enumerate(node.input):
            if target_index == source_index or target_name not in initializers:
                continue
            if target_name in selected_targets or counts[target_name] != 1:
                continue
            target_term = node_terms[target_index]
            target_array = numpy_helper.to_array(initializers[target_name])
            if (
                target_array.dtype != source_array.dtype
                or "..." in target_term
                or len(target_term) != target_array.ndim
                or len(set(target_term)) != len(target_term)
                or not set(source_term).issubset(set(target_term))
            ):
                continue
            target_dims = {
                label: int(size) for label, size in zip(target_term, target_array.shape)
            }
            if any(
                target_dims[label] != int(size)
                for label, size in zip(source_term, source_array.shape)
            ):
                continue
            product = align_product(target_array, target_term, source_array, source_term)
            options.append((int(product.size), target_index, target_name, product))
        if not options:
            return None
        _, target_index, target_name, product = min(options, key=lambda item: item[0])
        selected_targets.add(target_name)
        plan.append({
            "node_index": node_index,
            "source_index": source_index,
            "source_term": source_term,
            "target_index": target_index,
            "target_name": target_name,
            "target_term": node_terms[target_index],
            "product": product,
        })
    return plan


def build_candidate(
    model: onnx.ModelProto, source_name: str, plan: list[dict[str, object]]
) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    result = copy.deepcopy(model)
    graph = result.graph
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    evidence: list[dict[str, object]] = []
    existing = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {tensor.name for tensor in graph.initializer}
        | {name for node in graph.node for name in (*node.input, *node.output) if name}
    )
    for ordinal, item in enumerate(plan):
        fused_name = f"__esof_{source_name}_{ordinal}"
        while fused_name in existing:
            fused_name += "_"
        existing.add(fused_name)
        product = item["product"]
        assert isinstance(product, np.ndarray)
        graph.initializer.append(numpy_helper.from_array(product, fused_name))
        copied = {key: value for key, value in item.items() if key != "product"}
        copied["fused_name"] = fused_name
        grouped[int(item["node_index"])].append(copied)
        evidence.append(copied)
    for node_index, node_plan in grouped.items():
        node = graph.node[node_index]
        original = get_equation(node)
        node_terms = terms(node)
        replacements = {
            int(item["target_index"]): str(item["fused_name"]) for item in node_plan
        }
        removals = {int(item["source_index"]) for item in node_plan}
        new_inputs = [
            replacements.get(index, name)
            for index, name in enumerate(node.input)
            if index not in removals
        ]
        new_terms = [term for index, term in enumerate(node_terms) if index not in removals]
        del node.input[:]
        node.input.extend(new_inputs)
        set_equation(node, new_terms, original)
    protected = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {name for node in graph.node for name in node.input if name}
    )
    kept = [tensor for tensor in graph.initializer if tensor.name in protected]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    live = protected | {name for node in graph.node for name in node.output if name}
    kept_vi = [value for value in graph.value_info if value.name in live]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result, evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_costs = json.loads(args.base_costs.read_text())
    costs = raw_costs.get("costs") or {
        str(row["task"]): row["cost"] for row in raw_costs["ranked"]
    }
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            counts, locations = uses(model.graph)
            init_map = {tensor.name: tensor for tensor in model.graph.initializer}
            ordinal = 0
            for source_name, tensor in init_map.items():
                if counts[source_name] < 2:
                    continue
                plan = plan_source(model, source_name, counts, locations)
                if not plan:
                    continue
                ordinal += 1
                try:
                    candidate, evidence = build_candidate(model, source_name, plan)
                    with tempfile.TemporaryDirectory(prefix=f"esof_{task:03d}_") as tmp:
                        probe = Path(tmp) / member
                        onnx.save(candidate, probe)
                        memory, params, cost = cost_of(str(probe))
                    baseline_cost = int(costs[str(task)])
                    if cost < 0 or cost >= baseline_cost:
                        continue
                    output = args.out_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, output)
                    rows.append({
                        "task": task,
                        "path": str(output.relative_to(ROOT)),
                        "source": source_name,
                        "source_elements": int(np.prod(tensor.dims) if tensor.dims else 1),
                        "source_uses": counts[source_name],
                        "baseline_cost": baseline_cost,
                        "candidate_cost": int(cost),
                        "candidate_memory": int(memory),
                        "candidate_params": int(params),
                        "projected_gain": math.log(baseline_cost / cost),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "fusions": evidence,
                    })
                except Exception as exc:
                    errors.append({"task": task, "source": source_name, "error": repr(exc)})
    payload = {
        "baseline": str(args.baseline),
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "rows": rows,
        "errors": errors,
    }
    manifest = args.out_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "candidate_count": len(rows),
        "projected_gain": payload["projected_gain"],
        "error_count": len(errors),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

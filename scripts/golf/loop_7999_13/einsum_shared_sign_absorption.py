#!/usr/bin/env python3
"""Remove a unique signed-power-of-two Einsum factor through exact absorption.

For a contraction containing ``T[..., i, ...] * S[i]``, with ``S`` made only
of signed powers of two, absorb ``S`` into the shared initializer ``T``.  Every
other use of ``T`` is compensated by absorbing ``1/S`` into a unique constant
operand at that use.  Binary exponent shifts are exact for the finite values
retained by the builder, so the rewrite removes all parameters of ``S`` without
changing the contraction.
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


def get_terms(node: onnx.NodeProto) -> list[str]:
    return get_equation(node).split("->", 1)[0].split(",")


def set_terms(node: onnx.NodeProto, terms: list[str]) -> None:
    original = get_equation(node)
    rhs = original.split("->", 1)[1] if "->" in original else None
    value = ",".join(terms) + (("->" + rhs) if rhs is not None else "")
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum equation missing")


def use_locations(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    locations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                locations[name].append((node_index, input_index))
    return counts, locations


def align(array: np.ndarray, source_term: str, target: np.ndarray, target_term: str) -> np.ndarray:
    order = [source_term.index(label) for label in target_term if label in source_term]
    ordered = array.transpose(order) if order else array
    shape = [target.shape[index] if label in source_term else 1 for index, label in enumerate(target_term)]
    return ordered.reshape(shape)


def signed_power_of_two(array: np.ndarray) -> bool:
    if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
        return False
    magnitude = np.abs(array)
    if np.any(magnitude == 0):
        return False
    mantissa, _ = np.frexp(magnitude)
    return bool(np.all(mantissa == 0.5))


def make_plan(
    model: onnx.ModelProto,
    source_name: str,
    source_node_index: int,
    source_input_index: int,
    shared_name: str,
    shared_input_index: int,
    counts: Counter[str],
    locations: dict[str, list[tuple[int, int]]],
    allow_any_scale: bool,
) -> dict[str, object] | None:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    source = numpy_helper.to_array(initializers[source_name])
    shared = numpy_helper.to_array(initializers[shared_name])
    valid_arbitrary_scale = bool(
        np.issubdtype(source.dtype, np.floating)
        and np.all(np.isfinite(source))
        and np.all(source != 0)
    )
    if source.dtype != shared.dtype or not (
        signed_power_of_two(source) or (allow_any_scale and valid_arbitrary_scale)
    ):
        return None
    source_node = graph.node[source_node_index]
    source_terms = get_terms(source_node)
    source_term = source_terms[source_input_index]
    shared_term = source_terms[shared_input_index]
    if (
        "..." in source_term
        or "..." in shared_term
        or len(source_term) != source.ndim
        or len(shared_term) != shared.ndim
        or len(set(source_term)) != len(source_term)
        or len(set(shared_term)) != len(shared_term)
        or not set(source_term).issubset(set(shared_term))
    ):
        return None
    shared_dims = {label: int(size) for label, size in zip(shared_term, shared.shape)}
    if any(shared_dims[label] != int(size) for label, size in zip(source_term, source.shape)):
        return None
    source_axes = [shared_term.index(label) for label in source_term]
    compensations: list[dict[str, object]] = []
    reserved = {source_name, shared_name}
    for node_index, input_index in locations[shared_name]:
        if node_index == source_node_index and input_index == shared_input_index:
            continue
        node = graph.node[node_index]
        if node.op_type != "Einsum":
            return None
        terms = get_terms(node)
        if len(terms) != len(node.input):
            return None
        occurrence_term = terms[input_index]
        if len(occurrence_term) != shared.ndim or "..." in occurrence_term:
            return None
        needed_term = "".join(occurrence_term[axis] for axis in source_axes)
        options: list[tuple[int, int, str, np.ndarray]] = []
        for compensation_index, compensation_name in enumerate(node.input):
            if compensation_index == input_index or compensation_name in reserved:
                continue
            tensor = initializers.get(compensation_name)
            if tensor is None or counts[compensation_name] != 1:
                continue
            compensation = numpy_helper.to_array(tensor)
            compensation_term = terms[compensation_index]
            if (
                compensation.dtype != source.dtype
                or "..." in compensation_term
                or len(compensation_term) != compensation.ndim
                or len(set(compensation_term)) != len(compensation_term)
                or not set(needed_term).issubset(set(compensation_term))
            ):
                continue
            dims = {label: int(size) for label, size in zip(compensation_term, compensation.shape)}
            if any(dims[label] != int(size) for label, size in zip(needed_term, source.shape)):
                continue
            inverse = np.reciprocal(source, dtype=source.dtype)
            product = np.multiply(
                compensation,
                align(inverse, needed_term, compensation, compensation_term),
                dtype=compensation.dtype,
            )
            if not np.all(np.isfinite(product)):
                continue
            options.append((int(product.size), compensation_index, compensation_name, product))
        if not options:
            return None
        _, compensation_index, compensation_name, product = min(options, key=lambda item: item[0])
        reserved.add(compensation_name)
        compensations.append(
            {
                "node_index": node_index,
                "shared_input_index": input_index,
                "shared_term": occurrence_term,
                "needed_term": needed_term,
                "compensation_input_index": compensation_index,
                "compensation_name": compensation_name,
                "product": product,
            }
        )
    shared_product = np.multiply(
        shared,
        align(source, source_term, shared, shared_term),
        dtype=shared.dtype,
    )
    if not np.all(np.isfinite(shared_product)):
        return None
    return {
        "source_name": source_name,
        "source_node_index": source_node_index,
        "source_input_index": source_input_index,
        "source_term": source_term,
        "shared_name": shared_name,
        "shared_input_index": shared_input_index,
        "shared_term": shared_term,
        "shared_product": shared_product,
        "compensations": compensations,
    }


def plans(model: onnx.ModelProto, allow_any_scale: bool = False) -> list[dict[str, object]]:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    counts, locations = use_locations(graph)
    result: list[dict[str, object]] = []
    for source_name in initializers:
        if counts[source_name] != 1:
            continue
        source_node_index, source_input_index = locations[source_name][0]
        node = graph.node[source_node_index]
        if node.op_type != "Einsum":
            continue
        for shared_input_index, shared_name in enumerate(node.input):
            if shared_name == source_name or shared_name not in initializers or counts[shared_name] < 2:
                continue
            plan = make_plan(
                model,
                source_name,
                source_node_index,
                source_input_index,
                shared_name,
                shared_input_index,
                counts,
                locations,
                allow_any_scale,
            )
            if plan is not None:
                result.append(plan)
    return result


def build(model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    graph = result.graph
    replacements = {
        str(plan["shared_name"]): plan["shared_product"],
        **{
            str(item["compensation_name"]): item["product"]
            for item in plan["compensations"]
        },
    }
    kept = []
    for tensor in graph.initializer:
        if tensor.name == plan["source_name"]:
            continue
        array = replacements.get(tensor.name)
        kept.append(numpy_helper.from_array(array, tensor.name) if isinstance(array, np.ndarray) else tensor)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    node = graph.node[int(plan["source_node_index"])]
    remove_index = int(plan["source_input_index"])
    node_terms = get_terms(node)
    new_inputs = [name for index, name in enumerate(node.input) if index != remove_index]
    new_terms = [term for index, term in enumerate(node_terms) if index != remove_index]
    del node.input[:]
    node.input.extend(new_inputs)
    set_terms(node, new_terms)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {
        "source_name": plan["source_name"],
        "source_term": plan["source_term"],
        "shared_name": plan["shared_name"],
        "shared_term": plan["shared_term"],
        "compensations": [
            {key: value for key, value in item.items() if key != "product"}
            for item in plan["compensations"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-any-scale", action="store_true")
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_costs = json.loads(args.base_costs.read_text())
    costs = raw_costs.get("costs") or {str(row["task"]): row["cost"] for row in raw_costs["ranked"]}
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            for ordinal, plan in enumerate(plans(model, args.allow_any_scale), 1):
                try:
                    candidate = build(model, plan)
                    with tempfile.TemporaryDirectory(prefix=f"essa_{task:03d}_") as tmp:
                        probe = Path(tmp) / member
                        onnx.save(candidate, probe)
                        memory, params, cost = cost_of(str(probe))
                    baseline_cost = int(costs[str(task)])
                    if cost < 0 or cost >= baseline_cost:
                        continue
                    output = args.out_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, output)
                    rows.append(
                        {
                            "task": task,
                            "path": str(output.relative_to(ROOT)),
                            "baseline_cost": baseline_cost,
                            "candidate_cost": int(cost),
                            "candidate_memory": int(memory),
                            "candidate_params": int(params),
                            "projected_gain": math.log(baseline_cost / cost),
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "rewrite": concise(plan),
                        }
                    )
                except Exception as exc:
                    errors.append({"task": task, "rewrite": concise(plan), "error": repr(exc)})
    payload = {
        "baseline": str(args.baseline),
        "allow_any_scale": args.allow_any_scale,
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "rows": rows,
        "errors": errors,
    }
    manifest = args.out_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "projected_gain": payload["projected_gain"], "error_count": len(errors), "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

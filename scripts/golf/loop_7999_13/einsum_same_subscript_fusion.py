#!/usr/bin/env python3
"""Fuse constant Einsum operands with equal or nested subscripts.

For one Einsum node, ``A[i,j] * B[j]`` can be replaced by the constant
``C[i,j] = A[i,j] * B[j]`` without introducing an intermediate tensor.
This pass emits one candidate per eligible operand pair and removes an
original initializer only when it becomes unused.  Runtime differential and
fresh verification remain mandatory adoption gates.
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
from collections import Counter
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
            return attr.s.decode() if isinstance(attr.s, bytes) else str(attr.s)
    raise ValueError("Einsum has no equation")


def input_terms(eq: str) -> list[str]:
    lhs = eq.split("->", 1)[0]
    return lhs.split(",") if lhs else []


def replace_equation(node: onnx.NodeProto, terms: list[str], original: str) -> None:
    rhs = original.split("->", 1)[1] if "->" in original else None
    value = ",".join(terms) + (("->" + rhs) if rhs is not None else "")
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum has no equation")


def count_uses(graph: onnx.GraphProto) -> Counter[str]:
    return Counter(name for node in graph.node for name in node.input if name)


def prune_unused_initializers(graph: onnx.GraphProto) -> list[str]:
    protected = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {name for node in graph.node for name in node.input if name}
    )
    removed = [tensor.name for tensor in graph.initializer if tensor.name not in protected]
    kept = [tensor for tensor in graph.initializer if tensor.name in protected]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    live = protected | {name for node in graph.node for name in node.output if name}
    kept_vi = [value for value in graph.value_info if value.name in live]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)
    return removed


def eligible_pairs(model: onnx.ModelProto) -> list[tuple[int, int, int, str]]:
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    uses = count_uses(model.graph)
    pairs: list[tuple[int, int, int, str]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        terms = input_terms(equation(node))
        if len(terms) != len(node.input):
            continue
        for first in range(len(node.input)):
            lname = node.input[first]
            if (
                lname not in initializers
                or uses[lname] != 1
                or "..." in terms[first]
            ):
                continue
            larr = numpy_helper.to_array(initializers[lname])
            for second in range(first + 1, len(node.input)):
                rname = node.input[second]
                if (
                    rname not in initializers
                    or uses[rname] != 1
                    or "..." in terms[second]
                ):
                    continue
                rarr = numpy_helper.to_array(initializers[rname])
                if larr.dtype != rarr.dtype:
                    continue
                lterm, rterm = terms[first], terms[second]
                if (
                    len(lterm) != larr.ndim
                    or len(rterm) != rarr.ndim
                    or len(set(lterm)) != len(lterm)
                    or len(set(rterm)) != len(rterm)
                ):
                    continue
                if set(rterm).issubset(set(lterm)):
                    target, source, target_term = first, second, lterm
                    tarr, sarr = larr, rarr
                elif set(lterm).issubset(set(rterm)):
                    target, source, target_term = second, first, rterm
                    tarr, sarr = rarr, larr
                else:
                    continue
                tdims = {label: int(size) for label, size in zip(target_term, tarr.shape)}
                source_term = terms[source]
                if any(tdims[label] != int(size) for label, size in zip(source_term, sarr.shape)):
                    continue
                pairs.append((node_index, target, source, target_term))
    return pairs


def fuse(
    model: onnx.ModelProto, node_index: int, target: int, source: int
) -> tuple[onnx.ModelProto, dict[str, object]]:
    result = copy.deepcopy(model)
    graph = result.graph
    node = graph.node[node_index]
    original_equation = equation(node)
    terms = input_terms(original_equation)
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    target_name, source_name = node.input[target], node.input[source]
    target_term, source_term = terms[target], terms[source]
    target_array = numpy_helper.to_array(initializers[target_name])
    source_array = numpy_helper.to_array(initializers[source_name])
    target_order_source_labels = [label for label in target_term if label in source_term]
    transpose = [source_term.index(label) for label in target_order_source_labels]
    aligned = source_array.transpose(transpose) if transpose else source_array
    aligned_shape = [
        target_array.shape[index] if label in source_term else 1
        for index, label in enumerate(target_term)
    ]
    aligned = aligned.reshape(aligned_shape)
    product = np.multiply(target_array, aligned, dtype=target_array.dtype)
    fused_name = f"__esf_{node_index}_{target}_{source}"
    suffix = 0
    existing = (
        {value.name for value in graph.input}
        | {value.name for value in graph.output}
        | {tensor.name for tensor in graph.initializer}
        | {name for n in graph.node for name in (*n.input, *n.output) if name}
    )
    while fused_name in existing:
        suffix += 1
        fused_name = f"__esf_{node_index}_{target}_{source}_{suffix}"
    graph.initializer.append(numpy_helper.from_array(product, fused_name))
    new_inputs = [
        fused_name if index == target else name
        for index, name in enumerate(node.input)
        if index != source
    ]
    del node.input[:]
    node.input.extend(new_inputs)
    terms = [term for index, term in enumerate(terms) if index != source]
    replace_equation(node, terms, original_equation)
    removed = prune_unused_initializers(graph)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result, {
        "node_index": node_index,
        "target_index": target,
        "source_index": source,
        "target_term": target_term,
        "source_term": source_term,
        "target_initializer": target_name,
        "source_initializer": source_name,
        "fused_initializer": fused_name,
        "elements": int(product.size),
        "removed_initializers": removed,
    }


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
    costs = raw_costs.get("costs")
    if costs is None:
        costs = {str(row["task"]): row["cost"] for row in raw_costs["ranked"]}
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            for ordinal, (node_index, target, source, term) in enumerate(eligible_pairs(model), 1):
                try:
                    candidate, change = fuse(model, node_index, target, source)
                    with tempfile.TemporaryDirectory(prefix=f"esf_{task:03d}_") as tmp:
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
                        "baseline_cost": baseline_cost,
                        "candidate_cost": int(cost),
                        "candidate_memory": int(memory),
                        "candidate_params": int(params),
                        "projected_gain": math.log(baseline_cost / cost),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "change": change,
                    })
                except Exception as exc:
                    errors.append({
                        "task": task,
                        "node_index": node_index,
                        "target": target,
                        "source": source,
                        "term": term,
                        "error": repr(exc),
                    })
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

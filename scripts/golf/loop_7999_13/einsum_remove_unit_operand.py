#!/usr/bin/env python3
"""Remove unique all-one Einsum operands whose labels are otherwise bound."""

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
            return attr.s.decode()
    raise ValueError("equation missing")


def set_equation(node: onnx.NodeProto, terms: list[str], original: str) -> None:
    rhs = original.split("->", 1)[1] if "->" in original else None
    value = ",".join(terms) + (("->" + rhs) if rhs is not None else "")
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("equation missing")


def opportunities(model: onnx.ModelProto) -> list[tuple[int, int, str]]:
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    uses = Counter(name for node in graph.node for name in node.input if name)
    rows: list[tuple[int, int, str]] = []
    for node_index, node in enumerate(graph.node):
        if node.op_type != "Einsum":
            continue
        eq = equation(node)
        lhs = eq.split("->", 1)[0]
        terms = lhs.split(",")
        if len(terms) != len(node.input) or "..." in lhs:
            continue
        for input_index, name in enumerate(node.input):
            tensor = initializers.get(name)
            if tensor is None or uses[name] != 1:
                continue
            array = numpy_helper.to_array(tensor)
            if not np.all(array == 1):
                continue
            other_labels = set("".join(term for i, term in enumerate(terms) if i != input_index))
            if not set(terms[input_index]).issubset(other_labels):
                continue
            rows.append((node_index, input_index, name))
    return rows


def build(model: onnx.ModelProto, node_index: int, input_index: int) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    graph = result.graph
    node = graph.node[node_index]
    original = equation(node)
    terms = original.split("->", 1)[0].split(",")
    inputs = [name for index, name in enumerate(node.input) if index != input_index]
    terms = [term for index, term in enumerate(terms) if index != input_index]
    del node.input[:]
    node.input.extend(inputs)
    set_equation(node, terms, original)
    protected = (
        {name for item in graph.node for name in item.input if name}
        | {value.name for value in graph.input}
        | {value.name for value in graph.output}
    )
    kept = [tensor for tensor in graph.initializer if tensor.name in protected]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.loads(args.base_costs.read_text())
    costs = raw.get("costs") or {str(row["task"]): row["cost"] for row in raw["ranked"]}
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.baseline) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            for ordinal, (node_index, input_index, name) in enumerate(opportunities(model), 1):
                try:
                    candidate = build(model, node_index, input_index)
                    with tempfile.TemporaryDirectory(prefix=f"eru_{task:03d}_") as tmp:
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
                        "node_index": node_index,
                        "input_index": input_index,
                        "initializer": name,
                        "baseline_cost": baseline_cost,
                        "candidate_cost": int(cost),
                        "candidate_memory": int(memory),
                        "candidate_params": int(params),
                        "projected_gain": math.log(baseline_cost / cost),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    })
                except Exception as exc:
                    errors.append({"task": task, "initializer": name, "error": repr(exc)})
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

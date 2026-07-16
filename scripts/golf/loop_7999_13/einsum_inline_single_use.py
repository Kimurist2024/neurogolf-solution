#!/usr/bin/env python3
"""Inline a single-use Einsum producer into an Einsum consumer.

The rewrite removes one materialized intermediate tensor by composing the two
Einstein equations.  It does not alter initializers or add operators.  Each
candidate is still subjected to checker, strict inference, dual-runtime known,
fresh, and differential gates before adoption.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import string
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


LABELS = string.ascii_lowercase + string.ascii_uppercase


def get_equation(node: onnx.NodeProto) -> str:
    for attr in node.attribute:
        if attr.name == "equation":
            return attr.s.decode()
    raise ValueError("Einsum equation missing")


def split_equation(equation: str) -> tuple[list[str], str] | None:
    if "->" not in equation or "..." in equation:
        return None
    lhs, rhs = equation.split("->", 1)
    return lhs.split(","), rhs


def set_equation(node: onnx.NodeProto, inputs: list[str], output: str) -> None:
    value = ",".join(inputs) + "->" + output
    for attr in node.attribute:
        if attr.name == "equation":
            attr.s = value.encode()
            return
    raise ValueError("Einsum equation missing")


def compose(
    producer: onnx.NodeProto,
    consumer: onnx.NodeProto,
    consumer_input_index: int,
    operand_cap: int,
) -> tuple[list[str], list[str], str] | None:
    producer_parts = split_equation(get_equation(producer))
    consumer_parts = split_equation(get_equation(consumer))
    if producer_parts is None or consumer_parts is None:
        return None
    producer_terms, producer_output = producer_parts
    consumer_terms, consumer_output = consumer_parts
    if len(producer_terms) != len(producer.input) or len(consumer_terms) != len(consumer.input):
        return None
    consumer_operand = consumer_terms[consumer_input_index]
    if (
        len(producer_output) != len(consumer_operand)
        or len(set(producer_output)) != len(producer_output)
        or len(set(consumer_operand)) != len(consumer_operand)
    ):
        return None
    if len(consumer.input) - 1 + len(producer.input) > operand_cap:
        return None
    used = set("".join(consumer_terms) + consumer_output)
    mapping = dict(zip(producer_output, consumer_operand))
    producer_labels = set("".join(producer_terms))
    internal = sorted(producer_labels - set(producer_output))
    free = [label for label in LABELS if label not in used]
    if len(internal) > len(free):
        return None
    mapping.update(dict(zip(internal, free)))
    transformed = ["".join(mapping[label] for label in term) for term in producer_terms]
    new_inputs = (
        list(consumer.input[:consumer_input_index])
        + list(producer.input)
        + list(consumer.input[consumer_input_index + 1 :])
    )
    new_terms = (
        consumer_terms[:consumer_input_index]
        + transformed
        + consumer_terms[consumer_input_index + 1 :]
    )
    return new_inputs, new_terms, consumer_output


def opportunities(model: onnx.ModelProto, operand_cap: int) -> list[dict[str, object]]:
    graph = model.graph
    producer_by_output: dict[str, tuple[int, onnx.NodeProto]] = {}
    use_count: Counter[str] = Counter()
    use_location: dict[str, tuple[int, int]] = {}
    for node_index, node in enumerate(graph.node):
        for name in node.output:
            if name:
                producer_by_output[name] = (node_index, node)
        for input_index, name in enumerate(node.input):
            if name:
                use_count[name] += 1
                use_location[name] = (node_index, input_index)
    graph_outputs = {value.name for value in graph.output}
    rows: list[dict[str, object]] = []
    for output_name, (producer_index, producer) in producer_by_output.items():
        if (
            producer.op_type != "Einsum"
            or len(producer.output) != 1
            or output_name in graph_outputs
            or use_count[output_name] != 1
        ):
            continue
        consumer_index, consumer_input_index = use_location[output_name]
        consumer = graph.node[consumer_index]
        if consumer.op_type != "Einsum" or consumer_index <= producer_index:
            continue
        composed = compose(producer, consumer, consumer_input_index, operand_cap)
        if composed is None:
            continue
        new_inputs, new_terms, consumer_output = composed
        rows.append({
            "producer_index": producer_index,
            "consumer_index": consumer_index,
            "consumer_input_index": consumer_input_index,
            "intermediate": output_name,
            "producer_equation": get_equation(producer),
            "consumer_equation": get_equation(consumer),
            "new_inputs": new_inputs,
            "new_terms": new_terms,
            "consumer_output": consumer_output,
        })
    return rows


def build(model: onnx.ModelProto, opportunity: dict[str, object]) -> onnx.ModelProto:
    result = copy.deepcopy(model)
    graph = result.graph
    producer_index = int(opportunity["producer_index"])
    consumer_index = int(opportunity["consumer_index"])
    consumer = graph.node[consumer_index]
    new_inputs = [str(value) for value in opportunity["new_inputs"]]
    new_terms = [str(value) for value in opportunity["new_terms"]]
    del consumer.input[:]
    consumer.input.extend(new_inputs)
    set_equation(consumer, new_terms, str(opportunity["consumer_output"]))
    del graph.node[producer_index]
    intermediate = str(opportunity["intermediate"])
    kept_vi = [value for value in graph.value_info if value.name != intermediate]
    del graph.value_info[:]
    graph.value_info.extend(kept_vi)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--base-costs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--operand-cap", type=int, default=32)
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
            for ordinal, opportunity in enumerate(opportunities(model, args.operand_cap), 1):
                try:
                    candidate = build(model, opportunity)
                    with tempfile.TemporaryDirectory(prefix=f"eisi_{task:03d}_") as tmp:
                        probe = Path(tmp) / member
                        onnx.save(candidate, probe)
                        memory, params, cost = cost_of(str(probe))
                    baseline_cost = int(costs[str(task)])
                    if cost < 0 or cost >= baseline_cost:
                        continue
                    output = args.out_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, output)
                    concise = {
                        key: value for key, value in opportunity.items()
                        if key not in {"new_inputs", "new_terms"}
                    }
                    rows.append({
                        "task": task,
                        "path": str(output.relative_to(ROOT)),
                        "baseline_cost": baseline_cost,
                        "candidate_cost": int(cost),
                        "candidate_memory": int(memory),
                        "candidate_params": int(params),
                        "projected_gain": math.log(baseline_cost / cost),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                        "opportunity": concise,
                    })
                except Exception as exc:
                    errors.append({
                        "task": task,
                        "intermediate": opportunity["intermediate"],
                        "error": repr(exc),
                    })
    payload = {
        "baseline": str(args.baseline),
        "operand_cap": args.operand_cap,
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

#!/usr/bin/env python3
"""Precontract pairs of unique constant operands in one Einsum node."""

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
import onnxruntime as ort
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def locations(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                result[name].append((node_index, input_index))
    return counts, result


def plan_product(
    first: np.ndarray,
    first_term: str,
    second: np.ndarray,
    second_term: str,
    output: str,
    all_terms: list[str],
) -> tuple[np.ndarray, str, list[str]] | None:
    if (
        first.dtype != second.dtype
        or len(first_term) != first.ndim
        or len(second_term) != second.ndim
        or len(set(first_term)) != len(first_term)
        or len(set(second_term)) != len(second_term)
        or "..." in first_term
        or "..." in second_term
    ):
        return None
    occurrences = Counter("".join(all_terms))
    contracted = [
        label
        for label in first_term
        if label in second_term and label not in output and occurrences[label] == 2
    ]
    product_labels: list[str] = []
    for label in first_term + second_term:
        if label in contracted or label in product_labels:
            continue
        product_labels.append(label)
    product_term = "".join(product_labels)
    if len(product_term) > 52:
        return None
    equation = f"{first_term},{second_term}->{product_term}"
    try:
        if first.dtype == np.float16:
            product = np.einsum(
                equation,
                first.astype(np.float32),
                second.astype(np.float32),
                optimize=False,
            ).astype(np.float16)
        else:
            product = np.einsum(equation, first, second, optimize=False)
            product = np.asarray(product, dtype=first.dtype)
    except Exception:
        return None
    if first.size + second.size <= product.size:
        return None
    return np.asarray(product), product_term, contracted


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: np.asarray(numpy_helper.to_array(item)) for name, item in initializers.items()}
    counts, _ = locations(model.graph)
    result: list[dict[str, object]] = []
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Einsum":
            continue
        text = equation_attribute(node).s.decode("ascii")
        if "->" not in text or "..." in text:
            continue
        lhs, output = text.split("->", 1)
        terms = lhs.split(",")
        if len(terms) != len(node.input):
            continue
        eligible = [
            index
            for index, name in enumerate(node.input)
            if name in arrays and counts[name] == 1
        ]
        for first_index, second_index in itertools.combinations(eligible, 2):
            first_name = node.input[first_index]
            second_name = node.input[second_index]
            built = plan_product(
                arrays[first_name],
                terms[first_index],
                arrays[second_name],
                terms[second_index],
                output,
                terms,
            )
            if built is None:
                continue
            product, product_term, contracted = built
            saving = int(arrays[first_name].size + arrays[second_name].size - product.size)
            result.append(
                {
                    "node_index": node_index,
                    "first_index": first_index,
                    "first": first_name,
                    "first_term": terms[first_index],
                    "first_params": int(arrays[first_name].size),
                    "second_index": second_index,
                    "second": second_name,
                    "second_term": terms[second_index],
                    "second_params": int(arrays[second_name].size),
                    "product_term": product_term,
                    "product_params": int(product.size),
                    "parameter_saving": saving,
                    "contracted_labels": contracted,
                    "product": product,
                }
            )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object], ordinal: int) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    first_name = str(plan["first"])
    second_name = str(plan["second"])
    product = plan["product"]
    if not isinstance(product, np.ndarray):
        raise TypeError("missing product")
    product_name = f"__fuc_{first_name}_{second_name}_{ordinal}"
    existing = {
        item.name for item in graph.initializer
    } | {name for node in graph.node for name in (*node.input, *node.output) if name}
    while product_name in existing:
        product_name += "_"
    graph.initializer.append(numpy_helper.from_array(product, product_name))

    node = graph.node[int(plan["node_index"])]
    attr = equation_attribute(node)
    lhs, output = attr.s.decode("ascii").split("->", 1)
    terms = lhs.split(",")
    first_index = int(plan["first_index"])
    second_index = int(plan["second_index"])
    low, high = sorted((first_index, second_index))
    inputs = list(node.input)
    inputs[low] = product_name
    terms[low] = str(plan["product_term"])
    del inputs[high]
    del terms[high]
    del node.input[:]
    node.input.extend(inputs)
    attr.s = (",".join(terms) + "->" + output).encode("ascii")

    kept = [
        item
        for item in graph.initializer
        if item.name not in {first_name, second_name}
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"fuse_constants_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in plan.items() if key != "product"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ort.set_default_logger_severity(4)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    opportunity_counts: dict[str, int] = {}
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            member = f"task{task:03d}.onnx"
            model = onnx.load_model_from_string(archive.read(member))
            task_plans = plans(model)
            if not task_plans:
                continue
            opportunity_counts[str(task)] = len(task_plans)
            try:
                base_memory, base_params, base_cost = measure(model, task)
            except Exception as exc:
                errors.append({"task": task, "stage": "base_cost", "error": repr(exc)})
                continue
            for ordinal, plan in enumerate(task_plans[: args.max_per_task], 1):
                info = concise(plan)
                try:
                    candidate = build(model, plan, ordinal)
                    memory, params, cost = measure(candidate, task)
                    if cost < 0 or cost >= base_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:02d}.onnx"
                    onnx.save(candidate, path)
                    rows.append(
                        {
                            "task": task,
                            "path": str(path),
                            **info,
                            "baseline_memory": base_memory,
                            "baseline_params": base_params,
                            "baseline_cost": base_cost,
                            "candidate_memory": memory,
                            "candidate_params": params,
                            "candidate_cost": cost,
                            "projected_gain": math.log(base_cost / cost),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    )
                except Exception as exc:
                    errors.append({"task": task, "stage": "build", **info, "error": repr(exc)})
    payload = {
        "source_zip": str(args.zip),
        "opportunity_counts": opportunity_counts,
        "rows": rows,
        "errors": errors,
    }
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "opportunity_task_count": len(opportunity_counts),
        "opportunity_count": sum(opportunity_counts.values()),
        "candidate_count": len(rows),
        "projected_gain": sum(float(row["projected_gain"]) for row in rows),
        "error_count": len(errors),
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

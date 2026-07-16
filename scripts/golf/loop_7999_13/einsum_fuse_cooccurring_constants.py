#!/usr/bin/env python3
"""Fuse two constants that co-occur exactly once in the same Einsum nodes."""

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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from einsum_fuse_unique_constants import equation_attribute, plan_product  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    node_initializer_positions: dict[int, dict[str, list[int]]] = {}
    valid_names = set(arrays)
    for node_index, node in enumerate(model.graph.node):
        by_name: dict[str, list[int]] = defaultdict(list)
        for input_index, name in enumerate(node.input):
            if name in valid_names:
                uses[name].append((node_index, input_index))
                by_name[name].append(input_index)
        node_initializer_positions[node_index] = by_name

    signatures: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name, locations in uses.items():
        if not locations:
            continue
        node_ids = tuple(node_index for node_index, _ in locations)
        if len(set(node_ids)) != len(node_ids):
            continue
        if any(model.graph.node[node_index].op_type != "Einsum" for node_index in node_ids):
            continue
        signatures[node_ids].append(name)

    result: list[dict[str, object]] = []
    for node_ids, names in signatures.items():
        for first_name, second_name in itertools.combinations(names, 2):
            products: list[dict[str, object]] = []
            valid = True
            unique_products: dict[tuple[str, tuple[int, ...], bytes], int] = {}
            for node_index in node_ids:
                node = model.graph.node[node_index]
                text = equation_attribute(node).s.decode("ascii")
                if "->" not in text or "..." in text:
                    valid = False
                    break
                lhs, output = text.split("->", 1)
                terms = lhs.split(",")
                first_positions = node_initializer_positions[node_index][first_name]
                second_positions = node_initializer_positions[node_index][second_name]
                if len(first_positions) != 1 or len(second_positions) != 1:
                    valid = False
                    break
                first_index = first_positions[0]
                second_index = second_positions[0]
                built = plan_product(
                    arrays[first_name],
                    terms[first_index],
                    arrays[second_name],
                    terms[second_index],
                    output,
                    terms,
                )
                if built is None:
                    valid = False
                    break
                product, product_term, contracted = built
                contiguous = np.ascontiguousarray(product)
                key = (contiguous.dtype.str, tuple(contiguous.shape), contiguous.tobytes())
                product_id = unique_products.setdefault(key, len(unique_products))
                products.append(
                    {
                        "node_index": node_index,
                        "first_index": first_index,
                        "first_term": terms[first_index],
                        "second_index": second_index,
                        "second_term": terms[second_index],
                        "product_term": product_term,
                        "contracted_labels": contracted,
                        "product_id": product_id,
                        "product": contiguous,
                    }
                )
            if not valid:
                continue
            unique_arrays: dict[int, np.ndarray] = {}
            for item in products:
                unique_arrays.setdefault(int(item["product_id"]), item["product"])
            new_params = sum(int(array.size) for array in unique_arrays.values())
            saving = int(arrays[first_name].size + arrays[second_name].size - new_params)
            if saving <= 0:
                continue
            result.append(
                {
                    "first": first_name,
                    "first_params": int(arrays[first_name].size),
                    "second": second_name,
                    "second_params": int(arrays[second_name].size),
                    "new_params": new_params,
                    "parameter_saving": saving,
                    "products": products,
                    "unique_products": unique_arrays,
                }
            )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object], ordinal: int) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    product_names: dict[int, str] = {}
    existing = {
        item.name for item in graph.initializer
    } | {name for node in graph.node for name in (*node.input, *node.output) if name}
    for product_id, array in plan["unique_products"].items():
        name = f"__fcc_{plan['first']}_{plan['second']}_{ordinal}_{product_id}"
        while name in existing:
            name += "_"
        existing.add(name)
        product_names[int(product_id)] = name
        graph.initializer.append(numpy_helper.from_array(array, name))

    for item in plan["products"]:
        node = graph.node[int(item["node_index"])]
        attr = equation_attribute(node)
        lhs, output = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        inputs = list(node.input)
        first_index = int(item["first_index"])
        second_index = int(item["second_index"])
        low, high = sorted((first_index, second_index))
        inputs[low] = product_names[int(item["product_id"])]
        terms[low] = str(item["product_term"])
        del inputs[high]
        del terms[high]
        del node.input[:]
        node.input.extend(inputs)
        attr.s = (",".join(terms) + "->" + output).encode("ascii")

    removed = {str(plan["first"]), str(plan["second"])}
    kept = [item for item in graph.initializer if item.name not in removed]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"cooccur_constants_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {
        **{
            key: value
            for key, value in plan.items()
            if key not in {"products", "unique_products"}
        },
        "products": [
            {key: value for key, value in item.items() if key != "product"}
            for item in plan["products"]
        ],
        "unique_product_count": len(plan["unique_products"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=8)
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

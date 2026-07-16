#!/usr/bin/env python3
"""Eliminate a shared Einsum constant by precontracting each unique target."""

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
import onnxruntime as ort
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from einsum_fuse_unique_constants import equation_attribute, plan_product  # noqa: E402
from scripts.golf.rank_dir import cost_of  # noqa: E402


def locations(graph: onnx.GraphProto) -> tuple[Counter[str], dict[str, list[tuple[int, int]]]]:
    counts: Counter[str] = Counter()
    result: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(graph.node):
        for input_index, name in enumerate(node.input):
            if name:
                counts[name] += 1
                result[name].append((node_index, input_index))
    return counts, result


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: np.asarray(numpy_helper.to_array(item)) for name, item in initializers.items()}
    counts, use_locations = locations(model.graph)
    result: list[dict[str, object]] = []
    for source_name, source in arrays.items():
        if counts[source_name] < 2:
            continue
        selected_targets: set[str] = set()
        use_plans: list[dict[str, object]] = []
        valid = True
        for node_index, source_index in use_locations[source_name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum":
                valid = False
                break
            text = equation_attribute(node).s.decode("ascii")
            if "->" not in text or "..." in text:
                valid = False
                break
            lhs, output = text.split("->", 1)
            terms = lhs.split(",")
            if len(terms) != len(node.input):
                valid = False
                break
            options: list[tuple[int, dict[str, object]]] = []
            for target_index, target_name in enumerate(node.input):
                if (
                    target_index == source_index
                    or target_name not in arrays
                    or counts[target_name] != 1
                    or target_name in selected_targets
                ):
                    continue
                built = plan_product(
                    source,
                    terms[source_index],
                    arrays[target_name],
                    terms[target_index],
                    output,
                    terms,
                )
                if built is None:
                    continue
                product, product_term, contracted = built
                delta = int(product.size - arrays[target_name].size)
                options.append(
                    (
                        delta,
                        {
                            "node_index": node_index,
                            "source_index": source_index,
                            "source_term": terms[source_index],
                            "target_index": target_index,
                            "target": target_name,
                            "target_term": terms[target_index],
                            "target_params": int(arrays[target_name].size),
                            "product_term": product_term,
                            "product_params": int(product.size),
                            "contracted_labels": contracted,
                            "product": product,
                        },
                    )
                )
            if not options:
                valid = False
                break
            _, selected = min(options, key=lambda item: item[0])
            selected_targets.add(str(selected["target"]))
            use_plans.append(selected)
        if not valid:
            continue
        total_product = sum(int(item["product_params"]) for item in use_plans)
        total_target = sum(int(item["target_params"]) for item in use_plans)
        saving = int(source.size + total_target - total_product)
        if saving <= 0:
            continue
        result.append(
            {
                "source": source_name,
                "source_params": int(source.size),
                "source_uses": counts[source_name],
                "targets": sorted(selected_targets),
                "parameter_saving": saving,
                "uses": use_plans,
            }
        )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object], ordinal: int) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    product_initializers: list[onnx.TensorProto] = []
    product_names: dict[tuple[int, int], str] = {}
    existing = {
        item.name for item in graph.initializer
    } | {name for node in graph.node for name in (*node.input, *node.output) if name}
    for use_ordinal, use in enumerate(plan["uses"]):
        if not isinstance(use, dict):
            raise TypeError("invalid use plan")
        name = f"__fsc_{plan['source']}_{ordinal}_{use_ordinal}"
        while name in existing:
            name += "_"
        existing.add(name)
        product = use["product"]
        if not isinstance(product, np.ndarray):
            raise TypeError("missing product")
        product_initializers.append(numpy_helper.from_array(product, name))
        key = (int(use["node_index"]), int(use["source_index"]))
        product_names[key] = name
        grouped[key[0]].append(use)

    for node_index, node_uses in grouped.items():
        node = graph.node[node_index]
        attr = equation_attribute(node)
        lhs, output = attr.s.decode("ascii").split("->", 1)
        old_terms = lhs.split(",")
        by_index: dict[int, tuple[dict[str, object], bool]] = {}
        for use in node_uses:
            source_index = int(use["source_index"])
            target_index = int(use["target_index"])
            by_index[source_index] = (use, source_index < target_index)
            by_index[target_index] = (use, target_index < source_index)
        new_inputs: list[str] = []
        new_terms: list[str] = []
        for index, (input_name, term) in enumerate(zip(node.input, old_terms)):
            pair = by_index.get(index)
            if pair is None:
                new_inputs.append(input_name)
                new_terms.append(term)
                continue
            use, emit = pair
            if not emit:
                continue
            key = (node_index, int(use["source_index"]))
            new_inputs.append(product_names[key])
            new_terms.append(str(use["product_term"]))
        del node.input[:]
        node.input.extend(new_inputs)
        attr.s = (",".join(new_terms) + "->" + output).encode("ascii")

    removed = {str(plan["source"]), *(str(name) for name in plan["targets"])}
    kept = [item for item in graph.initializer if item.name not in removed]
    kept.extend(product_initializers)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"shared_constant_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {
        **{key: value for key, value in plan.items() if key != "uses"},
        "uses": [
            {key: value for key, value in use.items() if key != "product"}
            for use in plan["uses"]
        ],
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

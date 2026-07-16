#!/usr/bin/env python3
"""Collapse uniform Einsum initializer operands to scalar tensors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from collections import defaultdict
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


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in arrays:
                uses[name].append((node_index, input_index))
    result: list[dict[str, object]] = []
    for name, array in arrays.items():
        if array.size <= 1 or not uses[name]:
            continue
        first = array.reshape(-1)[0]
        if not np.all(array == first):
            continue
        rewrites: list[dict[str, object]] = []
        valid = True
        for node_index, input_index in uses[name]:
            node = model.graph.node[node_index]
            if node.op_type != "Einsum":
                valid = False
                break
            attr = equation_attribute(node)
            text = attr.s.decode("ascii")
            if "->" not in text or "..." in text:
                valid = False
                break
            lhs, output = text.split("->", 1)
            terms = lhs.split(",")
            if input_index >= len(terms):
                valid = False
                break
            term = terms[input_index]
            other_labels = set(
                "".join(value for index, value in enumerate(terms) if index != input_index)
            )
            if not set(term).issubset(other_labels):
                valid = False
                break
            rewrites.append(
                {
                    "node_index": node_index,
                    "input_index": input_index,
                    "old_term": term,
                }
            )
        if valid:
            result.append(
                {
                    "initializer": name,
                    "shape": list(array.shape),
                    "original_params": int(array.size),
                    "candidate_params": 1,
                    "parameter_saving": int(array.size - 1),
                    "value": np.asarray(first, dtype=array.dtype).reshape(()),
                    "rewrites": rewrites,
                }
            )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    name = str(plan["initializer"])
    value = plan["value"]
    if not isinstance(value, np.ndarray):
        raise TypeError("missing scalar")
    kept: list[onnx.TensorProto] = []
    for item in graph.initializer:
        if item.name == name:
            kept.append(numpy_helper.from_array(value, name))
        else:
            kept.append(item)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for rewrite in plan["rewrites"]:
        node = graph.node[int(rewrite["node_index"])]
        attr = equation_attribute(node)
        lhs, output = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        terms[int(rewrite["input_index"])] = ""
        attr.s = (",".join(terms) + "->" + output).encode("ascii")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"uniform_einsum_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    result = {key: value for key, value in plan.items() if key != "value"}
    scalar = plan["value"]
    result["scalar_value"] = scalar.item() if isinstance(scalar, np.ndarray) else scalar
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
            for ordinal, plan in enumerate(task_plans, 1):
                info = concise(plan)
                try:
                    candidate = build(model, plan)
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

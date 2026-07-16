#!/usr/bin/env python3
"""Pre-sum initializer axes whose Einsum labels occur nowhere else."""

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
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.golf.rank_dir import cost_of  # noqa: E402


def equation_attribute(node: onnx.NodeProto) -> onnx.AttributeProto:
    return next(attr for attr in node.attribute if attr.name == "equation")


def uses_of(model: onnx.ModelProto) -> dict[str, list[tuple[int, int]]]:
    initializers = {item.name for item in model.graph.initializer}
    uses: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_index, node in enumerate(model.graph.node):
        for input_index, name in enumerate(node.input):
            if name in initializers:
                uses[name].append((node_index, input_index))
    return uses


def plans(model: onnx.ModelProto) -> list[dict[str, object]]:
    arrays = {
        item.name: np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    uses = uses_of(model)
    result: list[dict[str, object]] = []
    for name, array in arrays.items():
        if array.ndim == 0 or not uses.get(name) or not np.all(np.isfinite(array)):
            continue
        descriptions: list[tuple[int, int, str, str, set[str]]] = []
        valid = True
        for node_index, input_index in uses[name]:
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
            term = terms[input_index]
            if len(term) != array.ndim or len(set(term)) != len(term):
                valid = False
                break
            other_labels = set(
                "".join(value for index, value in enumerate(terms) if index != input_index)
            )
            descriptions.append((node_index, input_index, term, output, other_labels))
        if not valid:
            continue

        safe_axes: list[int] = []
        for axis in range(array.ndim):
            if all(
                term[axis] not in output and term[axis] not in other_labels
                for _, _, term, output, other_labels in descriptions
            ):
                safe_axes.append(axis)
        if not safe_axes:
            continue

        reduced = array
        for axis in sorted(safe_axes, reverse=True):
            reduced = np.sum(reduced, axis=axis, dtype=array.dtype)
        reduced = np.asarray(reduced, dtype=array.dtype)
        if reduced.size >= array.size or not np.all(np.isfinite(reduced)):
            continue
        axis_set = set(safe_axes)
        rewrites = []
        for node_index, input_index, term, _, _ in descriptions:
            rewrites.append(
                {
                    "node_index": node_index,
                    "input_index": input_index,
                    "old_term": term,
                    "new_term": "".join(
                        label for axis, label in enumerate(term) if axis not in axis_set
                    ),
                }
            )
        result.append(
            {
                "initializer": name,
                "old_shape": list(array.shape),
                "new_shape": list(reduced.shape),
                "collapsed_axes": safe_axes,
                "original_params": int(array.size),
                "candidate_initializer_params": int(reduced.size),
                "parameter_saving": int(array.size - reduced.size),
                "reduced": reduced,
                "rewrites": rewrites,
            }
        )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source)
    name = str(plan["initializer"])
    reduced = plan["reduced"]
    if not isinstance(reduced, np.ndarray):
        raise TypeError("missing reduced initializer")
    for index, item in enumerate(model.graph.initializer):
        if item.name == name:
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(reduced, name))
            break
    else:
        raise KeyError(name)
    for rewrite in plan["rewrites"]:
        node = model.graph.node[int(rewrite["node_index"])]
        attr = equation_attribute(node)
        lhs, output = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        terms[int(rewrite["input_index"])] = str(rewrite["new_term"])
        attr.s = (",".join(terms) + "->" + output).encode("ascii")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"private_axis_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def concise(plan: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in plan.items() if key != "reduced"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    opportunities: dict[str, int] = {}
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            task_plans = plans(model)
            if not task_plans:
                continue
            opportunities[str(task)] = len(task_plans)
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
        "opportunity_counts": opportunities,
        "rows": rows,
        "errors": errors,
    }
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "opportunity_task_count": len(opportunities),
                "opportunity_count": sum(opportunities.values()),
                "candidate_count": len(rows),
                "projected_gain": sum(float(row["projected_gain"]) for row in rows),
                "error_count": len(errors),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

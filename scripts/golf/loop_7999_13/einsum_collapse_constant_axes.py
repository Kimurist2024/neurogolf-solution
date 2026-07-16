#!/usr/bin/env python3
"""Remove initializer axes that are constant and bound by other Einsum inputs."""

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


def constant_axes(array: np.ndarray) -> list[int]:
    result: list[int] = []
    for axis, size in enumerate(array.shape):
        if size <= 1:
            continue
        first = np.take(array, [0], axis=axis)
        if np.array_equal(array, np.repeat(first, size, axis=axis), equal_nan=True):
            result.append(axis)
    return result


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
        candidates = constant_axes(array)
        if not candidates or not uses[name]:
            continue
        safe_axes = list(candidates)
        rewrites: list[dict[str, object]] = []
        use_terms: list[tuple[int, int, str, str, set[str]]] = []
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
            use_terms.append((node_index, input_index, term, output, other_labels))
        if not valid:
            continue
        axis_modes: dict[int, str] = {}
        for axis in safe_axes:
            modes: list[str] = []
            for _, _, term, output, other_labels in use_terms:
                label = term[axis]
                if label in other_labels:
                    modes.append("bound")
                elif label not in output:
                    modes.append("summed")
                else:
                    modes.append("unsafe")
            if modes and len(set(modes)) == 1 and modes[0] != "unsafe":
                axis_modes[axis] = modes[0]
        safe_axes = sorted(axis_modes)
        if not safe_axes:
            continue
        reduced = array
        for axis in sorted(safe_axes, reverse=True):
            reduced = np.take(reduced, 0, axis=axis)
        summed_axes = [axis for axis in safe_axes if axis_modes[axis] == "summed"]
        if summed_axes:
            scale = math.prod(int(array.shape[axis]) for axis in summed_axes)
            reduced = np.multiply(reduced, scale, dtype=array.dtype)
        if reduced.size >= array.size:
            continue
        for node_index, input_index, term, _, _ in use_terms:
            new_term = "".join(
                label for axis, label in enumerate(term) if axis not in set(safe_axes)
            )
            rewrites.append(
                {
                    "node_index": node_index,
                    "input_index": input_index,
                    "old_term": term,
                    "new_term": new_term,
                }
            )
        result.append(
            {
                "initializer": name,
                "old_shape": list(array.shape),
                "new_shape": list(reduced.shape),
                "collapsed_axes": safe_axes,
                "summed_axes": summed_axes,
                "original_params": int(array.size),
                "candidate_params": int(reduced.size),
                "parameter_saving": int(array.size - reduced.size),
                "reduced": np.asarray(reduced),
                "rewrites": rewrites,
            }
        )
    result.sort(key=lambda item: -int(item["parameter_saving"]))
    return result


def build(source_model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    model = copy.deepcopy(source_model)
    graph = model.graph
    name = str(plan["initializer"])
    reduced = plan["reduced"]
    if not isinstance(reduced, np.ndarray):
        raise TypeError("missing reduced initializer")
    kept: list[onnx.TensorProto] = []
    for item in graph.initializer:
        if item.name == name:
            kept.append(numpy_helper.from_array(reduced, name))
        else:
            kept.append(item)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    for rewrite in plan["rewrites"]:
        node = graph.node[int(rewrite["node_index"])]
        attr = equation_attribute(node)
        lhs, output = attr.s.decode("ascii").split("->", 1)
        terms = lhs.split(",")
        terms[int(rewrite["input_index"])] = str(rewrite["new_term"])
        attr.s = (",".join(terms) + "->" + output).encode("ascii")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"collapse_axes_{task:03d}_") as tmp:
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

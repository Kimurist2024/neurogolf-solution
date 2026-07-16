#!/usr/bin/env python3
"""Generate lower-rank probes by deleting one constant-only Einsum component."""

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


def equation(node: onnx.NodeProto) -> str:
    return next(attr.s.decode("ascii") for attr in node.attribute if attr.name == "equation")


def measure(model: onnx.ModelProto, task: int) -> tuple[int, int, int]:
    with tempfile.TemporaryDirectory(prefix=f"latent_prune_{task:03d}_") as tmp:
        path = Path(tmp) / f"task{task:03d}.onnx"
        onnx.save(model, path)
        memory, params, cost = cost_of(str(path))
        return int(memory), int(params), int(cost)


def opportunities(model: onnx.ModelProto) -> list[dict[str, object]]:
    if len(model.graph.node) != 1 or model.graph.node[0].op_type != "Einsum":
        return []
    node = model.graph.node[0]
    text = equation(node)
    if "..." in text or "->" not in text:
        return []
    lhs, output = text.split("->", 1)
    terms = lhs.split(",")
    if len(terms) != len(node.input):
        return []
    initializers = {item.name: item for item in model.graph.initializer}
    input_labels = set(
        "".join(term for name, term in zip(node.input, terms) if name not in initializers)
    )
    label_dimensions: dict[str, int] = {}
    label_uses: dict[str, list[tuple[str, int]]] = defaultdict(list)
    name_terms: dict[str, list[str]] = defaultdict(list)
    for name, term in zip(node.input, terms):
        if name not in initializers:
            continue
        shape = list(initializers[name].dims)
        if len(shape) != len(term) or len(set(term)) != len(term):
            return []
        name_terms[name].append(term)
        for axis, (label, size) in enumerate(zip(term, shape)):
            if label in label_dimensions and label_dimensions[label] != size:
                return []
            label_dimensions[label] = int(size)
            label_uses[label].append((name, axis))

    result: list[dict[str, object]] = []
    for label, size in label_dimensions.items():
        if label in input_labels or label in output or not 1 < size <= 8:
            continue
        uses = label_uses[label]
        if len(uses) < 2:
            continue
        axes_by_name: dict[str, set[int]] = defaultdict(set)
        for name, axis in uses:
            axes_by_name[name].add(axis)
        if any(len(axes) != 1 for axes in axes_by_name.values()):
            continue
        consistent = True
        for name, axes in axes_by_name.items():
            axis = next(iter(axes))
            if any(term[axis] != label for term in name_terms[name]):
                consistent = False
                break
        if not consistent:
            continue
        for removed in range(size):
            result.append(
                {
                    "label": label,
                    "dimension": size,
                    "removed_component": removed,
                    "axes_by_initializer": {
                        name: next(iter(axes)) for name, axes in axes_by_name.items()
                    },
                }
            )
    return result


def build(model: onnx.ModelProto, plan: dict[str, object]) -> onnx.ModelProto:
    candidate = copy.deepcopy(model)
    removed = int(plan["removed_component"])
    axes = dict(plan["axes_by_initializer"])
    for index, item in enumerate(candidate.graph.initializer):
        if item.name not in axes:
            continue
        array = np.asarray(numpy_helper.to_array(item))
        keep = [value for value in range(array.shape[int(axes[item.name])]) if value != removed]
        reduced = np.take(array, keep, axis=int(axes[item.name]))
        candidate.graph.initializer[index].CopyFrom(
            numpy_helper.from_array(np.asarray(reduced, dtype=array.dtype), item.name)
        )
    onnx.checker.check_model(candidate, full_check=True)
    onnx.shape_inference.infer_shapes(candidate, strict_mode=True, data_prop=True)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with zipfile.ZipFile(args.zip) as archive:
        for task in range(1, 401):
            model = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
            plans = opportunities(model)
            if not plans:
                continue
            try:
                base_memory, base_params, base_cost = measure(model, task)
            except Exception as exc:
                errors.append({"task": task, "stage": "base", "error": repr(exc)})
                continue
            for ordinal, plan in enumerate(plans, 1):
                try:
                    candidate = build(model, plan)
                    memory, params, cost = measure(candidate, task)
                    if cost < 0 or cost >= base_cost:
                        continue
                    path = args.output_dir / f"task{task:03d}_r{ordinal:03d}.onnx"
                    onnx.save(candidate, path)
                    rows.append(
                        {
                            "task": task,
                            "path": str(path),
                            **plan,
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
                    errors.append({"task": task, "stage": "build", **plan, "error": repr(exc)})
    rows.sort(key=lambda row: (-float(row["projected_gain"]), int(row["task"])))
    manifest = args.output_dir / "build_manifest.json"
    manifest.write_text(
        json.dumps({"source_zip": str(args.zip), "rows": rows, "errors": errors}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "candidate_count": len(rows),
                "task_count": len({int(row["task"]) for row in rows}),
                "max_gain": max((float(row["projected_gain"]) for row in rows), default=0.0),
                "error_count": len(errors),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

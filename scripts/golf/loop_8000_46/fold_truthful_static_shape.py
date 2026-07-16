#!/usr/bin/env python3
"""Fold Shape/Size of statically inferred tensors into charged-cheaper constants."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(__file__).resolve().parent / "submission_8000.46_wave4_safe_meta.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lane_static_shape_fold"


def attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    return next((int(item.i) for item in node.attribute if item.name == name), default)


def static_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result: dict[str, tuple[int, ...]] = {}
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not item.type.HasField("tensor_type") or not item.type.tensor_type.HasField("shape"):
            continue
        dims = item.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            result[item.name] = tuple(int(dim.dim_value) for dim in dims)
    for item in model.graph.initializer:
        result[item.name] = tuple(int(dim) for dim in item.dims)
    return result


def fold(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    model = copy.deepcopy(model)
    shapes = static_shapes(model)
    initializer_names = {item.name for item in model.graph.initializer}
    kept_nodes = []
    replacements = []
    changes: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in {"Shape", "Size"} or len(node.input) != 1 or len(node.output) != 1:
            kept_nodes.append(node)
            continue
        shape = shapes.get(node.input[0])
        output_name = node.output[0]
        if shape is None or not output_name or output_name in initializer_names:
            kept_nodes.append(node)
            continue
        if node.op_type == "Shape":
            rank = len(shape)
            start = attribute_int(node, "start", 0)
            end = attribute_int(node, "end", rank)
            if start < 0:
                start += rank
            if end < 0:
                end += rank
            start = min(max(start, 0), rank)
            end = min(max(end, 0), rank)
            value = np.asarray(shape[start:end], dtype=np.int64)
        else:
            value = np.asarray(math.prod(shape) if shape else 1, dtype=np.int64)
        replacements.append(numpy_helper.from_array(value, name=output_name))
        initializer_names.add(output_name)
        changes.append(
            {
                "node_index": index,
                "op": node.op_type,
                "source": node.input[0],
                "static_shape": list(shape),
                "output": output_name,
                "value": value.reshape(-1).tolist(),
            }
        )
    if not changes:
        raise RuntimeError("no static Shape/Size nodes")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(replacements)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, changes


def cost(path: Path) -> tuple[int, int, int]:
    from scripts.golf.rank_dir import cost_of

    return tuple(int(value) for value in cost_of(str(path)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(args.source) as archive:
        for task in range(1, 401):
            try:
                source = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                candidate, changes = fold(source)
                with tempfile.TemporaryDirectory(prefix=f"shape_fold_{task:03d}_") as tmp:
                    source_path = Path(tmp) / "source.onnx"
                    candidate_path = Path(tmp) / "candidate.onnx"
                    onnx.save(source, source_path)
                    onnx.save(candidate, candidate_path)
                    base_memory, base_params, base_cost = cost(source_path)
                    memory, params, candidate_cost = cost(candidate_path)
                row: dict[str, object] = {
                    "task": task,
                    "base_memory": base_memory,
                    "base_params": base_params,
                    "base_cost": base_cost,
                    "candidate_memory": memory,
                    "candidate_params": params,
                    "candidate_cost": candidate_cost,
                    "cost_reduction": base_cost - candidate_cost,
                    "changes": changes,
                }
                if candidate_cost < base_cost:
                    path = output_dir / f"task{task:03d}.onnx"
                    onnx.save(candidate, path)
                    row["path"] = str(path.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(row)
            except RuntimeError as exc:
                if str(exc) != "no static Shape/Size nodes":
                    rows.append({"task": task, "error": repr(exc)})
            except Exception as exc:
                rows.append({"task": task, "error": repr(exc)})
    winners = [row for row in rows if int(row.get("cost_reduction", 0)) > 0]
    report = {
        "source": str(args.source),
        "winner_count": len(winners),
        "projected_gain_upper_bound": sum(math.log(row["base_cost"] / row["candidate_cost"]) for row in winners),
        "rows": rows,
    }
    manifest = output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"winner_count": len(winners), "projected_gain_upper_bound": report["projected_gain_upper_bound"], "manifest": str(manifest.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

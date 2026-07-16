#!/usr/bin/env python3
"""Remove redundant optional axes/steps inputs using standard ONNX defaults."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(__file__).resolve().parent / "submission_8000.46_wave4_safe_meta.zip"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "lane_optional_axes"


def inferred_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    result: dict[str, tuple[int, ...]] = {}
    for item in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not item.type.HasField("tensor_type") or not item.type.tensor_type.HasField("shape"):
            continue
        dims = item.type.tensor_type.shape.dim
        if all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            result[item.name] = tuple(int(dim.dim_value) for dim in dims)
    return result


def array(initializers: dict[str, onnx.TensorProto], name: str) -> np.ndarray | None:
    item = initializers.get(name)
    return None if item is None else np.asarray(numpy_helper.to_array(item))


def trim_empty_inputs(node: onnx.NodeProto) -> None:
    while node.input and not node.input[-1]:
        del node.input[-1]


def build(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]], int]:
    model = copy.deepcopy(model)
    shapes = inferred_shapes(model)
    initializers = {item.name: item for item in model.graph.initializer}
    changes: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Slice" and len(node.input) >= 4 and node.input[3]:
            starts = array(initializers, node.input[1])
            axes = array(initializers, node.input[3])
            if starts is None or axes is None:
                continue
            expected = list(range(int(starts.size)))
            if [int(value) for value in axes.reshape(-1)] != expected:
                continue
            removed = [node.input[3]]
            if len(node.input) >= 5 and node.input[4]:
                steps = array(initializers, node.input[4])
                if steps is None or not np.all(steps == 1):
                    node.input[3] = ""
                    changes.append({"node_index": index, "op": node.op_type, "removed_defaults": removed})
                    continue
                removed.append(node.input[4])
            del node.input[3:]
            changes.append({"node_index": index, "op": node.op_type, "removed_defaults": removed})
        elif node.op_type == "Pad" and len(node.input) >= 4 and node.input[3]:
            axes = array(initializers, node.input[3])
            rank = len(shapes.get(node.input[0], ()))
            if axes is None or rank == 0:
                continue
            normalized = [int(value) % rank for value in axes.reshape(-1)]
            if normalized != list(range(rank)):
                continue
            removed = node.input[3]
            del node.input[3:]
            changes.append({"node_index": index, "op": node.op_type, "removed_defaults": [removed]})
        elif node.op_type == "Resize" and len(node.input) >= 5 and node.input[4]:
            axes = array(initializers, node.input[4])
            rank = len(shapes.get(node.input[0], ()))
            if axes is None or rank == 0:
                continue
            normalized = [int(value) % rank for value in axes.reshape(-1)]
            if normalized != list(range(rank)):
                continue
            removed = node.input[4]
            del node.input[4:]
            changes.append({"node_index": index, "op": node.op_type, "removed_defaults": [removed]})
        elif node.op_type == "Squeeze" and len(node.input) >= 2 and node.input[1]:
            axes = array(initializers, node.input[1])
            shape = shapes.get(node.input[0])
            if axes is None or shape is None:
                continue
            singleton = [axis for axis, size in enumerate(shape) if size == 1]
            normalized = sorted({int(value) % len(shape) for value in axes.reshape(-1)})
            if normalized != singleton:
                continue
            removed = node.input[1]
            del node.input[1:]
            changes.append({"node_index": index, "op": node.op_type, "removed_defaults": [removed]})
        elif node.op_type == "Split" and len(node.input) >= 2 and node.input[1]:
            sizes = array(initializers, node.input[1])
            if sizes is None:
                continue
            flat = [int(value) for value in sizes.reshape(-1)]
            if len(flat) != len(node.output) or not flat or len(set(flat)) != 1:
                continue
            removed = node.input[1]
            del node.input[1:]
            changes.append({"node_index": index, "op": node.op_type, "removed_defaults": [removed]})
        trim_empty_inputs(node)
    if not changes:
        raise RuntimeError("no redundant optional inputs")

    uses = Counter(name for node in model.graph.node for name in node.input if name)
    uses.update(item.name for item in model.graph.output)
    kept = []
    saving = 0
    removed_initializers = {name for change in changes for name in change["removed_defaults"]}
    for item in model.graph.initializer:
        if item.name in removed_initializers and uses[item.name] == 0:
            saving += math.prod(item.dims) if item.dims else 1
        else:
            kept.append(item)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    if saving <= 0:
        raise RuntimeError("no unique initializer saving")
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    return model, changes, int(saving)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with zipfile.ZipFile(args.source) as archive:
        for task in range(1, 401):
            try:
                source = onnx.load_model_from_string(archive.read(f"task{task:03d}.onnx"))
                candidate, changes, saving = build(source)
            except RuntimeError as exc:
                if str(exc) not in {"no redundant optional inputs", "no unique initializer saving"}:
                    failures.append({"task": task, "error": repr(exc)})
                continue
            except Exception as exc:
                failures.append({"task": task, "error": repr(exc)})
                continue
            path = output_dir / f"task{task:03d}.onnx"
            onnx.save(candidate, path)
            rows.append(
                {
                    "task": task,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "parameter_saving": saving,
                    "changes": changes,
                }
            )
    report = {"source": str(args.source), "candidate_count": len(rows), "rows": rows, "failures": failures}
    manifest = output_dir / "build_manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate_count": len(rows), "failures": len(failures), "manifest": str(manifest.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

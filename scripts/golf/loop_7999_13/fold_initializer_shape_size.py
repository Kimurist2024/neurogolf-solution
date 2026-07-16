#!/usr/bin/env python3
"""Fold fixed Shape/Size/ConstantOfShape nodes into dense initializers.

The rewrite is exact by construction: an initializer's dimensions and number
of elements are independent of runtime input data.  Each folded node is
replaced by an int64 initializer with the same output name.  Candidates are
still required to pass the strict structural, runtime-shape, differential,
and fresh correctness gates before adoption.
"""

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
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parents[3]


def _attribute_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def fold(model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[dict[str, object]]]:
    result = copy.deepcopy(model)
    initializers = {tensor.name: tensor for tensor in result.graph.initializer}
    replacements: list[onnx.TensorProto] = []
    kept_nodes: list[onnx.NodeProto] = []
    changes: list[dict[str, object]] = []

    for index, node in enumerate(result.graph.node):
        source = initializers.get(node.input[0]) if node.input else None
        if source is None or node.op_type not in {"Shape", "Size", "ConstantOfShape"}:
            kept_nodes.append(node)
            continue

        source_shape = tuple(int(dimension) for dimension in source.dims)
        if node.op_type == "Shape":
            rank = len(source_shape)
            start = _attribute_int(node, "start", 0)
            end = _attribute_int(node, "end", rank)
            if start < 0:
                start += rank
            if end < 0:
                end += rank
            start = min(max(start, 0), rank)
            end = min(max(end, 0), rank)
            value = np.asarray(source_shape[start:end], dtype=np.int64)
        elif node.op_type == "Size":
            value = np.asarray(math.prod(source_shape) if source_shape else 1, dtype=np.int64)
        else:
            requested_shape = np.asarray(numpy_helper.to_array(source), dtype=np.int64).reshape(-1)
            if np.any(requested_shape < 0):
                raise ValueError("ConstantOfShape has a negative dimension")
            scalar = np.asarray(0.0, dtype=np.float32)
            for attribute in node.attribute:
                if attribute.name == "value":
                    scalar = np.asarray(numpy_helper.to_array(attribute.t)).reshape(())
            value = np.full(tuple(int(item) for item in requested_shape), scalar.item(), dtype=scalar.dtype)

        if len(node.output) != 1 or not node.output[0]:
            raise ValueError(f"task node {index} has unsupported outputs")
        output_name = node.output[0]
        if output_name in initializers:
            raise ValueError(f"folded output already exists as initializer: {output_name}")
        replacements.append(numpy_helper.from_array(value, name=output_name))
        changes.append(
            {
                "node_index": index,
                "op_type": node.op_type,
                "source": source.name,
                "source_shape": list(source_shape),
                "output": output_name,
                "value": value.reshape(-1).tolist(),
                "output_shape": list(value.shape),
            }
        )

    if not changes:
        return result, []
    del result.graph.node[:]
    result.graph.node.extend(kept_nodes)
    result.graph.initializer.extend(replacements)
    onnx.checker.check_model(result, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(result, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    return result, changes


def cost_of(path: Path) -> tuple[int, int, int]:
    from scripts.golf.rank_dir import cost_of as repository_cost

    memory, params, cost = repository_cost(str(path))
    return int(memory), int(params), int(cost)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="1-400")
    args = parser.parse_args()

    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks: set[int] = set()
    for part in args.tasks.split(","):
        if "-" in part:
            left, right = part.split("-", 1)
            tasks.update(range(int(left), int(right) + 1))
        elif part.strip():
            tasks.add(int(part))

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(baseline) as archive:
        for task in sorted(tasks):
            member = f"task{task:03d}.onnx"
            try:
                source = onnx.load_model_from_string(archive.read(member))
                candidate, changes = fold(source)
                if not changes:
                    continue
                with tempfile.TemporaryDirectory(prefix=f"shape_size_{task:03d}_") as tmp:
                    base_path = Path(tmp) / "base.onnx"
                    candidate_path = Path(tmp) / "candidate.onnx"
                    onnx.save(source, base_path)
                    onnx.save(candidate, candidate_path)
                    base_memory, base_params, base_cost = cost_of(base_path)
                    memory, params, cost = cost_of(candidate_path)
                row: dict[str, object] = {
                    "task": task,
                    "built": True,
                    "base_memory": base_memory,
                    "base_params": base_params,
                    "base_cost": base_cost,
                    "candidate_memory": memory,
                    "candidate_params": params,
                    "candidate_cost": cost,
                    "cost_reduction": base_cost - cost,
                    "projected_gain": math.log(base_cost / cost) if 0 < cost < base_cost else 0.0,
                    "changes": changes,
                }
                if cost < base_cost:
                    path = out_dir / member
                    onnx.save(candidate, path)
                    row["path"] = str(path.relative_to(ROOT))
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(row)
            except Exception as error:
                rows.append({"task": task, "built": False, "error": repr(error)})

    document = {
        "baseline": str(baseline.relative_to(ROOT)),
        "rows": rows,
        "strict_cost_winners": sum(
            bool(row.get("built")) and int(row.get("cost_reduction", 0)) > 0 for row in rows
        ),
        "projected_gain": sum(float(row.get("projected_gain", 0.0)) for row in rows),
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
